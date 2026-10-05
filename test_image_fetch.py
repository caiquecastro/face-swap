import asyncio
import importlib
import io
import socket
import ssl
import sys
import unittest
from unittest.mock import Mock, patch

from image_fetch import fetch_image_from_url


def answer(address, port=80):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    destination = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", destination)


class ImageFetchTests(unittest.TestCase):
    def setUp(self):
        self.resolver = patch("image_fetch.socket.getaddrinfo").start()
        self.resolver.return_value = [answer("8.8.8.8")]
        self.socket_factory = patch("image_fetch.socket.socket").start()
        self.sock = self.socket_factory.return_value
        self.sock.makefile.return_value = io.BytesIO(
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nimage"
        )
        self.addCleanup(patch.stopall)

    def test_public_http_pins_socket_and_preserves_request(self):
        with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:9000"}):
            self.assertEqual(fetch_image_from_url("http://example.com/photo?q=1"), b"image")
        self.resolver.assert_called_once_with("example.com", 80, type=socket.SOCK_STREAM)
        self.sock.connect.assert_called_once_with(("8.8.8.8", 80))
        sent = b"".join(call.args[0] for call in self.sock.sendall.call_args_list)
        self.assertIn(b"GET /photo?q=1 HTTP/1.1", sent)
        self.assertIn(b"Host: example.com", sent)
        self.sock.close.assert_called_once()

    def test_https_preserves_tls_hostname_and_certificate_verification(self):
        context = ssl.create_default_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        for suffix, port in (("", 443), (":80", 80), (":8443", 8443)):
            self.resolver.reset_mock()
            self.sock.reset_mock()
            self.resolver.return_value = [answer("8.8.8.8", port)]
            self.sock.makefile.return_value = io.BytesIO(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nimage"
            )
            with self.subTest(port=port), patch(
                "image_fetch.ssl.create_default_context", return_value=context
            ), patch.object(context, "wrap_socket", return_value=self.sock) as wrap:
                self.assertEqual(fetch_image_from_url(f"https://example.com{suffix}/photo"), b"image")
                wrap.assert_called_once_with(self.sock, server_hostname="example.com")
            self.resolver.assert_called_once_with("example.com", port, type=socket.SOCK_STREAM)
            self.sock.connect.assert_called_once_with(("8.8.8.8", port))
            sent = b"".join(call.args[0] for call in self.sock.sendall.call_args_list)
            self.assertIn(f"Host: example.com{suffix}\r\n".encode(), sent)

    def test_failed_address_falls_back_only_to_another_validated_address(self):
        self.resolver.return_value = [answer("8.8.8.8"), answer("1.1.1.1")]
        failed_socket = Mock()
        failed_socket.connect.side_effect = OSError()
        self.socket_factory.side_effect = [failed_socket, self.sock]
        self.assertEqual(fetch_image_from_url("http://example.com/photo"), b"image")
        failed_socket.close.assert_called_once()
        self.sock.connect.assert_called_once_with(("1.1.1.1", 80))
        self.resolver.assert_called_once()

    def test_tls_failure_closes_connected_socket(self):
        with patch("image_fetch.ssl.create_default_context") as context:
            context.return_value.wrap_socket.side_effect = ssl.SSLCertVerificationError()
            with self.assertRaisesRegex(ValueError, "Could not fetch"):
                fetch_image_from_url("https://example.com/photo")
        self.sock.close.assert_called_once()
        self.sock.sendall.assert_not_called()

    def test_invalid_urls_never_resolve_or_connect(self):
        for url in (
            "file:///etc/passwd", "ftp://example.com/photo", "data:image/png,foo",
            "//example.com/photo", "example.com/photo", "http:///photo",
            "http://user:pass@example.com/photo", "http://example.com:0/",
            "http://example.com:65536/", "http://example.com:abc/",
            "http://[::1", "http://[fe80::1%25eth0]/", "http://example.com\n/",
            "http://example.com\x00/", "http://exa mple.com/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetch_image_from_url(url)
        self.resolver.assert_not_called()
        self.socket_factory.assert_not_called()

    def test_nonpublic_literals_never_resolve_or_connect(self):
        for address in (
            "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1",
            "169.254.169.254", "0.0.0.0", "100.64.0.1", "192.0.0.8",
            "192.0.2.1", "198.18.0.1", "224.0.0.1", "240.0.0.1",
            "::", "::1", "fc00::1", "fe80::1", "fec0::1", "ff02::1",
            "2001:db8::1", "::ffff:127.0.0.1", "64:ff9b::7f00:1",
        ):
            host = f"[{address}]" if ":" in address else address
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "public"):
                fetch_image_from_url(f"http://{host}/photo")
        self.resolver.assert_not_called()
        self.socket_factory.assert_not_called()

    def test_private_dns_answers_and_unusual_ip_spellings_never_connect(self):
        for host, addresses in (
            ("example.com", ["10.0.0.1"]), ("example.com", ["::1"]),
            ("example.com", ["8.8.8.8", "169.254.169.254"]),
            ("example.com", ["8.8.8.8", "fec0::1"]),
            ("2130706433", ["127.0.0.1"]), ("127.1", ["127.0.0.1"]),
            ("0177.0.0.1", ["127.0.0.1"]), ("0x7f000001", ["127.0.0.1"]),
        ):
            self.resolver.return_value = [answer(address) for address in addresses]
            with self.subTest(host=host, addresses=addresses), self.assertRaisesRegex(ValueError, "public"):
                fetch_image_from_url(f"http://{host}/photo")
        self.socket_factory.assert_not_called()

    def test_dns_rebinding_cannot_change_the_connected_address(self):
        self.resolver.side_effect = [[answer("8.8.8.8")], [answer("127.0.0.1")]]
        self.assertEqual(fetch_image_from_url("http://example.com/photo"), b"image")
        self.resolver.assert_called_once()
        self.sock.connect.assert_called_once_with(("8.8.8.8", 80))

    def test_public_ipv6_and_mapped_public_addresses(self):
        for address in ("2606:4700:4700::1111", "::ffff:8.8.8.8"):
            self.resolver.return_value = [answer(address)]
            self.sock.makefile.return_value = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nimage")
            with self.subTest(address=address):
                self.assertEqual(fetch_image_from_url(f"http://[{address}]/photo"), b"image")
                self.sock.connect.assert_called_with((address, 80, 0, 0))

    def test_redirect_to_internal_address_is_not_followed(self):
        self.sock.makefile.return_value = io.BytesIO(
            b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/secret\r\nContent-Length: 0\r\n\r\n"
        )
        with self.assertRaisesRegex(ValueError, "direct image URL"):
            fetch_image_from_url("http://example.com/photo")
        self.resolver.assert_called_once()
        self.sock.connect.assert_called_once()
        self.assertTrue(self.sock.makefile.return_value.closed)

    def test_fetch_failures_become_value_errors_and_close_socket(self):
        for failure in (socket.timeout(), ssl.SSLCertVerificationError(), OSError()):
            with self.subTest(failure=failure):
                self.sock.sendall.side_effect = failure
                with self.assertRaisesRegex(ValueError, "Could not fetch"):
                    fetch_image_from_url("http://example.com/photo")
        self.sock.close.assert_called()
        self.sock.sendall.side_effect = None
        self.resolver.side_effect = socket.gaierror()
        with self.assertRaisesRegex(ValueError, "Could not fetch"):
            fetch_image_from_url("http://example.com/photo")

    def test_http_error_becomes_value_error(self):
        self.sock.makefile.return_value = io.BytesIO(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        with self.assertRaisesRegex(ValueError, "successful response"):
            fetch_image_from_url("http://example.com/photo")


class FormErrorTests(unittest.TestCase):
    def test_both_url_fields_return_clear_form_errors(self):
        # Import the real app/routes without loading or downloading ML models.
        from starlette.requests import Request
        from starlette.datastructures import UploadFile

        with patch.dict(sys.modules, {
            "cv2": Mock(), "numpy": Mock(), "insightface": Mock(), "insightface.app": Mock(),
        }):
            app = importlib.import_module("app")
        request = Request({"type": "http", "method": "POST", "path": "/swap", "headers": []})
        for field in ("source_url", "target_url"):
            options = dict(source_image=None, target_image=None, source_url=None, target_url=None)
            options[field] = "http://127.0.0.1/secret"
            other_image = "target_image" if field == "source_url" else "source_image"
            options[other_image] = UploadFile(io.BytesIO(b"image"), filename="photo.jpg")
            with self.subTest(field=field):
                response = asyncio.run(app.swap(request, **options))
                self.assertEqual(response.status_code, 400)
                self.assertIn(b"public internet address", response.body)
        sys.modules.pop("app", None)


if __name__ == "__main__":
    unittest.main()
