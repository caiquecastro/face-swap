import io
import ipaddress
import socket
import ssl
import unittest
from unittest.mock import patch

import httpcore

from image_fetch import fetch_image_from_url


def answer(address, port=80):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    destination = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", destination)


class NetworkStream:
    """Mock only network I/O; exercise the real HTTPX/httpcore wire transport."""

    def __init__(self):
        self.response = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nimage"
        self.sent = []
        self.closed = False
        self.tls = []
        self.tls_failure = None
        self.write_failure = None
        self.read_failure = None

    async def read(self, max_bytes, timeout=None):
        if self.read_failure:
            raise self.read_failure
        response, self.response = self.response, b""
        return response

    async def write(self, data, timeout=None):
        if self.write_failure:
            raise self.write_failure
        self.sent.append(data)

    async def aclose(self):
        self.closed = True

    async def start_tls(self, **kwargs):
        self.tls.append(kwargs)
        if self.tls_failure:
            await self.aclose()  # The real AnyIO backend also closes on failed TLS.
            raise self.tls_failure
        return self

    def get_extra_info(self, info):
        return None


class ImageFetchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.resolver = patch("image_fetch.socket.getaddrinfo").start()
        self.resolver.return_value = [answer("8.8.8.8")]
        self.connect = patch("httpcore._backends.anyio.AnyIOBackend.connect_tcp").start()
        self.stream = NetworkStream()
        self.connect.return_value = self.stream
        self.addCleanup(patch.stopall)

    async def test_public_http_pins_socket_and_preserves_request(self):
        with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:9000", "HTTPS_PROXY": "http://127.0.0.1:9000", "ALL_PROXY": "http://127.0.0.1:9000"}):
            self.assertEqual(await fetch_image_from_url("http://example.com/photo?q=1"), b"image")
        self.resolver.assert_called_once_with("example.com", 80, 0, socket.SOCK_STREAM, 0, 0)
        self.assertEqual(self.connect.call_args.args[0], "8.8.8.8")
        self.assertEqual(self.connect.call_args.args[1], 80)
        sent = b"".join(self.stream.sent)
        self.assertIn(b"GET /photo?q=1 HTTP/1.1", sent)
        self.assertIn(b"Host: example.com", sent)
        self.assertTrue(self.stream.closed)

    async def test_https_preserves_tls_hostname_and_certificate_verification(self):
        for suffix, port in (("", 443), (":80", 80), (":8443", 8443)):
            self.resolver.reset_mock()
            self.connect.reset_mock()
            self.stream = NetworkStream()
            self.connect.return_value = self.stream
            self.resolver.return_value = [answer("8.8.8.8", port)]
            with self.subTest(port=port):
                self.assertEqual(await fetch_image_from_url(f"https://example.com{suffix}/photo"), b"image")
                tls = self.stream.tls[0]
                self.assertEqual(tls["server_hostname"], "example.com")
                self.assertTrue(tls["ssl_context"].check_hostname)
                self.assertEqual(tls["ssl_context"].verify_mode, ssl.CERT_REQUIRED)
            self.resolver.assert_called_once_with("example.com", port, 0, socket.SOCK_STREAM, 0, 0)
            self.assertEqual(self.connect.call_args.args[0], "8.8.8.8")
            self.assertEqual(self.connect.call_args.args[1], port)
            self.assertIn(f"Host: example.com{suffix}\r\n".encode(), b"".join(self.stream.sent))

    async def test_failed_address_falls_back_only_to_another_validated_address(self):
        self.resolver.return_value = [answer("8.8.8.8"), answer("1.1.1.1")]
        self.connect.side_effect = [httpcore.ConnectError("unreachable"), self.stream]
        self.assertEqual(await fetch_image_from_url("http://example.com/photo"), b"image")
        self.assertEqual([call.args[0] for call in self.connect.call_args_list], ["8.8.8.8", "1.1.1.1"])
        self.resolver.assert_called_once()
        self.assertTrue(self.stream.closed)

    async def test_tls_failure_closes_connected_socket(self):
        self.stream.tls_failure = httpcore.ConnectError("certificate verification failed")
        with self.assertRaisesRegex(ValueError, "Could not fetch"):
            await fetch_image_from_url("https://example.com/photo")
        self.assertTrue(self.stream.closed)
        self.assertEqual(self.stream.sent, [])

    async def test_invalid_urls_never_resolve_or_connect(self):
        for url in (
            "file:///etc/passwd", "ftp://example.com/photo", "data:image/png,foo",
            "//example.com/photo", "example.com/photo", "http:///photo",
            "http://user:pass@example.com/photo", "http://example.com:0/",
            "http://example.com:65536/", "http://example.com:abc/",
            "http://[::1", "http://[fe80::1%25eth0]/", "http://example.com\n/",
            "http://example.com\x00/", "http://exa mple.com/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                await fetch_image_from_url(url)
        self.resolver.assert_not_called()
        self.connect.assert_not_called()

    async def test_nonpublic_literals_never_resolve_or_connect(self):
        for address in (
            "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1",
            "169.254.169.254", "0.0.0.0", "100.64.0.1", "192.0.0.8",
            "192.0.2.1", "198.18.0.1", "224.0.0.1", "240.0.0.1",
            "::", "::1", "fc00::1", "fe80::1", "fec0::1", "ff02::1",
            "2001:db8::1", "::ffff:127.0.0.1", "64:ff9b::7f00:1",
        ):
            host = f"[{address}]" if ":" in address else address
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "public"):
                await fetch_image_from_url(f"http://{host}/photo")
        self.resolver.assert_not_called()
        self.connect.assert_not_called()

    async def test_private_dns_answers_and_unusual_ip_spellings_never_connect(self):
        for host, addresses in (
            ("example.com", ["10.0.0.1"]), ("example.com", ["::1"]),
            ("example.com", ["8.8.8.8", "169.254.169.254"]),
            ("example.com", ["8.8.8.8", "fec0::1"]),
            ("2130706433", ["127.0.0.1"]), ("127.1", ["127.0.0.1"]),
            ("0177.0.0.1", ["127.0.0.1"]), ("0x7f000001", ["127.0.0.1"]),
        ):
            self.resolver.return_value = [answer(address) for address in addresses]
            with self.subTest(host=host, addresses=addresses), self.assertRaisesRegex(ValueError, "public"):
                await fetch_image_from_url(f"http://{host}/photo")
        self.connect.assert_not_called()

    async def test_dns_rebinding_cannot_change_the_connected_address(self):
        self.resolver.side_effect = [[answer("8.8.8.8")], [answer("127.0.0.1")]]
        self.assertEqual(await fetch_image_from_url("http://example.com/photo"), b"image")
        self.resolver.assert_called_once()
        self.assertEqual(self.connect.call_args.args[0], "8.8.8.8")

    async def test_public_ipv6_and_mapped_public_addresses(self):
        for address in ("2606:4700:4700::1111", "::ffff:8.8.8.8"):
            self.resolver.return_value = [answer(address)]
            self.stream = NetworkStream()
            self.connect.return_value = self.stream
            with self.subTest(address=address):
                self.assertEqual(await fetch_image_from_url(f"http://[{address}]:8080/photo"), b"image")
                self.assertEqual(ipaddress.ip_address(self.connect.call_args.args[0]), ipaddress.ip_address(address))
                self.assertEqual(self.connect.call_args.args[1], 8080)
                self.assertIn(f"Host: [{address}]:8080\r\n".encode(), b"".join(self.stream.sent))

    async def test_redirect_to_internal_address_is_not_followed(self):
        self.stream.response = b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/secret\r\nContent-Length: 0\r\n\r\n"
        with self.assertRaisesRegex(ValueError, "direct image URL"):
            await fetch_image_from_url("http://example.com/photo")
        self.resolver.assert_called_once()
        self.connect.assert_called_once()
        self.assertTrue(self.stream.closed)

    async def test_fetch_failures_become_value_errors_and_close_socket(self):
        for failure in (socket.timeout(), ssl.SSLCertVerificationError(), httpcore.ReadError()):
            self.stream = NetworkStream()
            self.connect.return_value = self.stream
            if isinstance(failure, httpcore.ReadError):
                self.stream.read_failure = failure
            else:
                self.stream.write_failure = failure
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                await fetch_image_from_url("http://example.com/photo")
            self.assertTrue(self.stream.closed)
        self.resolver.side_effect = socket.gaierror()
        with self.assertRaisesRegex(ValueError, "Could not fetch"):
            await fetch_image_from_url("http://example.com/photo")

    async def test_http_error_becomes_value_error(self):
        self.stream.response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        with self.assertRaisesRegex(ValueError, "successful response"):
            await fetch_image_from_url("http://example.com/photo")



class FormErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_url_fields_return_clear_form_errors(self):
        from starlette.requests import Request
        from starlette.datastructures import UploadFile
        from test_limits import app  # Shared import replaces only pretrained models.

        request = Request({"type": "http", "method": "POST", "path": "/swap", "headers": []})
        for field in ("source_url", "target_url"):
            options = dict(source_image=None, target_image=None, source_url=None, target_url=None)
            options[field] = "http://127.0.0.1/secret"
            other_image = "target_image" if field == "source_url" else "source_image"
            options[other_image] = UploadFile(io.BytesIO(b"image"), filename="photo.jpg")
            with self.subTest(field=field):
                response = await app.swap(request, **options)
                self.assertEqual(response.status_code, 400)
                self.assertIn(b"public internet address", response.body)
            await options[other_image].close()


if __name__ == "__main__":
    unittest.main()
