"""Fetch image URLs without allowing access to local or private networks."""

import http.client
import ipaddress
import re
import socket
import ssl
from urllib.parse import urlsplit


def _public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (
        ip.is_global
        and not ip.is_multicast
        and not ip.is_reserved
        and not getattr(ip, "is_site_local", False)
    )


def fetch_image_from_url(url: str) -> bytes:
    if any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ValueError("Image URL must not contain whitespace or control characters.")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        if parsed.scheme not in ("http", "https") or not host:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or "%" in host:
            raise ValueError
        host = host.encode("idna").decode("ascii")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            if ":" in host or not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
                raise ValueError
            literal = None
        port = port if port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Provide an absolute HTTP or HTTPS image URL without credentials.") from exc

    if literal is not None and not _public_address(str(literal)):
        raise ValueError("Image URL must point to a public internet address.")

    connection = http.client.HTTPConnection(host, port, timeout=15)
    connection.default_port = 443 if parsed.scheme == "https" else 80
    # Never let http.client reconnect by resolving the hostname again.
    connection.auto_open = False
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if not addresses or any(
            family not in (socket.AF_INET, socket.AF_INET6)
            or not _public_address(sockaddr[0])
            for family, _, _, _, sockaddr in addresses
        ):
            raise ValueError("Image URL must point to a public internet address.")

        # All DNS answers must pass before any connection is attempted.
        for family, kind, protocol, _, sockaddr in addresses:
            sock = socket.socket(family, kind, protocol)
            try:
                sock.settimeout(15)
                sock.connect(sockaddr)
            except OSError:
                sock.close()
                continue
            connection.sock = sock
            break
        if connection.sock is None:
            raise OSError("No reachable public address")
        if parsed.scheme == "https":
            connection.sock = ssl.create_default_context().wrap_socket(
                connection.sock, server_hostname=host
            )
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request(
            "GET",
            path,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FaceSwap/1.0)",
                "Accept": "image/*,*/*",
            },
        )
        with connection.getresponse() as response:
            if 300 <= response.status < 400:
                raise ValueError("Image URL redirects are not allowed. Provide a direct image URL.")
            if not 200 <= response.status < 300:
                raise ValueError("Image URL did not return a successful response.")
            return response.read()
    except (OSError, http.client.HTTPException, UnicodeError) as exc:
        raise ValueError("Could not fetch the image URL. Check the URL and try again.") from exc
    finally:
        connection.close()
