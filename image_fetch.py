"""Fetch image URLs without allowing access to local or private networks."""

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlsplit

import httpx


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


def declared_length(headers) -> int | None:
    value = headers.get("content-length", "")
    try:
        return int(value) if value.isascii() and value.isdecimal() else None
    except ValueError:
        return None


async def fetch_image_from_url(
    url: str, *, max_bytes: int = 10 * 1024 * 1024,
    total_timeout: float = 15, idle_timeout: float = 5,
) -> bytes:
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

    try:
        async with asyncio.timeout(total_timeout):
            addresses = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
            if not addresses or any(
                family not in (socket.AF_INET, socket.AF_INET6)
                or not _public_address(sockaddr[0])
                for family, _, _, _, sockaddr in addresses
            ):
                raise ValueError("Image URL must point to a public internet address.")

            authority = f"[{host}]" if ":" in host else host
            if parsed.port is not None:
                authority += f":{port}"
            # Never resolve the hostname again, or consult environmental proxies.
            async with httpx.AsyncClient(
                timeout=idle_timeout, trust_env=False, follow_redirects=False,
            ) as client:
                for _, _, _, _, sockaddr in addresses:
                    destination = httpx.URL(url).copy_with(host=sockaddr[0], port=port)
                    try:
                        async with client.stream(
                            "GET", destination,
                            headers={
                                "Host": authority,
                                "User-Agent": "Mozilla/5.0 (compatible; FaceSwap/1.0)",
                                "Accept": "image/*,*/*", "Accept-Encoding": "identity",
                            },
                            # httpcore connects to the numeric origin, but TLS must
                            # send SNI and verify the certificate for the original host.
                            extensions={"sni_hostname": host},
                        ) as response:
                            if 300 <= response.status_code < 400:
                                raise ValueError("Image URL redirects are not allowed. Provide a direct image URL.")
                            if not 200 <= response.status_code < 300:
                                raise ValueError("Image URL did not return a successful response.")
                            if response.headers.get("content-encoding", "identity").lower() != "identity":
                                raise ValueError("Image URL returned an unsupported content encoding.")
                            size = declared_length(response.headers)
                            if size is not None and size > max_bytes:
                                raise ValueError(f"Downloaded image exceeds the limit of {max_bytes} bytes.")
                            data = bytearray()
                            async for chunk in response.aiter_raw():
                                if len(data) + len(chunk) > max_bytes:
                                    raise ValueError(f"Downloaded image exceeds the limit of {max_bytes} bytes.")
                                data.extend(chunk)
                            return bytes(data)
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        # Only retry addresses from the already validated DNS result.
                        continue
                raise ValueError("Could not fetch the image URL. Check the URL and try again.")
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise ValueError("Image URL download timed out.") from exc
    except (OSError, httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
        raise ValueError("Could not fetch the image URL. Check the URL and try again.") from exc
