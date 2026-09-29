import argparse
import http.client
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


LOGGER = logging.getLogger("ollama_proxy")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization", "set-cookie", "x-api-key"}


def parse_upstream(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("upstream must be an http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("upstream URL must not contain a query or fragment")
    return value.rstrip("/")


def filtered_headers(headers, extra_hop_by_hop=()):
    connection_tokens = set(extra_hop_by_hop)
    for value in headers.get_all("Connection", []):
        connection_tokens.update(token.strip().lower() for token in value.split(","))

    excluded = HOP_BY_HOP_HEADERS | connection_tokens
    return [
        (name, value)
        for name, value in headers.items()
        if name.lower() not in excluded and name.lower() != "host"
    ]


def logged_headers(headers):
    return {
        name: "[REDACTED]" if name.lower() in SENSITIVE_HEADERS else value
        for name, value in headers.items()
    }


class OllamaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OllamaProxy/1.0"

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def _read_request_body(self):
        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower().strip()
        if transfer_encoding:
            if transfer_encoding != "chunked":
                raise ValueError("unsupported Transfer-Encoding")
            return self._read_chunked_body()

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return b""
        length = int(content_length)
        if length < 0:
            raise ValueError("invalid Content-Length")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("incomplete request body")
        return body

    def _read_chunked_body(self):
        body = bytearray()
        while True:
            line = self.rfile.readline(8192)
            if not line:
                raise ValueError("incomplete chunked request body")
            try:
                chunk_size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError as error:
                raise ValueError("invalid chunk size") from error

            if chunk_size == 0:
                while True:
                    trailer = self.rfile.readline(8192)
                    if not trailer:
                        raise ValueError("incomplete chunked request trailers")
                    if trailer in (b"\r\n", b"\n"):
                        return bytes(body)

            chunk = self.rfile.read(chunk_size)
            if len(chunk) != chunk_size or self.rfile.read(2) != b"\r\n":
                raise ValueError("incomplete chunked request body")
            body.extend(chunk)

    def _body_for_log(self, body, total_length=None):
        if total_length is None:
            total_length = len(body)
        if not self.server.log_bodies:
            return f"<{total_length} bytes; body logging disabled>"
        limit = self.server.max_log_body_bytes
        preview = repr(body[:limit])
        if total_length > len(body) or len(body) > limit:
            return f"{preview} <truncated; {total_length} bytes total>"
        return preview

    def _proxy(self):
        started = time.monotonic()
        try:
            request_body = self._read_request_body()
        except (ValueError, OverflowError) as error:
            LOGGER.warning("invalid request from %s: %s", self.client_address[0], error)
            self.send_error(400, str(error))
            return

        request_headers = filtered_headers(self.headers, {"expect"})
        LOGGER.info(
            "request %s %s headers=%s body=%s",
            self.command,
            self.path,
            logged_headers(self.headers),
            self._body_for_log(request_body),
        )

        upstream = urlsplit(self.server.upstream_url)
        connection_type = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        port = upstream.port or (443 if upstream.scheme == "https" else 80)
        connection = connection_type(upstream.hostname, port, timeout=self.server.timeout)
        upstream_path = f"{upstream.path.rstrip('/')}{self.path}"
        if not upstream_path:
            upstream_path = "/"

        try:
            connection.request(self.command, upstream_path, body=request_body, headers=dict(request_headers))
            response = connection.getresponse()
        except (OSError, http.client.HTTPException, ValueError) as error:
            LOGGER.exception("upstream request failed for %s %s", self.command, self.path)
            connection.close()
            self.send_error(502, f"Ollama upstream error: {error}")
            return

        response_body = bytearray()
        response_bytes = 0
        try:
            self.send_response(response.status, response.reason)
            for name, value in filtered_headers(response.headers):
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                response_bytes += len(chunk)
                if self.server.log_bodies:
                    remaining = self.server.max_log_body_bytes - len(response_body)
                    if remaining > 0:
                        response_body.extend(chunk[:remaining])
        except (OSError, http.client.HTTPException):
            LOGGER.exception("response forwarding failed for %s %s", self.command, self.path)
            self.close_connection = True
        finally:
            response.close()
            connection.close()
            LOGGER.info(
                "response %s %s status=%s bytes=%d duration_ms=%.1f body=%s",
                self.command,
                self.path,
                response.status,
                response_bytes,
                (time.monotonic() - started) * 1000,
                self._body_for_log(bytes(response_body), response_bytes)
                if self.server.log_bodies
                else "<body logging disabled>",
            )

    def log_message(self, format_string, *args):
        LOGGER.debug("%s - %s", self.client_address[0], format_string % args)


def main():
    parser = argparse.ArgumentParser(description="Log and forward requests to an Ollama API server.")
#   parser.add_argument("--listen-host", default="127.0.0.1", help="interface to listen on (default: 127.0.0.1)")
    parser.add_argument("--listen-host", default="0.0.0.0", help="interface to listen on (default: 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, default=11434, help="port to listen on (default: 11434)")
    parser.add_argument(
        "--upstream",
        type=parse_upstream,
        default="http://127.0.0.1:11435",
        help="Ollama base URL (default: http://127.0.0.1:11435)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="upstream timeout in seconds")
    parser.add_argument("--log-file", help="write logs to this file instead of stderr")
    parser.add_argument(
        "--log-body",
        action="store_true",
        help="include request and response body previews in logs (may contain private data)",
        default=False
    )
    parser.add_argument(
        "--max-log-body-bytes",
        type=int,
        default=4096,
        help="maximum bytes of each body to log when --log-body is enabled (default: 4096)",
    )
    args = parser.parse_args()
    if args.listen_port < 0 or args.listen_port > 65535:
        parser.error("--listen-port must be between 0 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_log_body_bytes < 1:
        parser.error("--max-log-body-bytes must be positive")

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), OllamaProxyHandler)
    server.daemon_threads = True
    server.upstream_url = args.upstream
    server.timeout = args.timeout
    server.log_bodies = args.log_body
    server.max_log_body_bytes = args.max_log_body_bytes

    LOGGER.info(
        "proxy listening on http://%s:%d -> %s",
        args.listen_host,
        server.server_address[1],
        args.upstream,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("proxy stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()