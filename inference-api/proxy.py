#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import jwt
import requests
from inference_config import inference_config
from jwt import InvalidTokenError

HTTP_UNAUTHORIZED = 401
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hostname = f"http://127.0.0.1:{inference_config.reverse_proxy_port}"
    jwt_secret = None

    # pylint: disable=invalid-name
    def do_POST(self):
        try:
            jwt_payload = self.read_authorization()
            if jwt_payload is not None:
                start_time = time.time()
                url = f"{self.hostname}{self.path}"
                fwd_headers = self.get_fwd_headers()
                content_len = int(self.headers.get("Content-Length", 0))
                content = self.rfile.read(content_len)
                response = requests.post(
                    url,
                    data=content,
                    headers=fwd_headers,
                    verify=False,
                    stream=True,
                    timeout=60,
                )
                stats = None
                self.send_response(response.status_code)
                if response.headers.get("transfer-encoding") == "chunked":
                    self.send_resp_headers(response)
                    # writing chunks
                    # note: chunks-size must be in hex format, chunk-data in bytes
                    # see: https://httpwg.org/specs/rfc7230.html#chunked.encoding
                    # use iter_content chunk_size=None to read each chunk immediately
                    for chunk in response.iter_content(
                        chunk_size=None, decode_unicode=False
                    ):
                        chunk = (
                            "{0:x}\r\n".format(len(chunk))
                            + chunk.decode("utf-8")
                            + "\r\n"
                        )
                        self.wfile.write(chunk.encode(encoding="utf-8"))
                    # writing close sequence
                    close_chunk = "0\r\n\r\n"
                    self.wfile.write(close_chunk.encode(encoding="utf-8"))
                else:
                    try:
                        json_response = json.loads(response.content)
                        stats = json_response.pop("__stats", None)
                        response_bytes = bytes(
                            json.dumps(
                                json_response,
                                # Compact JSON by dropping all spaces
                                separators=(",", ":"),
                            ),
                            "utf-8",
                        )
                        # set content-length before calling self.send_resp_headers()
                        response.headers["content-length"] = str(len(response_bytes))
                        self.send_resp_headers(response)
                        self.wfile.write(response_bytes)
                    # pylint: disable=broad-except
                    except Exception as exc:
                        self.log_error(f"Exception: {exc}")
                        self.send_error(HTTP_BAD_REQUEST)

                self.wfile.flush()
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)
                self.log_transaction(
                    response.status_code, jwt_payload, stats, duration_ms
                )
            else:
                self.send_error(HTTP_UNAUTHORIZED)

        # pylint: disable=broad-except
        except Exception as exc:
            self.log_error(f"Exception: {exc}")
            try:
                self.send_error(HTTP_INTERNAL_SERVER_ERROR)
            except BrokenPipeError as bpe:
                self.log_error(f"BrokenPipeError: {bpe}")

    def log_transaction(self, status_code, jwt_payload, stats, duration_ms):
        self.log_message(
            "[Transaction] path=%s client_address=%s status_code=%s jwt=%s stats=%s duration_ms=%s",  # noqa: E501
            self.path,
            self.client_address,
            status_code,
            jwt_payload,
            stats,
            duration_ms,
        )

    def read_authorization(self) -> Optional[dict]:
        [scheme, parameters] = normalize_token(self.headers.get("authorization", ""))
        if scheme != "bearer":
            self.log_error(f"Authorization scheme was '{scheme}' instead of bearer")
            return None
        try:
            payload = jwt.decode(parameters, self.jwt_secret, algorithms=["HS256"])
            return payload
        except InvalidTokenError as exc:
            self.log_error(f"JWT payload decode error: {exc}")
            return None

    def get_fwd_headers(self):
        fwd_headers = {}
        for key in self.headers:
            fwd_headers[key] = self.headers[key]
        fwd_headers.update({"Host": self.hostname})
        return fwd_headers

    def send_resp_headers(self, resp):
        for key in resp.headers:
            self.send_header(key, resp.headers[key])
        self.end_headers()


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy HTTP requests")
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=inference_config.reverse_proxy_port,
        help="serve HTTP requests on specified port.",
    )
    parser.add_argument(
        "--hostname",
        dest="hostname",
        type=str,
        default=f"http://127.0.0.1:{inference_config.backend_server_port}",
        help="hostname to send requests to.",
    )
    args = parser.parse_args()
    return args


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def normalize_token(token) -> [str, str]:
    """
    Note that scheme is case insensitive for the authorization header.
    See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization#directives
    """  # noqa: E501
    one_space = " "
    words = token.split(one_space)
    scheme = words[0].lower()
    return [scheme, " ".join(words[1:])]


def main():
    try:
        jwt_secret = os.environ["JWT_SECRET"]
    except KeyError:
        print("ERROR: Expected JWT_SECRET environment variable to be provided")
        sys.exit(1)

    args = parse_args()
    ProxyHTTPRequestHandler.hostname = args.hostname
    ProxyHTTPRequestHandler.jwt_secret = jwt_secret

    route_entry = "0.0.0.0"
    print(f"http server is starting on {route_entry} port {args.port}...")
    server_address = (route_entry, args.port)
    httpd = ThreadedHTTPServer(server_address, ProxyHTTPRequestHandler)
    print(f"http server is running as reverse proxy to {args.hostname}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("http server stopped.")


if __name__ == "__main__":
    main()
