from __future__ import annotations

import json
import logging
import msvcrt
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

IP_CHECK_DIC = {"TARGET_PC_NAME": ""}

POLL_INTERVAL_SECONDS = 60
OLLAMA_DIR = Path.home() / "AppData" / "Local" / "Ollama"
SERVER_LOG_PATH = OLLAMA_DIR / "server.log"
MONITOR_LOG_DIR = Path.cwd() / "ollama_log"
MONITOR_LOG_PATH = MONITOR_LOG_DIR / f"ollama_server_{time.strftime('%Y%m%d_%H%M%S')}.log"


LOGGED_JSON_PATTERN = re.compile(r'msg="logged to (.+?\.json)(?:[,"]|$)')

class RequestJsonLog:
    def __init__(self, path: Path):
        self.path = path
        self.model: str | None = None
        self.messages: list[dict] | None = None
        self.think: str | None = None
        self.tools: list[str] = []

    def parse(self, body: dict | None, logger: logging.Logger) -> dict | None:
        if self.model is not None and self.messages is not None and self.think is not None and self.tools is not None:
            return {"model": self.model, "messages": self.messages, "think": self.think, "tools": self.tools}
        if body is not None:
            self.model = body.get("model")
            self.messages = body.get("messages")
            self.think = body.get("think")
            tools_list = body.get("tools")
            self.tools = []
            for tool in tools_list:
                tool_type = tool.get("type") if isinstance(tool, dict) else None
                tool_body = tool.get(tool_type) if tool_type is not None else None
                tool_name = tool_body.get("name") if isinstance(tool_body, dict) else None
                if tool_name is not None:
                    self.tools.append(tool_name)
        return body


def should_log_line(line: str) -> bool:
    if line.startswith("time="):
        return True
    if not line.startswith("[GIN]"):
        return False

    quoted_parts = line.rsplit('"', 2)
    return len(quoted_parts) == 3 and urlsplit(quoted_parts[-2]).path == "/api/chat"


def extract_logged_json_path(line: str) -> Path | None:
    if not line.startswith("time="):
        return None
    match = LOGGED_JSON_PATTERN.search(line)
    if match is None:
        return None
    return Path(match.group(1).replace("\\\\", "\\"))


def parse_request_json(path: Path, logger: logging.Logger) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as source:
            body = json.load(source)
            req_json = RequestJsonLog(path)
            req_json.parse(body, logger)
            logger.info("Parsed request JSON from %s", path)
            logger.info("  model: %s", req_json.model)
            for reverse_msg in reversed(req_json.messages or []):
                if reverse_msg is None:
                    continue

                if reverse_msg.get("role") is None or reverse_msg.get("content") is None:
                    continue

                if reverse_msg.get("role") == "user":
                    logger.info("  message: %s", reverse_msg.get("content")) 
                    break
                
            logger.info("  think: %s", req_json.think)
            logger.info("  tools: %s", req_json.tools)

    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Failed to parse %s: %s", path, error)
        return None

    return req_json


def handle_logged_json(line: str, logger: logging.Logger) -> None:
    path = extract_logged_json_path(line)
    if path is None:
        return
    if not path.is_file():
        logger.warning("Logged JSON file not found: %s", path)
        return
    parse_request_json(path, logger)


def read_new_data(
    path: Path,
    offset: int,
    file_identity: tuple[int, int] | None,
) -> tuple[bytes, int, tuple[int, int]]:
    metadata = path.stat()
    current_identity = (metadata.st_dev, metadata.st_ino)

    if current_identity != file_identity or metadata.st_size < offset:
        offset = 0

    with path.open("rb") as source:
        source.seek(offset)
        data = source.read()

    return data, offset + len(data), current_identity


def wait_for_next_poll() -> bool:
    deadline = time.monotonic() + POLL_INTERVAL_SECONDS
    while time.monotonic() < deadline:
        if msvcrt.kbhit() and msvcrt.getwch() == "\x1b":
            return True
        time.sleep(0.1)
    return False


def update_ip_check_dic(logger: logging.Logger) -> None:
    for pc_name in IP_CHECK_DIC:
        try:
            addresses = socket.getaddrinfo(
                pc_name, None, family=socket.AF_INET, type=socket.SOCK_STREAM
            )
        except socket.gaierror as error:
            logger.warning("Could not resolve IPv4 address for %s: %s", pc_name, error)
            continue

        if not addresses:
            logger.warning("No IPv4 address found for %s", pc_name)
            continue

        address = addresses[0][4][0]
        previous_address = IP_CHECK_DIC[pc_name]
        IP_CHECK_DIC[pc_name] = address
        if address != previous_address:
            logger.info("Updated IPv4 address for %s: %s", pc_name, address)


def main() -> None:
    MONITOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(MONITOR_LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("server_log_monitor")

    try:
        metadata = SERVER_LOG_PATH.stat()
        offset = metadata.st_size
        file_identity = (metadata.st_dev, metadata.st_ino)
    except FileNotFoundError:
        offset = 0
        file_identity = None

    logger.info(
        "Monitoring %s every %d seconds (starting at byte %d); press ESC to exit",
        SERVER_LOG_PATH,
        POLL_INTERVAL_SECONDS,
        offset,
    )

    missing_reported = False
    while True:
        update_ip_check_dic(logger)
        try:
            data, offset, file_identity = read_new_data(
                SERVER_LOG_PATH, offset, file_identity
            )
            missing_reported = False
        except FileNotFoundError:
            offset = 0
            file_identity = None
            if not missing_reported:
                logger.warning("Waiting for %s to appear", SERVER_LOG_PATH)
                missing_reported = True
            if wait_for_next_poll():
                logger.info("ESC pressed; exiting")
                break
            continue

        if data:
            content = data.decode("utf-8", errors="replace").rstrip("\r\n")
            logged_lines = [line for line in content.splitlines() if should_log_line(line)]
            if logged_lines:
                logger.info(
                    "Detected %d new bytes in %s:\n%s",
                    len(data),
                    SERVER_LOG_PATH.name,
                    "\n".join(logged_lines),
                )
                for line in logged_lines:
                    handle_logged_json(line, logger)

        if wait_for_next_poll():
            logger.info("ESC pressed; exiting")
            break


if __name__ == "__main__":
    main()