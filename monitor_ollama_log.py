from __future__ import annotations

import logging
import msvcrt
import time
from pathlib import Path


POLL_INTERVAL_SECONDS = 60
OLLAMA_DIR = Path.home() / "AppData" / "Local" / "Ollama"
SERVER_LOG_PATH = OLLAMA_DIR / "server.log"
MONITOR_LOG_DIR = Path.cwd() / "ollama_log"
MONITOR_LOG_PATH = MONITOR_LOG_DIR / f"ollama_server_{time.strftime('%Y%m%d_%H%M%S')}.log"


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
            logger.info("Detected %d new bytes in %s:\n%s", len(data), SERVER_LOG_PATH.name, content)

        if wait_for_next_poll():
            logger.info("ESC pressed; exiting")
            break


if __name__ == "__main__":
    main()