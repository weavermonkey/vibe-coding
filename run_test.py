import logging
import sys

from dotenv import load_dotenv

from config import get_gemini_api_key


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main() -> int:
    _configure_logging()
    load_dotenv()
    get_gemini_api_key()

    import pytest

    logging.getLogger(__name__).info("Starting pytest run for all tests.")
    return pytest.main(["-x", "tests"])


if __name__ == "__main__":
    sys.exit(main())
