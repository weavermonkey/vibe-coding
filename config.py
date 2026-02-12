import logging
import os

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


load_dotenv()


def get_gemini_api_key() -> str:
    """
    Return the GEMINI_API_KEY from the environment.

    This function is the single source of truth for accessing the Gemini API key.
    It never reads the .env file directly; python-dotenv is used to populate
    environment variables before access.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        raise RuntimeError("GEMINI_API_KEY environment variable is required to use Gemini.")
    return api_key

