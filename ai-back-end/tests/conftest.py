"""Shared pytest configuration."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env.local so integration tests can pick up OPENAI_API_KEY.
load_dotenv(Path(__file__).parent.parent / ".env.local")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that make real network calls (skipped when keys are unset)",
    )
