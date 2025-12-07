"""
Pytest configuration for test setup.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client
