"""
conftest.py
-----------
Pytest configuration for the SA Credit Stress Monitor test suite.

Loads model artefacts once before the session starts, so TestClient
requests find the model already in memory (mirrors production startup).
"""

import pytest
from src.api import model_loader


@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Load model artefacts once for the entire test session."""
    model_loader.load_model_artefacts()
