"""
Test configuration and fixtures.
"""

import pytest


@pytest.fixture
def sample_data():
    """Sample data fixture for tests."""
    return {"key": "value", "number": 42}


@pytest.fixture(scope="session")
def session_config():
    """Session-scoped configuration fixture."""
    return {"test_mode": True}


# Configure test environment
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (can be skipped)")
