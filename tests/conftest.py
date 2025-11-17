import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_config(test_data_dir):
    return test_data_dir / "sample_config.yaml"

@pytest.fixture
def sample_context(test_data_dir):
    return test_data_dir / "sample_context.jsonl"
