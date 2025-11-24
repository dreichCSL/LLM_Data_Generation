import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_config_questions(test_data_dir):
    return test_data_dir / "sample_config_questions.yaml"

@pytest.fixture
def sample_config_answers(test_data_dir):
    return test_data_dir / "sample_config_answers.yaml"

@pytest.fixture
def sample_context(test_data_dir):
    return test_data_dir / "sample_context.jsonl"

@pytest.fixture
def sample_discourse_acts(test_data_dir):
    return test_data_dir / "sample_discourse_acts.jsonl"

