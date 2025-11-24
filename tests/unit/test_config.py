import pytest
from data_generation.conf.config import load_config

@pytest.mark.parametrize("config_type", 
                         ["sample_config_questions", 
                          "sample_config_answers"])
def test_config(config_type, request):
    load_config(request.getfixturevalue(config_type))