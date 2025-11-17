from data_generation.conf.config import load_config

def test_config(sample_config):
    load_config(sample_config)

    # load_config('tests/data/sample_config.yaml')