import pytest
from data_generation.src.datasets import TextConditioned
from data_generation.conf.config import load_config


@pytest.mark.parametrize("process_type", ['questions'])
def test_dataset(sample_config, sample_context, process_type):
    cfg = load_config(sample_config)
    # cfg = load_config('tests/data/sample_config.yaml')
    cfg.input_paths.context_file = sample_context
    if process_type == 'questions':
        # no chat template / typically cloud llm
        dataset = TextConditioned(cfg.input_paths.context_file, 
                                  cfg.llm_gen_config.model_setup.prompt_templates.model_dump(
                                      exclude={"chat_template"}))
        context_id, prompt = next(iter(dataset))
        assert context_id.isdigit()
        assert len(context_id) == 6
        assert len(prompt) > 0
        assert type(prompt) == list
        assert len(prompt) == 2
        assert prompt[0]['role'] == 'system'
        assert prompt[0]['content']
        assert prompt[1]['role'] == 'user'
        assert prompt[1]['content']
        # with chat template / typically local llm
        dataset = TextConditioned(cfg.input_paths.context_file, 
                                  cfg.llm_gen_config.model_setup.prompt_templates.model_dump())
        context_id, prompt = next(iter(dataset))
        assert len(prompt) > 0
        assert type(prompt) == str
        assert prompt