import pytest
from data_generation.src.datasets import TextConditioned, TextAndDiscourseConditioned
from data_generation.conf.config import load_config

@pytest.mark.parametrize("config_type, dataset_type", 
                         [
                             ("sample_config_questions", "TextConditioned"),
                             ("sample_config_answers", "TextAndDiscourseConditioned")
                         ])
def test_dataset(config_type, dataset_type, sample_context, sample_discourse_acts, request):
    cfg = load_config(request.getfixturevalue(config_type))
    cfg.input_paths.context_file = sample_context
    if dataset_type == 'TextConditioned':
        # not using a chat template (typically cloud llm)
        dataset = TextConditioned(cfg.input_paths.context_file, 
                                  cfg.llm_gen_config.model_setup.prompt_templates.model_dump(
                                      exclude={"chat_template"}))
        context_id, prompt = next(iter(dataset))
        # validate context_id
        assert context_id.isdigit()
        assert len(context_id) == 6
        # validate prompt
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0]['role'] == 'system'
        assert prompt[0]['content']
        assert prompt[1]['role'] == 'user'
        assert prompt[1]['content']

        # when using a chat template (typically local llm)
        dataset = TextConditioned(cfg.input_paths.context_file, 
                                  cfg.llm_gen_config.model_setup.prompt_templates.model_dump())
        context_id, prompt = next(iter(dataset))
        # validate context_id
        assert context_id.isdigit()
        assert len(context_id) == 6
        # validate prompt
        assert isinstance(prompt, str)
        assert prompt, "Should not be empty."

    elif dataset_type == 'TextAndDiscourseConditioned':
        cfg.input_paths.discourse_file = sample_discourse_acts
        # not using a chat template (typically cloud llm)
        dataset = TextAndDiscourseConditioned(cfg.input_paths.context_file, 
                                              cfg.input_paths.discourse_file,
                                              cfg.llm_gen_config.model_setup.prompt_templates.model_dump(
                                                  exclude={"chat_template"}))
        
        context_id, prompt = next(iter(dataset))
        # validate context_id
        assert len(context_id.split('_')) == 2, "Should consist of two digit strings."
        assert context_id.split('_')[0].isdigit(), "Should be digits."
        assert context_id.split('_')[1].isdigit(), "Should be digits."
        assert len(context_id) == 6+4+1, "Should be filled to length 6 (context) + 4 (counter) + 1 (underscore)."
        # validate prompt
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0]['role'] == 'system'
        assert prompt[0]['content']
        assert prompt[1]['role'] == 'user'
        assert prompt[1]['content']

        # when using a chat template (typically local llm)
        dataset = TextAndDiscourseConditioned(cfg.input_paths.context_file, 
                                              cfg.input_paths.discourse_file,
                                              cfg.llm_gen_config.model_setup.prompt_templates.model_dump())
        context_id, prompt = next(iter(dataset))
        # validate context_id
        assert len(context_id.split('_')) == 2, "Should consist of two digit strings."
        assert context_id.split('_')[0].isdigit(), "Should be digits."
        assert context_id.split('_')[1].isdigit(), "Should be digits."
        assert len(context_id) == 6+4+1, "Should be filled to length 6 (context) + 4 (counter) + 1 (underscore)."
        # assert context_id.isdigit()
        # assert len(context_id) == 6
        # validate prompt
        assert isinstance(prompt, str)
        assert prompt, "Should not be empty."

