import pytest
import json

@pytest.mark.gpu
def test_generate_questions(sample_config, tmp_path):
    from data_generation.wrappers.llm_generator_wrappers import generate_questions

    generate_questions(config_yaml=sample_config, output_dir=tmp_path)
    
    outfile = tmp_path / 'gen_questions.jsonl'
    assert outfile.exists()
    
    json_dict = json.loads(outfile.read_text().splitlines()[0])

    questions = json_dict["resp"]
    context_id = json_dict["sample_id"]
    
    # validate questions
    assert isinstance(questions, list), "Should be a list."
    assert len(questions) > 0, "Should be at least one question."
    assert all(q.endswith("?") for q in questions), "Questions should end with question mark."

    # validate context_id
    assert context_id.isdigit(), "Should be digits."
    assert len(context_id) == 6, "Should be filled to length 6."