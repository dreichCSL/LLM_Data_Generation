import pytest
import json

@pytest.mark.gpu
def test_generate_answers(sample_config_answers, tmp_path):
    from data_generation.wrappers.llm_generator_wrappers import generate_answers

    generate_answers(config_yaml=sample_config_answers, output_dir=tmp_path)
    
    outfile = tmp_path / 'gen_answers.jsonl'
    assert outfile.exists()
    
    json_dict = json.loads(outfile.read_text().splitlines()[0])

    answer = json_dict["resp"]
    sample_id = json_dict["sample_id"]
    
    # validate answers
    assert isinstance(answer, str), "Should be a string."
    assert len(answer) > 0, "Should not be an empty string."
    assert answer.endswith("."), "Answers should end with period."

    # validate context_id
    assert len(sample_id.split('_')) == 2, "Should consist of two digit strings."
    assert sample_id.split('_')[0].isdigit(), "Should be digits."
    assert sample_id.split('_')[1].isdigit(), "Should be digits."
    assert len(sample_id) == 6+4+1, "Should be filled to length 6 (context) + 4 (counter) + 1 (underscore)."