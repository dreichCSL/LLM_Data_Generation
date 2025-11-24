import pytest
import json

@pytest.mark.gpu
def test_generate_questions(sample_config_questions, tmp_path):
    from data_generation.wrappers.llm_generator_wrappers import generate_questions

    generate_questions(config_yaml=sample_config_questions, output_dir=tmp_path)
    
    outfile = tmp_path / 'gen_questions.jsonl'
    assert outfile.exists()
    
    json_dict = json.loads(outfile.read_text().splitlines()[0])

    question = json_dict["resp"]
    sample_id = json_dict["sample_id"]
    
    # validate questions
    assert isinstance(question, str), "Should be a string."
    assert len(question) > 0, "Should not be an empty string."
    assert question.endswith("?"), "Question should end with question mark."

    # validate sample_id
    assert len(sample_id.split('_')) == 2, "Should consist of two digit strings."
    assert sample_id.split('_')[0].isdigit(), "Should be digits."
    assert sample_id.split('_')[1].isdigit(), "Should be digits."
    assert len(sample_id) == 6+4+1, "Should be filled to length 6 (context) + 4 (counter) + 1 (underscore)."