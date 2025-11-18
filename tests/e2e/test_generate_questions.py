import pytest
import json
from pathlib import Path
import tempfile

@pytest.mark.gpu
def test_generate_questions():
    from data_generation.wrappers.llm_generator_wrappers import generate_questions
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_questions(config_yaml='tests/data/sample_config.yaml', output_dir=tmp_dir)

        output_path = Path(tmp_dir)
        outfile = output_path / 'gen_questions.jsonl'
        assert outfile.exists()
        
        with open(outfile, 'r', encoding='utf-8') as f:
            json_dict = json.loads(f.readline())
            questions = json_dict["resp"]
            context_id = json_dict["sample_id"]
            assert type(questions) == list
            assert len(questions) > 0
            assert questions[0].endswith("?")
            assert all(q for q in questions)
            assert context_id.isdigit()
            assert len(context_id) == 6