import pytest
from textwrap import dedent
from data_generation.src.llm_response_processor import get_llm_response_processor


@pytest.mark.parametrize("text_type", ['questions', 'statements', 'answer', 'rating'])
def test_llm_response_processor_by_type(text_type):
    """Test the LLM response processor which processes the LLM output."""
    llm_response_processor = get_llm_response_processor(text_type)

    if text_type == 'questions':
        text = dedent("""\
                    1. Question one?
                    2) Question two?
                    
                    - Question three?
                    • Question four?
                    
                    Any more questions?
                    """)

        question_list = llm_response_processor(text)
        assert type(question_list) == list
        assert len(question_list) == 4
        assert question_list[0].endswith("?")
        assert all(q for q in question_list)  # no empty strings

    if text_type == 'statements':
        text = dedent("""Assistant: 
                    1. Statement one.
                    2) Statement two.
                    
                    - Statement three.
                    • Statement four.
                    
                    Any more statements, let me know.
                    """)
        statement_list = llm_response_processor(text)
        assert type(statement_list) == list
        assert len(statement_list) == 4
        assert statement_list[0].endswith(".")
        assert all(q for q in statement_list)  # no empty strings

    if text_type == 'answer':
        for text in ["Assistant: The answer to the question.\n",
                     "User :\nThe answer.",
                     "antwort:\nDie Antwort.",
                     "The answer to the question.\n"]:
            answer = llm_response_processor(text)
            assert type(answer) == str
            assert answer.endswith(".")
            assert len(text) > len(answer) > 1
    
    if text_type == 'rating':
        for text in ["Assistant: 1.\n",
                     "Rating :\n 0.",
                     "Rating:\n3",
                     "my rating: 2.\n"]:
            rating = llm_response_processor(text)
            assert type(rating) == str
            assert rating.isdigit()
            assert len(rating) == 1