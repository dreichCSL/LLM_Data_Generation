import re
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_llm_response_processor(response_type: str):
    return LLMResponseProcessorRegistry.get(response_type)

class LLMResponseProcessorRegistry:
    _processors = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._processors[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._processors.get(name, lambda text: text)

@LLMResponseProcessorRegistry.register('questions')
def _(response_text: str) -> List[str]:
    """Extract questions (lines starting with digits and bullet points, ending with q-mark) """
    question_pattern = re.compile(r'^\s*(?:[\d\.\)\-•]+\s*)(?P<question>.+\?)\s*$')
    return [
        match.group('question').strip()
        for line in response_text.splitlines()
        if (match := question_pattern.match(line))
    ]

# TODO: add this
# @LLMResponseProcessorRegistry.register('statements')
# def _(response_text: str) -> List[str]:
#     """Extract numbered statements from text."""
#     statement_pattern = re.compile(r'^\s*(?:[\d\.\)\-•]+\s*)(?P<statement>.+\.)\s*$')
#     return [
#         match.group('statement').strip()
#         for line in response_text.splitlines()
#         if (match := statement_pattern.match(line))
#     ]

@LLMResponseProcessorRegistry.register('answer')
def _(response_text: str) -> str:
    """Extract answer from response (starting with a descriptor, ending with period.) """
    prefix_cleaner = re.compile(r'^(?:assistant|user|antwort)[\s:]+|^[^\s:]+\n', re.IGNORECASE)
    response_text = prefix_cleaner.sub('', response_text, count=1)
    return response_text.strip()

# TODO: add this
# @LLMResponseProcessorRegistry.register('rating')
# def _(response_text: str) -> str:
#     """Extract a numeric rating (0–9) from rating text."""
#     qrating_pattern = re.compile(r'(?:^assistant[\s:]*|rating[\s:]*)(\d)\b', re.IGNORECASE)
#     match = qrating_pattern.search(response_text)
#     return match.group(1) if match else response_text.strip()