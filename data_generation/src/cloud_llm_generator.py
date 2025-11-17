from .data_models import GenRun
from .llm_generator import LLMGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudLLMGenerator(LLMGenerator):
    def generate(self, gen_specs: GenRun):
        pass
    
