from .llm_generator import LLMGenerator
from .cloud_llm_generator import CloudLLMGenerator
from .local_llm_generator import LocalLLMGenerator
from .data_models import GenInit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_llm_generator(engine_specs: GenInit) -> LLMGenerator:
    if engine_specs.use_cloud:
        return CloudLLMGenerator(engine_specs)
    else:
        return LocalLLMGenerator(engine_specs)