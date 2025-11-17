import json
from abc import ABC, abstractmethod
from typing import List, Dict
from .data_models import GenInit, GenRun

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMGenerator(ABC):
    def __init__(self, engine_specs: GenInit):
        self.engine_specs = engine_specs
        
    @abstractmethod
    def generate(self, gen_specs: GenRun):
        pass
    
    @staticmethod
    def _write_jsonl(output: List[Dict], output_file: str, write_mode='a'):
        """Write output as jsonl file."""
        with open(output_file, write_mode, encoding='utf-8') as f:
            for entry in output:
                f.write(json.dumps(entry) + "\n")