import os
from torch.utils.data import DataLoader
from data_generation.src.llm_generator_factory import get_llm_generator
from data_generation.src.llm_response_processor import get_llm_response_processor
from data_generation.src.data_models import (
    GenRun, 
    GenInit
    )
from data_generation.src.datasets import TextConditioned
from data_generation.conf.config import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_generation(parameter_dict, cfg):
    # include parameters
    parameter_dict.update({'llm_sampling_params': cfg.llm_gen_config.llm_sampling_params.model_dump(),
                           'vllm_engine_params': cfg.llm_gen_config.vllm_engine_params.model_dump(),
                           'model_name': cfg.llm_gen_config.model_setup.model_name,
                           'use_cloud': cfg.llm_gen_config.model_setup.use_cloud})

    # init generation
    model_specs = GenInit(**parameter_dict)
    generator = get_llm_generator(model_specs)
    
    # run generation
    gen_specs = GenRun(**parameter_dict)
    logger.info("Generating with LLM.")
    generator.generate(gen_specs)
    del(generator)

def generate_questions(config_yaml, output_dir):
    # config
    cfg = load_config(config_yaml)
    
    # output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gen_questions.jsonl")

    # init dataset and dataloader
    dataset = TextConditioned(cfg.input_paths.context_file, cfg.llm_gen_config.model_setup.prompt_templates.model_dump())
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.llm_gen_config.batch_size_dl,
        shuffle=False,
        drop_last=False
    )

    llm_response_processor = get_llm_response_processor("questions")

    # additional parameters
    parameter_dict = {'outfile': output_file,
                      'dataset': dataset,
                      'dataloader': dataloader,
                      'llm_response_processor': llm_response_processor}

    run_generation(parameter_dict, cfg)