from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template
import yaml, pathlib

class BaseConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

class LLMSamplingParams(BaseConfig):
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    skip_special_tokens: bool = True

class VLLMEngineParams(BaseConfig):
    # https://docs.vllm.ai/en/v0.4.2/models/engine_args.html
    max_seq_len_to_capture: int = 6144  # should be le max_model_len; CUDA graph
    max_model_len: int = 6144  # max seq length (input + output tokens); kv cache
    tensor_parallel_size: int = 1  # number of GPUs to use
    compilation_config: dict = Field(default_factory=dict)

class Prompts(BaseConfig):
    user_prompt: str = ""
    system_prompt: str = ""
    chat_template: str = ""

class PromptTemplates(BaseConfig):
    user_prompt: Template = Template("")
    system_prompt: Template = Template("")
    chat_template: Optional[Template] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ModelSetup(BaseConfig):
    model_name: str
    use_cloud: bool = False
    prompts: Prompts
    prompt_templates: Optional[PromptTemplates] = None

class LLMGenConfig(BaseConfig):
    model_setup: ModelSetup
    batch_size: int = 16   # in vLLM: # of parallel processing "lanes"
    batch_size_dl: int = 128  # samples passed to vLLM (should be gt batch_size)
    llm_sampling_params: LLMSamplingParams = LLMSamplingParams()
    vllm_engine_params: VLLMEngineParams = VLLMEngineParams()

class InputPaths(BaseConfig):
    prompt_dir: str = "data_generation/prompts"
    context_file: str = ""

class AppConfig(BaseConfig):
    input_paths: InputPaths
    llm_gen_config: LLMGenConfig
    
def load_config(path: str) -> AppConfig:
    data = yaml.safe_load(pathlib.Path(path).read_text())
    cfg = AppConfig.model_validate(data)
    # set additional vllm parameter
    cfg.llm_gen_config.vllm_engine_params.compilation_config = \
        {"cudagraph_capture_sizes": [cfg.llm_gen_config.batch_size]}
    # load jinja2 templates
    env = Environment(loader=FileSystemLoader(cfg.input_paths.prompt_dir))
    prompt_templates = PromptTemplates(
        **{k: env.get_template(v) for k,v in 
         vars(cfg.llm_gen_config.model_setup.prompts).items()})
    cfg.llm_gen_config.model_setup.prompt_templates = prompt_templates
    return cfg