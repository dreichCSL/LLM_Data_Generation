from typing import Optional, Dict, Callable, Any
from pydantic import BaseModel, Field, ConfigDict
import torch

class GenRun(BaseModel):
    outfile: str
    llm_sampling_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    llm_response_processor: Callable[[str], Any] = lambda text: text
    # local gen specific
    dataloader: Optional[torch.utils.data.DataLoader] = None
    dataset: Optional[torch.utils.data.Dataset] = None
    # cloud gen specific
    infile_cloud_submission: Optional[str] = None
    # pydantic model settings
    model_config = ConfigDict(arbitrary_types_allowed=True)

class GenInit(BaseModel):
    model_name: str
    # cloud gen specific
    use_cloud: bool = False
    vllm_engine_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # pydantic model settings
    model_config = ConfigDict(arbitrary_types_allowed=True)