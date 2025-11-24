from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
from typing import List, Dict, Union
from .data_models import GenInit, GenRun
from .llm_generator import LLMGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

    
class LocalLLMGenerator(LLMGenerator):
    def __init__(self, engine_specs: GenInit):
        super().__init__(engine_specs)
        """
        Initialize the text generator with GenInit model parameters containing
        model_name (HuggingFace model name/path) and llm_sampling_params 
        (e.g., temperature, max token count, etc.)
        """
        self.model_name = self.engine_specs.model_name
        self.vllm_engine_params = self.engine_specs.vllm_engine_params
        if not self.vllm_engine_params:
            raise ValueError("vllm_model_params is missing for vLLM-based text generation. Exiting.")

        # Init model and tokenizer and put on device(s) depending on model size
        # if self.vllm_model_params.get('tensor_parallel_size') is None:

        logger.info(f"Loading model {self.model_name} on {self.vllm_engine_params['tensor_parallel_size']} GPU(s)...")

        # self.model = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
        self.model = LLM(model=self.model_name, **self.vllm_engine_params)
        # get model's dtype with: model.llm_engine.model_config.dtype
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = self.model.get_tokenizer()

        # Padding should be on the left when generating text
        self.tokenizer.padding_side = 'left'
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.model.set_tokenizer(self.tokenizer)

        # Influence the LLM's output
        self.llm_sampling_params = SamplingParams()  #SamplingParams(**llm_sampling_params)
        # {"max_tokens"=512,
        # "temperature"=1.3}
        self.outfile_keys = {}  # {'id_name': 'id', 'out_name': 'output'}
        self.output_file = None

    def generate(self, gen_specs: GenRun):
        """
        Process prompts and write LLM output to disk
        
        :param gen_specs: GenRun dataclass with params for inference
        """
        self.gen_specs = gen_specs
        if not self.gen_specs.dataloader:
            raise ValueError("Dataloader needed for vLLM-based text generation. Exiting.")
        outfile = self.gen_specs.outfile
        # remove output file if exists before starting
        if os.path.exists(outfile):
            logger.info(f"Removing already existing outfile.")
            os.remove(outfile)
        self._set_llm_sampling_params(self.gen_specs.llm_sampling_params)
        
        total_batches = len(self.gen_specs.dataloader)
        logger.info(f"Processing {total_batches} batches. Writing every 10 batches to {outfile}")
        
        output = []
        for b_nr, batched_data in enumerate(tqdm(self.gen_specs.dataloader, desc="Processing batches."), 1):
            output.extend(self._process_batch(batched_data))

            # Write in 10 batch increments or if final step
            if b_nr % 10 == 0 or b_nr == total_batches:
                self._write_jsonl(output, outfile, write_mode='a')
                output = []

        logger.info(f"Completed generation.")
        return

    def _process_batch(self, batched_data):
        "Generate responses for samples in a batch."
        sample_ids, prompts = batched_data
        resps = self.model.generate(list(prompts), sampling_params=self.llm_sampling_params, use_tqdm=False)
        output = self._process_llm_response(resps, sample_ids=list(sample_ids))
        return output
    
    def _process_llm_response(self,
                             responses: Union[List[str], List[dict]], 
                             sample_ids: List[str] = []) -> List[dict]:
        """
        Extract and process the LLM's actual text response and return as a list of dicts 
        with the text response and the sample_id. Local LLM (vLLM) output needs
        to also supply the sample_ids.
        """
        if not sample_ids:
            raise ValueError("Sample IDs need to be provided (in correct order) for vLLM response extraction.")
        response_dicts = []
        for idx, r in enumerate(responses):
            resp = r.outputs[0].text.strip()
            processed_resp = self.gen_specs.llm_response_processor(resp)
            if isinstance(processed_resp, list):
                # if a list of items were generated
                for item_idx, item in enumerate(processed_resp):
                    resp_dict = {"resp": item, 
                                "sample_id": "_".join([sample_ids[idx], str(item_idx).rjust(4, '0')])}
            else:
                # if only single item was generated
                resp_dict = {"resp": processed_resp, 
                            "sample_id": sample_ids[idx]}
            response_dicts.append(resp_dict)
        return response_dicts

    def _set_llm_sampling_params(self, params: Dict):
        self.llm_sampling_params = SamplingParams(**params)