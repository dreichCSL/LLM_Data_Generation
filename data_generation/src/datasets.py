from jinja2 import Template
from torch.utils.data import Dataset
from ..utils.helpers import (
    load_jsonl_as_dict, 
    load_jsonl_as_list, 
    assemble_prompt,
    sort_idx_by_length
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextConditioned(Dataset):
    """
    Dataset producing prompts for text generation conditioned on some context. 

    :param context_file: File with dict for each text entry (json or jsonl files).
    :param prompt_templates: Dict with prompt templates (jinja2 format).
    """
    def __init__(self, context_file: str, prompt_templates: dict):
        self.prompt_templates = prompt_templates
        self.text_data = load_jsonl_as_list(context_file)
        self.sorted_idx = sort_idx_by_length(self.text_data, field='text')

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        idx = self.sorted_idx[index]
        context = self.text_data[idx]["text"]
        context_id = self.text_data[idx]["text_id"]
        # default to English if language not specified in text file or user template
        language = self.text_data[idx].get("language", "en")
        placeholder_content = {"user": {"passage": context}}
        prompt = assemble_prompt(self.prompt_templates, placeholder_content, language)
        return context_id, prompt

class TextAndDiscourseConditioned(Dataset):
    """Dataset for answer generation for given questions and text passages."""
    # TODO: samples aren't sorted by length which might be inefficient for batch procesing
    def __init__(self, context_file: str, discourse_file: str, prompt_templates: dict):
        self.prompt_templates = prompt_templates
        self.text_data = load_jsonl_as_dict(context_file, idx_field='text_id')
        self.discourse_acts = load_jsonl_as_list(discourse_file)
        # self.sorted_idx = sort_idx_by_length(self.questions, field='text')

    def __len__(self):
        return len(self.discourse_acts)
    
    def __getitem__(self, index):
        data = self.discourse_acts[index]
        context_id = data['sample_id'].split('_')[0]  # sample_id consists of text_id + question_idx, one string
        context = self.text_data[context_id]['text']
        language = self.text_data[context_id].get('language', "en")
        placeholder_content = {"user": {"passage": context, "discourse_act": data['resp']}}
        prompt = assemble_prompt(self.prompt_templates, placeholder_content, language)
        return data['sample_id'], prompt

# TODO: possibly add more classes for different functionalities
# class QuestionRatingDataset(TextAndDiscourseConditioned):  
#     # same as AnswerGenDataset
#     pass
    
# class AnswerRatingDataset(TextAndDiscourseConditioned):
#     # same as above but using answer_file instead of question_file
#     """Dataset to determine entailment between answers and given passages."""
#     # def __init__(self, scraper_file: str, answer_file: str, prompt_templates: dict):
#     pass

# class QATest(Dataset):
#     """Dataset for testing questions without context."""
#     def __init__(self, test_file: str, prompt_templates: dict):
#         self.prompt_templates = prompt_templates
#         self.prompt_templates['user_prompt'] = Template("{{discourse_act}}")
#         self.qa_samples = load_jsonl_as_list(test_file)  #json.load(open(test_file, 'r', encoding='utf-8'))

#     def __len__(self):
#         return len(self.qa_samples)

#     def __getitem__(self, index):
#         qa_id = self.qa_samples[index]['sample_id']
#         q = self.qa_samples[index]['question']
#         placeholder_content = {"user": {"discourse_act": q}}
#         prompt = assemble_prompt(self.prompt_templates, placeholder_content)
#         return qa_id, prompt

# class JudgeTest(Dataset):
#     """Dataset to determine entailment between answers and given passages."""
#     def __init__(self, preds_file: str, refs_file: str, prompt_templates: dict):
#         self.prompt_templates = prompt_templates
#         self.refs_samples_dict = load_jsonl_as_dict(refs_file, "sample_id")
#         self.samples = load_jsonl_as_list(preds_file)
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         qa_id = sample['sample_id']
#         ref = self.refs_samples_dict[qa_id]['answer']
#         placeholder_content = {"user": {"pred": sample['text'], "ref": ref}}
#         prompt = assemble_prompt(self.prompt_templates, placeholder_content)
#         return qa_id, prompt