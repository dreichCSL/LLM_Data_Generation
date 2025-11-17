import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assemble_prompt(prompt_templates: dict, 
                    placeholder_content: dict, 
                    language: str = ""):
    system_content = prompt_templates['system_prompt'].render(
        **placeholder_content.get("system", {})
    )
    user_content = prompt_templates.get(
        'user_prompt_' + language, prompt_templates['user_prompt']
    ).render(**placeholder_content.get("user", {}))

    input_msgs = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]
    # render messages in chat template, if specified (e.g., in local llm generation)
    if 'chat_template' in prompt_templates:
        prompt = prompt_templates['chat_template'].render(messages=input_msgs)
    else:
        prompt = input_msgs
    return prompt

def load_jsonl_as_list(infile: str) -> list:
    """Read jsonl and return a list of dicts."""
    output = []
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            output.append(json.loads(line))
    return output

def load_jsonl_as_dict(infile: str, idx_field: str) -> dict:
    """Read jsonl and return one dict with each line being one entry
    and indexed by the specified field."""
    output = {}
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line)
            output[tmp[idx_field]] = tmp
    return output

def sort_idx_by_length(dict_list: list, field: str) -> list:
    """ 
    Sort input list of dicts by length of the specified field, in descending order.
    
    :param dict_list: A list of dictionaries containing the specified field (key).
    :param field: Field (key) of text to measure and sort by.

    Returns:
        Sorted indices of the list. 
    """
    return sorted(range(len(dict_list)), 
                  key=lambda x: len(dict_list[x][field]), 
                  reverse=True)

def make_step_namer(output_dir="output", base_name="output", start=1):
    """Returns a function that generates incrementing step-based file names."""
    os.makedirs(output_dir, exist_ok=True)
    counter = {"step": start - 1}

    def next_name(ext=".jsonl", step_name = ""):
        if not step_name:
            counter["step"] += 1
            step_name = f"{counter['step']:02d}"
        filename = f"{base_name}_{step_name}{ext}"
        return os.path.join(output_dir, filename)

    return next_name