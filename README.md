# LLM-driven Data Generation

## Overview
This repository contains scripts and processes for LLM-driven data generation. It relies on vLLM as inference engine. 
Currently supported functionality:
- Generating questions about given text passages

## Setup
Install:
```bash
git clone https://github.com/dreichCSL/LLM_Data_Generation.git
cd llm_data_generation/
pip install -e .[gpu]
```

## Usage
Callable scripts are in `data_generation/scripts/`. Sample configuration files are in `configs/`. The config files act as examples, modify them for your use-case. 

### Data Generation
```bash
python data_generation/scripts/generate_text.py --config configs/config.yaml --output_dir output_questions --type questions
```
This script acts as entry point for various text generation processes. Select from the available process by specifying the `--type` (see also the help message of the script). Output will be written to the directory passed with `--output_dir`.
