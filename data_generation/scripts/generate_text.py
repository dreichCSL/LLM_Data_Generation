from data_generation.wrappers.llm_generator_wrappers import (
    generate_questions, 
    generate_answers
)
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    if args.type == "questions":
        logger.info(f"Generating questions.")
        generate_questions(config_yaml=args.config_yaml, output_dir=args.output_dir)
    elif args.type == "answers":
        logger.info(f"Generating answers.")
        generate_answers(config_yaml=args.config_yaml, output_dir=args.output_dir)
    else:
        logger.info(f"Type {args.type} is not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate questions about text passages, and filter.")
    parser.add_argument("--config_yaml", "--config", "-c", type=str, dest="config_yaml", required=True,
                        help="yaml file with all settings")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output directory for produced files. Existing files might be overwritten.")
    parser.add_argument("--type", "-t", type=str, required=True, choices=['questions', 'answers'],
                        help="Type of text to generate.")
    
    args = parser.parse_args()
    main(args)
