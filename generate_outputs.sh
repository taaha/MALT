#!/bin/bash

# Make the script executable with: chmod +x generate_outputs.sh
# Run the script with: ./generate_outputs.sh

# Load environment variables from the .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Log in to Hugging Face CLI
huggingface-cli login --token $HUGGINGFACE_TOKEN

echo "Generating output for gemma-2-2b"
python generate.py --model "gemma-2-2b" --trans_dir_path "directions/gemma-2-2b_translation_dir.pt" --output_dir "results/gemma_results.csv"

echo "Generating output for llama-3.2-3b"
python generate.py --model "meta-llama/Llama-3.2-3B" --trans_dir_path "directions/Llama-3.2-3B_translation_dir.pt" --output_dir "results/llama_results.csv"