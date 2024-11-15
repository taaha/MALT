#!/bin/bash

# Make the script executable with: chmod +x find_translation_dir.sh
# Run the script with: ./find_translation_dir.sh

# Load environment variables from the .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Log in to Hugging Face CLI
huggingface-cli login --token $HUGGINGFACE_TOKEN

echo "Finding translation direction for gemma-2-2b"
python direction.py --model "gemma-2-2b" --layer 24

echo "Finding translation direction for llama-3.2-3b"
python direction.py --model "meta-llama/Llama-3.2-3B" --layer 25