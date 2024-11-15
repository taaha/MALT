import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
import os
import argparse
import util_functions

torch.cuda.empty_cache()


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process user information")

    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of training examples to use to find translation direction")
    parser.add_argument("--layer", type=int, required=True, help="layer to find translation direction")

    # Parse arguments
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    DEVICE = 'cuda'

    if "gemma" in MODEL_PATH.lower():
        dtype = torch.float16
    elif "llama" in MODEL_PATH.lower():
        dtype = torch.bfloat16

    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        dtype=dtype,
        default_padding_side='left',
        # fp16=True
    )

    model.tokenizer.padding_side = 'left'

    tokenizer = model.tokenizer

    N_INST_TRAIN = args.num_samples

    urdu_train = util_functions.get_urdu_questions("train")
    english_train = util_functions.get_english_questions("train")

    # tokenize instructions
    urdu_toks = util_functions.tokenize_instructions(tokenizer=tokenizer, prompts=urdu_train[:N_INST_TRAIN])
    english_toks = util_functions.tokenize_instructions(tokenizer=tokenizer, prompts=english_train[:N_INST_TRAIN])

    # run model on english and urdu questions, caching intermediate activations
    urdu_logits, urdu_cache = model.run_with_cache(urdu_toks, names_filter=lambda hook_name: 'resid' in hook_name)
    english_logits, english_cache = model.run_with_cache(english_toks, names_filter=lambda hook_name: 'resid' in hook_name)

    # compute difference of means between english and urdu activations at specified layer
    pos = -1
    layer = args.layer

    urdu_mean_act = urdu_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    english_mean_act = english_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    translation_dir = urdu_mean_act - english_mean_act
    translation_dir = translation_dir / translation_dir.norm()

    save_path = args.model.split('/')[-1]
    torch.save(translation_dir, f'directions/{save_path}_translation_dir.pt')

    del urdu_cache, english_cache, urdu_logits, english_logits
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
