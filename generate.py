import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import csv
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
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
    parser.add_argument("--trans_dir_path", type=str, default=16, help="Path for translation direction")
    parser.add_argument("--output_dir", type=str, default=16, help="Output path for results")

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

    urdu_test = util_functions.get_urdu_questions("test")
    english_test = util_functions.get_english_questions("test")

    translation_dir = torch.load(args.trans_dir_path)

    intervention_dir = translation_dir
    intervention_layers = list(range(model.cfg.n_layers)) # all layers

    hook_fn = functools.partial(util_functions.direction_ablation_hook,direction=intervention_dir)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

    intervention_generations = util_functions.get_generations(model, tokenizer, urdu_test, util_functions.tokenize_instructions, fwd_hooks=fwd_hooks)
    baseline_generations = util_functions.get_generations(model, tokenizer, urdu_test, util_functions.tokenize_instructions, fwd_hooks=[])

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # Loading translation model and doing translations
    model = MBartForConditionalGeneration.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
    tokenizer = MBart50TokenizerFast.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
    tokenizer.src_lang = "en_XX"
    # encoded_en = tokenizer(intervention_generations, return_tensors="pt", padding=True)
    # generated_tokens = model.generate(
    #     **encoded_en,
    #     forced_bos_token_id=tokenizer.lang_code_to_id["ur_PK"]
    # )
    # translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    translations=[]
    for intervention_generation in tqdm(intervention_generations, desc="Running translation"):
        encoded_en = tokenizer(intervention_generation, return_tensors="pt", padding=True)
        generated_tokens = model.generate(
            **encoded_en,
            forced_bos_token_id=tokenizer.lang_code_to_id["ur_PK"]
        )
        temp = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.append(temp[0])
    
    output_file = args.output_dir

    with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sr No.", "Original Instruction", "Translated Instruction", 
                        "Baseline Completion", "Intervention Completion", "Translated Completion (Final Output)"])
        
        for i in tqdm(range(len(urdu_test))):
            query_num = f"Query {i}"
            original_instruction = repr(urdu_test[i])
            translated_instruction = repr(english_test[i])            
            baseline_completion = textwrap.fill(repr(baseline_generations[i]), width=100)
            intervention_completion = textwrap.fill(repr(intervention_generations[i]), width=100)
            translation = translations[i]            
            writer.writerow([query_num, original_instruction, translated_instruction, 
                            baseline_completion, intervention_completion, translation])

if __name__ == "__main__":
    main()
