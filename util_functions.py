import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import csv
import gc
from datasets import load_dataset, load_from_disk
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


def get_urdu_questions(split):
    # hf_path = 'darthPanda/ur_en_questions'
    # dataset = load_dataset(hf_path, split=split)
    dataset_path = f'dataset/ur_en_questions/{split}'
    dataset = load_from_disk(dataset_path)
    
    # Extract the Urdu questions
    urdu_test_questions = dataset["urdu_question"]
        
    return urdu_test_questions


def get_english_questions(split):
    # hf_path = 'darthPanda/ur_en_questions'
    # dataset = load_dataset(hf_path, split=split)
    dataset_path = f'dataset/ur_en_questions/{split}'
    dataset = load_from_disk(dataset_path)
    
    
    # Extract the Urdu questions
    urdu_test_questions = dataset["english_question"]
        
    return urdu_test_questions


def tokenize_instructions(
    tokenizer: AutoTokenizer,
    prompts: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids


def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)


def get_generations(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions(prompts=instructions[i:i+batch_size], tokenizer=tokenizer)
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj