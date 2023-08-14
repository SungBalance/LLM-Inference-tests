import os

import tqdm

import torch
import deepspeed

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

models = [
    # 'EleutherAI/gpt-neo-2.7B',
    'bigscience/bloom-7b1',
    # 'facebook/opt-13b'
    # 'Flan-T5-XL', # 3B
    # 'google/flan-t5-xxl', # 11B
]

# download models

for model in models:
    generator = pipeline('text-generation', model=model,
                    batch_size=64, device=local_rank)
    del generator

generator = pipeline('text-generation', model='bigscience/bloom-7b1',
                    batch_size=64, device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)

generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

string = generator("DeepSpeed is", do_sample=True, min_length=50)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)


# dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
# generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
# for out in tqdm(generator(KeyDataset(dataset, "file"), do_sample=True, min_length=50)):
#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         print(out)
