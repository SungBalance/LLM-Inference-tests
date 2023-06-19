from pprint import pprint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-350m" # 66b
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("model loaded")

prompt = "Hello, I am conscious and"

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()

print(inputs.keys())
print("------ model.config ------")
pprint(model.config)

print("\n\n------ model.generation_config ------")
pprint(model.generation_config)

outputs = model.generate(input_ids,
    max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95,
    return_dict_in_generate=True, output_scores=True, output_attentions=True, output_hidden_states=True)
print(outputs.keys())
sequences = outputs.sequences
scores = outputs.scores
attentions = outputs.attentions
hidden_states = outputs.hidden_states

target_token_num = 1
target_layer_num = 1

print(f"A number of output tokens: {len(sequences)}")
print(f"target_token_num: {target_token_num}")
print(f"target_layer_num: {target_layer_num}")
print(f"sequences: {sequences}")
print(f"scores: {scores[target_token_num].shape}")
print(f"attentions: {attentions[target_token_num][target_layer_num].shape}")
print(f"hidden_states: {hidden_states[target_token_num][target_layer_num].shape}")

output_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
print(f"output text: {output_text}")
