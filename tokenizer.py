from transformers import GPT2Tokenizer
from datasets import load_dataset
import pickle

tokenizer = GPT2Tokenizer.from_pretrained("Qilex/tinyStories-10k-tokenizer")
dataset = load_dataset("roneneldan/TinyStories", split="train")

def tokenize_into_array(text_list):
    all_tokens = []
    for sample in text_list:
        #concat next sample
        all_tokens+=tokenizer(sample, truncation = True, max_length=1024)['input_ids']
        #join with eos tokens
        all_tokens+=[tokenizer.eos_token_id]
    return all_tokens

giant_array = tokenize_into_array(dataset['text'])

with open("dataset.pkl", "wb") as f: 
    pickle.dump(giant_array, f)

#with open("new_dataset.pkl", "rb") as f: 
#    arr=pickle.load(f)
