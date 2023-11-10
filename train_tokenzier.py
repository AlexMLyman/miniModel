from datasets import Dataset
from datasets import concatenate_datasets, load_dataset


# This can take a few minutes to load, so grab a coffee or tea while you wait!
train = load_dataset("roneneldan/TinyStories", split="train")
test = load_dataset("roneneldan/TinyStories", split="validation")

#concat train and test to ensure all tokens are present 
all_txt = train['text']+test['text']

my_dict = {}
my_dict['text']=all_txt

dataset = Dataset.from_dict(my_dict)

batch_size = 1000
all_texts = [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=10000)

