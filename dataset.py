from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset

class XnliDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.premises = data['premise']
        self.hypothesis = data['hypothesis']
        self.labels = data['label']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index):
        sent1 = self.premises[index]
        sent2 = self.hypothesis[index]
        encoded_pairs = self.tokenizer(sent1, sent2, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        input_ids = encoded_pairs['input_ids'].squeeze(0)
        # token_type_ids = encoded_pairs['token_type_ids'].squeeze(0)
        attention_mask = encoded_pairs['attention_mask'].squeeze(0)
        labels = self.labels[index]

        return input_ids, attention_mask, labels

def load_tokenizer(tokenizer_name):
    print('----------Instantiating a GPT2Tokenizer-------------')
    tokenizer = tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, do_lowercase=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def load_data(split, tokenizer, batch_size=16, max_len=128):
    print('--------------loading XNLI dataset-----------------')
    xnli_data = load_dataset('xnli', 'fr')
    data = xnli_data[split]
    
    dataset = XnliDataset(data, tokenizer, max_len=max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader



