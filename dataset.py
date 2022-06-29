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

def load_data(model_name, batch_size=16, max_len=128):
    print('----------Instantiating a GPT2Tokenizer-------------')
    tokenizer = tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_lowercase=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print('--------------loading XNLI dataset-----------------')
    xnli_data = load_dataset('xnli', 'fr')
    train = xnli_data['train']
    valid = xnli_data['validation']
    test = xnli_data['test']
    train_dataset = XnliDataset(train, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = XnliDataset(valid, tokenizer, max_len=max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = XnliDataset(test, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return tokenizer, train_loader, val_loader, test_loader



