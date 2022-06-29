import torch
from train import load_checkpoint
from model import load_model
from dataset import load_data
from sklearn.metrics import accuracy_score

def eval(model, test_loader, device):
    test_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            test_labels += labels.tolist()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[:2]
            predicted_labels += logits.argmax(-1).flatten().tolist()
    test_acc = accuracy_score(test_labels, predicted_labels)
    return test_acc

if __name__=="__main__":
    ckpt_path = './checkpoint/'
    checkpoint_name = "gptneo.pt"
    model_name = 'EleutherAI/gpt-neo-125M'
    tokenizer, _, _, test_loader = load_data(model_name)
    
    model = load_model(model_name, tokenizer)
    model = load_checkpoint(model, ckpt_path, model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_acc = eval(model, test_loader, device)
