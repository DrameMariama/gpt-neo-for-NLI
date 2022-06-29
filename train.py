import torch
from model import load_model
from dataset import load_data, load_tokenizer
from transformers import get_linear_schedule_with_warmup, set_seed
from torch.optim import Adam
from sklearn.metrics import accuracy_score
##from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os

#writer = SummaryWriter()

print("okay")
def save_checkpoint(model, ckpt_path, model_name):
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), ckpt_path+model_name)
    print("-------------Saved model-------------")

def load_checkpoint(model, ckpt_path, model_name):
  if os.path.exists(ckpt_path+model_name):
    print("----------loading model from checkpoint------------")
    if torch.cuda.is_available():
       model.load_state_dict(torch.load(ckpt_path+model_name))
    else:
       model.load_state_dict(torch.load(ckpt_path+model_name, map_location=torch.device('cpu')))
    print("--"*40)
    print("model loaded succesfully")
    print("--"*40)
  return model

def train(train_loader, val_loader, model, model_name, optimizer, scheduler, device, num_epochs, ckpt_path,  print_every=1000, save_every=1000):

    ##put model on the device
  model.to(device)
  model.train()
  total_loss = 0
  running_loss = 0
  predicted_labels = []
  true_labels = []
  eval_labels = []
  true_eval_labels = []
  global_steps = 0
  prev_eval_Acc = -np.inf
  for epoch in tqdm(range(num_epochs)):
    print(f"Training on batch for Epoch{epoch}")
    for batch_id, data in enumerate(tqdm(train_loader)):
        input_ids, attention_mask, labels = data
        ## put the data on the device##
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        #collect the true labels for accuracy computation
        true_labels += labels.tolist()
        ##zero grad the optimizer
        optimizer.zero_grad()
        ## fit data to the model
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs[:2]
        running_loss += loss.item()
        #update model parameters
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_steps += 1
        predicted_labels += logits.argmax(-1).flatten().tolist()
        if (batch_id + 1) % print_every == 0:  # Print training loss information
            print()
            print("Iteration {}/{}  complete. Loss : {} "
                .format(batch_id+1,len(train_loader), running_loss / print_every))
            train_acc = accuracy_score(true_labels, predicted_labels)
            mean_loss = running_loss / print_every
            # writer.add_scalar("Loss/Train", mean_loss, global_steps)
            # writer.add_scalar("Accuracy/Train", train_acc, global_steps)
            running_loss = 0.0
            true_labels = []
            predicted_labels = []
        ## evaluate the model
        if (batch_id + 1) % save_every == 0:
            with torch.no_grad():
                for batch, data in enumerate(val_loader):
                    input_ids, attention_mask, labels = data
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    true_eval_labels += labels.tolist()
                    outputs = model(input_ids, attention_mask=attention_mask,  labels=labels)
                    loss, logits = outputs[:2]
                    eval_labels += logits.argmax(-1).flatten().tolist()
            ## compute accuracy on validation set
            eval_acc = accuracy_score(true_eval_labels, eval_labels)
            #writer.add_scalar("Accuracy/Eval", eval_acc, global_steps)
            print("Validation Accuracy {} ".format(eval_acc))

            # save model if current accuracy is higher than the previous saved model
            if eval_acc > prev_eval_Acc:
                save_checkpoint(model, ckpt_path, model_name)
                prev_eval_Acc = eval_acc
            true_eval_labels = []
            eval_labels = []

if __name__=="__main__":
    model_name = 'EleutherAI/gpt-neo-125M'
    tokenizer = load_tokenizer(model_name)
    train_loader = load_data(split='train', tokenizer=tokenizer)
    val_loader = load_data(split='validation', tokenizer=tokenizer)
    model = load_model(model_name, tokenizer)
    optimizer = Adam(model.parameters(), lr=1e-5)
    epochs = 5
    total_steps = len(train_loader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = './checkpoint/'
    name = "gptneo.pt"
    model = load_checkpoint(model, ckpt_path, name)
    print(f"Training on {device}")
    train(train_loader, val_loader, model=model, model_name=name, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=epochs, ckpt_path=ckpt_path)
