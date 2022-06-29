from transformers import GPTNeoForSequenceClassification, GPTNeoConfig

def load_model(model_name, tokenizer):
    config = GPTNeoConfig.from_pretrained(model_name, num_labels=3)
    print("------------loading the model----------------")
    model = GPTNeoForSequenceClassification.from_pretrained(model_name, config=config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model