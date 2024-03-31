import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer


def encode_sentences(sentences, model_name="distilbert-base-uncased"):
    # Ensure input sentences are all unique
    assert len(sentences) == len(set(sentences))

    # Load pre-trained model and tokenizer
    model = DistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Tokenize and encode
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)

    # Forward pass to obtain embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings from the output
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    return pd.DataFrame(embeddings, index=sentences)
