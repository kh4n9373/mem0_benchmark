from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import torch
import numpy as np

model_name = "facebook/contriever"
print(f"Loading {model_name} with SentenceTransformer...")
try:
    model = SentenceTransformer(model_name)
    print("Model modules:")
    for mod in model.modules:
        print(f" - {mod}")
except Exception as e:
    print(f"Error loading model: {e}")

query = "Where is the Eiffel Tower?"
doc = "The Eiffel Tower is located in Paris, France."
irrelevant = "Bananas are yellow."

print("\nEncoding...")
q_emb = model.encode(query, convert_to_numpy=True)
d_emb = model.encode(doc, convert_to_numpy=True)
i_emb = model.encode(irrelevant, convert_to_numpy=True)

# Dot product (unnormalized)
dot_qd = np.dot(q_emb, d_emb)
dot_qi = np.dot(q_emb, i_emb)

# Cosine similarity
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_qd = cosine(q_emb, d_emb)
cos_qi = cosine(i_emb, i_emb)

print(f"Dot q-d: {dot_qd:.4f}")
print(f"Dot q-i: {dot_qi:.4f}")
print(f"Cos q-d: {cos_qd:.4f}")
print(f"Cos q-i: {cos_qi:.4f}")

print("\nChecking normalization default:")
print(f"Norm(q): {np.linalg.norm(q_emb):.4f}")

# Check if using transformers directly gives different result
print("\nComparing with Transformers (CLS vs Mean)...")
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = hf_model(**inputs)
    
last_hidden = outputs.last_hidden_state
# Mean pooling
mean_emb = last_hidden.mean(dim=1).squeeze().numpy()
# CLS pooling
cls_emb = last_hidden[:, 0, :].squeeze().numpy()

cos_sbert_mean = cosine(q_emb, mean_emb)
cos_sbert_cls = cosine(q_emb, cls_emb)

print(f"Sim SBERT vs HF-Mean: {cos_sbert_mean:.4f}")
print(f"Sim SBERT vs HF-CLS: {cos_sbert_cls:.4f}")
