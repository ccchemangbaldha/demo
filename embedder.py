# embedder.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeds = model_output.last_hidden_state  # (batch, seq_len, dim)
        mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        summed = torch.sum(token_embeds * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts

    def embed_texts(self, texts, batch_size=16):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                for k in enc: enc[k] = enc[k].to(self.device)
                out = self.model(**enc, return_dict=True)
                pooled = self._mean_pooling(out, enc['attention_mask'])
                # normalize
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)
