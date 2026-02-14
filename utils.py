# utils.py
import uuid, json
from pathlib import Path

def uid():
    return int(uuid.uuid4().int & (1<<63)-1)  # int id for faiss

def save_docstore(path, docstore):
    Path(path).write_text(json.dumps(docstore, indent=2))
def load_docstore(path):
    p = Path(path)
    if not p.exists():
        return {}

    try:
        content = p.read_text().strip()
        if not content:
            return {}
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
