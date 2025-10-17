import os, json, pickle
from typing import Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
