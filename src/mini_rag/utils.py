# utils.py
import json
from typing import List




def save_metadata(metas: List[dict], path='index/metadata.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)




def load_metadata(path='index/metadata.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)