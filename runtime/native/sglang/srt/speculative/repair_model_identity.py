"""Content identities for the locally loaded, immutable safetensors checkpoint.

Only the local safetensors loader is in this validation contract. A downloaded
HF id, an absent checkpoint, another serialization or changed files is rejected.
"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest_files(root,names):
    root=Path(root).resolve()
    entries=[]
    for name in sorted(set(names)):
        p=(root/name).resolve()
        if root not in p.parents or not p.is_file():
            raise ValueError('Checkpoint member absent or outside model directory: '+name)
        before=p.stat()
        digest=file_sha256(p)
        after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Checkpoint changed while hashing')
        entries.append((name,after.st_size,digest))
    return hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()


def model_identity(path):
    root=Path(path).resolve()
    if not root.is_dir():raise ValueError('Validated repair requires an existing local checkpoint')
    index=root/'model.safetensors.index.json'
    if index.is_file():
        names=set(json.loads(index.read_text(encoding='utf8'))['weight_map'].values())
        if not names or any(not n.endswith('.safetensors') for n in names):
            raise ValueError('Unsupported checkpoint weight index')
        names.add(index.name)
    elif (root/'model.safetensors').is_file():names={'model.safetensors'}
    else:raise ValueError('Validated repair supports local safetensors checkpoints only')
    names.add('config.json')
    if (root/'generation_config.json').is_file():names.add('generation_config.json')
    return str(root),digest_files(root,names)


def tokenizer_identity(path):
    root=Path(path).resolve()
    names=[n for n in ['tokenizer.json','tokenizer.model','tokenizer_config.json',
                      'special_tokens_map.json','added_tokens.json','chat_template.jinja'] if (root/n).is_file()]
    if not ({'tokenizer.json','tokenizer.model'}&set(names)):
        raise ValueError('Tokenizer checkpoint absent')
    return digest_files(root,names)
