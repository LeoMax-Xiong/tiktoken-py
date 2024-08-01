#!/usr/bin/env python3
from typing import Optional
import hashlib
import blobfile
import base64
import os

def check_hash(data: bytes, expected_hash: str) -> bool:
    actual_hash = hashlib.sha256(data).hexdigest()
    return actual_hash == expected_hash

def read_file(file_path: str, expected_hash: Optional[str] = None) -> bytes:
    contents = None
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_path, file_path)
    with blobfile.BlobFile(file_path, "rb") as f:
        contents = f.read()
    if expected_hash and not check_hash(contents, expected_hash):
        raise ValueError(
            f"Hash mismatch for data from {file_path} (expected {expected_hash}). "
            f"This may indicate a corrupted download. Please try again."
        )
    return contents


def load_tktoken_bpe(tiktoken_bpe_file: str, 
                     expected_hash: Optional[str] = None
        ) -> dict[bytes, int]:
    # NB: do not add caching to this function
    contents = read_file(tiktoken_bpe_file, expected_hash)
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
