#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
from multiprocessing.sharedctypes import Value
from tarfile import ENCODING
from tkinter import E
from typing import Any, Optional, Callable
import pkgutil
from .core import Encoding
import threading
import tiktoken_ext

_lock = threading.Lock()

ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: Optional[dict[str, Callable[[], dict[str, Any]]]] = None

def _available_plugin_modules():
    # tiktoken_ext is a namespace package
    # submodules inside tiktoken_ext will be inspected for ENCODING_CONSTRUCTORS attributes
    # - we use namespace package pattern so `pkgutil.iter_modules` is fast
    # - it's a separate top-level package because namespace subpackages of non-namespace
    #   packages don't quite do what you want with editable installs
    mods = []
    plugin_mods = pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + ".")
    for _, mode_name, _ in plugin_mods:
        mods.append(mode_name)
    return mods


def _find_constructor() -> None:
    global ENCODING_CONSTRUCTORS
    # with _lock:   # 使用锁有并发问题
    if ENCODING_CONSTRUCTORS is not None:
        return

    ENCODING_CONSTRUCTORS = {}
    for mod_name in _available_plugin_modules():
        mod = importlib.import_module(mod_name)
        try:
            constructors = mod.ENCODING_CONSTRUCTORS
        except AttributeError as e:
            raise ValueError(
                f"tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS"
            ) from e 
        for enc_name, constructor in constructors.items():
            if enc_name in ENCODING_CONSTRUCTORS:
                raise ValueError(
                    f"Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}"
                )
                ENCODING_CONSTRUCTORS[enc_name] = constructor

def get_encoding(encoding_name: str):
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    
    with _lock:
        # Check again in case another thread has already added it.
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructor()
            assert ENCODING_CONSTRUCTORS is not None

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODING[encoding_name] = enc
        return enc 
        
        
    