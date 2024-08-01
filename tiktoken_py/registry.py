#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import importlib
from multiprocessing.sharedctypes import Value
from tarfile import ENCODING
from tkinter import E
from typing import Any, Optional, Callable
import pkgutil
from .core import Encoding
import threading
from .openai_public import ENCODING_CONSTRUCTORS

_lock = threading.Lock()

ENCODINGS: dict[str, Encoding] = {}

def get_encoding(encoding_name: str):
    if encoding_name not in ENCODING_CONSTRUCTORS:
        raise ValueError(
                f"Unknown encoding {encoding_name}. Plugins found: {_available_plugin_modules()}"
            )  
    
    constructor = ENCODING_CONSTRUCTORS[encoding_name]
    enc = Encoding(**constructor())
    ENCODINGS[encoding_name] = enc
    return enc