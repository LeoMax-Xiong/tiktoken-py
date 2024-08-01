#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import regex as re
from itertools import islice, tee
class CoreBPE:
    def __init__(self,
                 encoder: dict[bytes, int]=None,
                 special_tokens_encoder: dict[int, int]=None,
                 pattern: str=None,
                #  decoder=None,
                #  special_tokens_decoder=None,
                #  regex_tls=None,
                #  special_regex_tls=None,
                #  sorted_token_bytes=None
                ) -> None:
        self.encoder = encoder
        self.special_tokens_encoder = special_tokens_encoder
        self.pattern = pattern
        
        # 构建匹配正则表达式
        self.regex = re.compile(pattern)
        
        # 构建特殊符号匹配正则表达式
        escaped_parts = [re.escape(key) for key in special_tokens_encoder.keys()]
        pattern = '|'.join(escaped_parts)
        try:
            special_regex = re.compile(pattern)
        except re.error:
            raise ValueError(f'Invalid pattern: {pattern}')
        self.special_regex_tls = special_regex

        # 构建反tokenizer的模块
        self.decoder  = {v: k for k, v in self.encoder.items()}
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens_encoder.items()}

        # 获取排序后的符号
        self.sorted_token_bytes = sorted(self.encoder.keys())

    def encode(self, text: str, allowed_special: set[str]):
        return  self._encode_native(text, allowed_special)

    def _encode_native(self, text, allowed_special):
        special_regex = None
        regex = None
        ret = []
        start = 0
        last_piece_token_len = 0
        while True:
            next_special = None 
            start_find = start
            while True:
                match = special_regex.search(text, start_find)
                if not match:
                    break
                special_token = match.group(0)
                if special_token in allowed_special:
                    next_special = match
                    break
                start_find = match.start() + 1
            
            end = len(text) if not next_special else next_special.start()
            for match in regex.finditer(text[start:end]):
                piece = match.group(0).encode('utf-8')
                if piece in self.encoder:
                    last_piece_token_len = 1
                    ret.append(self.encoder[piece])
                    continue
                
                tokens = self.bype_pair_encode(piece, self.encoder)
                last_piece_token_len = len(tokens)
                ret.extend(tokens)

            if next_special:
                token = self.special_tokens_encoder[next_special.group(0)]
                ret.append(token)
                start = next_special.end()
                last_piece_token_len = 0
            else:
                break

        return ret

    def bype_pair_encode(self, piece, ranks):
        assert len(piece) > 1
        pairs = zip(piece, islice(piece, 1, None))
        return [ranks[pair] for pair in pairs]

    def _decode_native(self, tokens):
        ret = bytearray()
        for token in tokens:
            token_bytes = self.decoder.get(token, None)
            if token_bytes is None:
                token_bytes = self.special_tokens_decoder.get(token, None)

            if token_bytes is not None:
                ret.append(token_bytes)

        return ret
    def decode_bytes(self, tokens):
        return self._decode_native(tokens)

        
