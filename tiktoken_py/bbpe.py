#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from copyreg import pickle
import regex as re
from itertools import islice, tee
import numpy as np


def byte_pair_merge(ranks: dict[list, int], piece: int):
    # parts表示分词的边界，保存的是每个词的开始位置以及该词的频率 (start, rank)
    # 每个词的内容是当前词的开始位置start与下一个词的开始位置之间的内容，包括下一个词的起始位置
    # 或者说是通用的bi-gram内容
    # parts 中每一个元素可以当做是一个广义上的bi-gram的token
    parts = []

    # 请注意，tiktoken 在对 ranks 进行索引时是对字节进行哈希处理，而不是对词对 tokens pair
    # 只要我们按照当前的方式进行BPE训练，这是等效的。打破这种等价关系的一个简单方法是解耦合并优先级与词索引，或者阻止特定的词合并。
    min_rank = (np.iinfo(np.int32).max, np.iinfo(np.int32).max)
    for i in range(0, len(piece) - 1):
        rank = ranks.get(piece[i: i + 2], np.iinfo(np.int32).max)
        if rank < min_rank[0]:
            min_rank = (rank, i)
        parts.append((i, rank))
    
    parts.append((len(piece) - 1, np.iinfo(np.int32).max))
    parts.append((len(piece), np.iinfo(np.int32).max))
    
    def get_rank(parts, i):
        # 用来服用 ranks和piece，不需要额外的通过参数形式传递到get_rank中
        # 计算trigram的频率，当前 i 位置的bigram需要合并，即 parts[i]与parts[i+1]需要合并到一起
        # 需要重新计算 (parts[i]parts[i+1], parts[i+2])的频率
        if i + 3 < len(parts):
            trigram = piece[parts[i][0]:parts[i+3][0]]
            rank =  ranks.get(trigram, np.iinfo(np.int32).max)
            return rank 
        else:
            return np.iinfo(np.int32).max

    # rank 表示频率的排名，rank 越小，表示该组合频率越高
    # 因此这里要找到rank最小的组合
    # 这里需要一直合并，直到没有需要合并的bigram为止
    while min_rank[0] != np.iinfo(np.int32).max:
        i = min_rank[1]
        if i > 0:
            # 获取合并之后的新的(left, i)的rank
            parts[i-1] = (parts[i-1][0], get_rank(parts, i-1))
        # 当前的元素min_rank已经合并了，需要合并min_rank后一个元素
        # 即 (piece[i], piece[i+1])合并了，将 (piece[i] piece[i+1])作为一个元素
        # 此时需要看一下 (piece[i]piece[i+1], piece[i+2])的频率
        # 获取合并之后的(i, right)的rank
        parts[i] = (parts[i][0], get_rank(parts, i))    # 重新赋值一个新的 tuple
        # 删掉 part[i+1]，因为第 (i, i+1)已经合并了，一定不会合并(i+1, i+2)了，(i+1)被用在(i, i+1)合并的地方了
        del parts[i+1]  # 合并后面一个元素

        # 此时需要重新找一个min_rank，即找出频率最大的待合并的bi-gram
        min_rank = (np.iinfo(np.int32).max, np.iinfo(np.int32).max)
        for i, bigram in enumerate(parts):
            rank = bigram[1]
            # 更新最小的rank， 即频率最大的组合
            if rank < min_rank[0]:
                min_rank = (rank, i)

    return parts

    
class CoreBPE:
    def __init__(self,
                 encoder: dict[bytes, int]=None,
                 special_tokens_encoder: dict[int, int]=None,
                 pattern: str=None,
                ) -> None:
        self.encoder = encoder
        self.special_tokens_encoder = special_tokens_encoder
        self.pattern = pattern
        
        # 构建匹配正则表达式
        self.regex_tls = re.compile(pattern)
        
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
        return self._encode_native(text, allowed_special)

    def _encode_native(self, text, allowed_special):
        special_regex = self.special_regex_tls
        regex_tls = self.regex_tls
        ret = []
        start = 0
        last_piece_token_len = 0
        while True:
            next_special = None 
            start_find = start

            # 找到第一个特殊符号，并记录其位置
            # 特殊符号之前的内容是有效的prompt输入
            while True:
                # 匹配是否命中了词典中的符号
                match = special_regex.search(text, start_find)
                if not match:
                    break
                special_token = match.group(0)
                if special_token in allowed_special:
                    next_special = match
                    break
                start_find = match.start() + 1
            
            # 匹配词典中的符号，找到真正有效的prompt输入之后确定end位置
            end = len(text) if not next_special else next_special.start()
            for match in regex_tls.finditer(text[start:end]):
                
                # 匹配词典中的符号，直接记录词典中的id
                piece = match.group(0).encode('utf-8')
                if piece in self.encoder:
                    last_piece_token_len = 1
                    ret.append(self.encoder[piece])
                    continue
                
                # 如果没有匹配到词典中的内容，将该token转换为bytes字节数据，使用bbpe算法进行拆分
                # tokens 是字节编码列表，通过extend接口附加在ret的后面
                tokens = self.bype_pair_encode(piece, self.encoder)
                last_piece_token_len = len(tokens)
                ret.extend(tokens)

            # 匹配到特殊符号，需要将特殊符号转好到对应的id
            if next_special:
                token = self.special_tokens_encoder[next_special.group(0)]
                ret.append(token)

                # 特殊符号后面的有效输入，需要重新开始匹配
                start = next_special.end()
                last_piece_token_len = 0
            else:
                break
        # last_piece_token_len is how many tokens came from the last regex split. This is used
        # for determining unstable tokens, since you can't merge across (stable) regex splits
        # last_piece_token_len 记录了最后一个匹配到的片段的长度
        return ret, last_piece_token_len

    def bype_pair_encode(self, piece, ranks):
        assert len(piece) > 1
        pairs =  byte_pair_merge(ranks, piece)
        tokens = []
        for idx in range(len(pairs) - 1):
            start, end = pairs[idx][0], pairs[idx + 1][0]
            tokens.append(ranks[piece[start:end]])

        return tokens
        # pairs = zip(piece, islice(piece, 1, None))
        # return [ranks[pair] for pair in pairs]

    def _decode_native(self, tokens):
        ret = bytearray()
        for token in tokens:
            token_bytes = self.decoder.get(token, None)
            if token_bytes is None:
                token_bytes = self.special_tokens_decoder.get(token, None)

            if token_bytes is not None:
                ret.extend(token_bytes)

        return ret

    def decode_bytes(self, tokens):
        return self._decode_native(tokens)

        
