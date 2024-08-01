
def cl100k_base():
    pass
    # mergeable_ranks = load_tiktoken_bpe(
    #     "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    #     expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    # )
    # special_tokens = {
    #     ENDOFTEXT: 100257,
    #     FIM_PREFIX: 100258,
    #     FIM_MIDDLE: 100259,
    #     FIM_SUFFIX: 100260,
    #     ENDOFPROMPT: 100276,
    # }
    # return {
    #     "name": "cl100k_base",
    #     "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    #     "mergeable_ranks": mergeable_ranks,
    #     "special_tokens": special_tokens,
    # }
def o200k_base():
    pass
    # mergeable_ranks = load_tiktoken_bpe(
    #     "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    #     expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    # )
    # special_tokens = {
    #     ENDOFTEXT: 199999,
    #     ENDOFPROMPT: 200018,
    # }
    # # This regex could be made more efficient
    # pat_str = "|".join(
    #     [
    #         r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    #         r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    #         r"""\p{N}{1,3}""",
    #         r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    #         r"""\s*[\r\n]+""",
    #         r"""\s+(?!\S)""",
    #         r"""\s+""",
    #     ]
    # )
    # return {
    #     "name": "o200k_base",
    #     "pat_str": pat_str,
    #     "mergeable_ranks": mergeable_ranks,
    #     "special_tokens": special_tokens,
    # }

ENCODING_CONSTRUCTORS = {
    "cl100k_base": cl100k_base,
    "o200k_base": o200k_base,
}