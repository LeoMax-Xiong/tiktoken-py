#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
from .bbpe import CoreBPE
import functools
import regex
class Encoding:
    def __init__(self,
                name: str,
                *,
                pat_str: str,
                mergeable_ranks: dict[bytes, int],
                special_tokens: dict[str, int],
                explicit_n_vocab: Optional[int] = None,
            ) -> None:
        """Creates an Encoding object.

        See openai_public.py for examples of how to construct an Encoding object.

        Args:
            name: The name of the encoding. It should be clear from the name of the encoding
                what behaviour to expect, in particular, encodings with different special tokens
                should have different names.
            pat_str: A regex pattern string that is used to split the input text.
            mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
                must correspond to merge priority.
            special_tokens: A dictionary mapping special token strings to their token values.
            explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
                that the number of mergeable tokens and special tokens is equal to this number.
        """
        self.name = name

        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        self.max_token_value = max(
            max(mergeable_ranks.values()), max(special_tokens.values(), default=0)
        )
        if explicit_n_vocab:
            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
            assert self.max_token_value == explicit_n_vocab - 1

        self._core_bpe = CoreBPE(encoder=self._mergeable_ranks,
                                special_tokens_encoder=self._special_tokens,
                                pattern=self._pat_str,
                            )

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        return set(self._special_tokens.keys())

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set 

        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special

        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)

            if match := _special_token_regex(disallowed_special).search(text):
                raise ValueError(f"Text contains disallowed special token: {match.group()}")
        
        if isinstance(allowed_special, frozenset):
            allowed_special = set(allowed_special)

        try:
            # Encode the text using BPE
            tokens, _ = self._core_bpe.encode(text, allowed_special)
            return tokens
        except UnicodeEncodeError:
            # BPE operates on bytes, but the regex operates on unicode. If we pass a str that is
            # invalid UTF-8 to Rust, it will rightfully complain. Here we do a quick and dirty
            # fixup for any surrogate pairs that may have sneaked their way into the text.
            # Technically, this introduces a place where encode + decode doesn't roundtrip a Python
            # string, but given that this is input we want to support, maybe that's okay.
            # Also we use errors="replace" to handle weird things like lone surrogates.
            text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
            tokens, _ = self._core_bpe.encode(text, allowed_special)
            return tokens
        


    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can control this behaviour using the `errors` parameter,
        for instance, setting `errors=strict`.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)


def _special_token_regex(tokens: frozenset[str]):
    inner = "|".join(regex.escape(token) for token in tokens)

    return regex.compile(f"({inner})")