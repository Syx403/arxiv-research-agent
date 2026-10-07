"""Token counting with cl100k for chunk sizes and cost estimates; providers count their own."""

from functools import cache

import tiktoken


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode_ordinary(text))
