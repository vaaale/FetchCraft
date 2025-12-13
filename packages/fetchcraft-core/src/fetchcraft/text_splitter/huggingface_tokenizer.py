from typing import List
from transformers import AutoTokenizer

from fetchcraft.text_splitter import Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer(text)["input_ids"]

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
