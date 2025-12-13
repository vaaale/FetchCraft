import re
from typing import Any

from llama_index.core.types import BaseOutputParser


class ThinkingOutputParser(BaseOutputParser):

    def parse(self, output: str) -> Any:
        match = re.match(r"<think>(?P<reasoning>.*)</think>(?P<answer>.*)", output, flags=re.DOTALL)
        if match:
            answer = match.group("answer").strip()
            return answer
        else:
            return output
