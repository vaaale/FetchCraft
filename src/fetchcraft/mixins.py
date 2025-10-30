from typing import *


class ObjectNodeMixin:

    def to_json(self) -> Dict[str, Any]:
        raise NotImplemented(f"{self.__class__.__name__}.to_json not implemented.")

