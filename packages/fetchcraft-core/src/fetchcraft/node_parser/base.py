from abc import ABC, abstractmethod
from typing import *
from typing import List, Optional

from pydantic import BaseModel

from fetchcraft.node import DocumentNode, Node


class NodeParser(BaseModel, ABC):

    @abstractmethod
    def get_nodes(self, documents: List[DocumentNode], metadata: Optional[Dict[str, Any]] = None) -> List[Node]:
        pass
