from typing import List, Optional

from llama_index import QueryBundle
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore


class DummyNodePostprocessor:
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        for n in nodes:
            n.score = 1 - n.score

        return nodes
