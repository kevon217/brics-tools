from typing import List, Optional
import pandas as pd

from llama_index import QueryBundle
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore
from llama_index.indices.postprocessor import SentenceTransformerRerank


class TopNForLLMSynthesisNodePostprocessor:
    def __init__(self, rerank_top_n: int, top_n_for_llm: int):
        """
        Initialize the postprocessor with the desired number of top results for LLM synthesis.

        :param rerank_top_n: Number of top results coming from reranking step.
        :param top_n_for_llm: Number of top results to keep for LLM synthesis.
        """
        self.rerank_top_n = rerank_top_n
        self.top_n_for_llm = top_n_for_llm

    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        if self.top_n_for_llm:
            if self.top_n_for_llm != self.rerank_top_n:
                sorted_nodes = sorted(
                    nodes, key=lambda node: node.score, reverse=True
                )  # Sort nodes by score in ascending order (since lower scores indicate greater similarity)
                if self.top_n_for_llm < self.rerank_top_n:
                    nodes = sorted_nodes[: self.top_n_for_llm]
                else:
                    nodes = sorted_nodes[
                        : self.rerank_top_n
                    ]  # Keep the top n nodes from the reranking step

        return nodes


# class NodePostprocessorCosineSimilarityTempFix:
#     def postprocess_nodes(
#         self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
#     ) -> List[NodeWithScore]:
#         # subtracts 1 from the score
#         for n in nodes:
#             n.score = 1 - n.score

#         return nodes
