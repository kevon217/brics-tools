from typing import List, Optional
import pandas as pd

from llama_index import QueryBundle
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore
from llama_index.indices.postprocessor import SentenceTransformerRerank


# class StudyInfoNodePostprocessor:
#     def postprocess_nodes(
#     self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
# ) -> List[NodeWithScore]:
#         # nodes = aggregate_studynodes(nodes)
#         scores = []
#         study_ids = []
#         study_titles = []
#         texts = []
#         study_keywords = []
#         idx = 0
#         for n in nodes:
#             idx += 1
#             scores.append(n.score)
#             study_ids.append(n.node.ref_doc_id)
#             metadata = n.node.source_node.metadata
#             n_dict = n.node.to_dict()
#             texts.append(n_dict['text'])
#             study_titles.append(metadata['title'])
#             study_keywords.append(n.node.metadata['excerpt_keywords'])
#         return nodes


class NodePostprocessorCosineSimilarityTempFix:
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        for n in nodes:
            n.score = 1 - n.score

        return nodes


# class DummyNodePostprocessor:

#     def postprocess_nodes(
#         self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
#     ) -> List[NodeWithScore]:

#         # subtracts 1 from the score
#         for n in nodes:
#             n.score = 1 - n.score

#         return nodes

# def node_results_to_dataframe(results):
#     source_nodes = results.source_nodes
#     results = []
#     for node in source_nodes:
#         result_dict = {}
#         node_dict = node.dict()
#         node_score = {"score": node_dict["score"]}
#         node_info = node_dict["node"]["relationships"]["1"]
#         node_id = {"node_id": node_info["node_id"]}
#         node_meta = node_info["metadata"]
#         result_dict = {**node_id, **node_meta, **node_score}
#         results.append(result_dict)
#     df_results = pd.DataFrame(results)
#     return df_results
