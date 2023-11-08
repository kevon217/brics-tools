from abc import ABC, abstractmethod
from typing import List, Dict
from brics_tools.brics_modules.data_dictionary.crossmap.llamaindex_setup.document_creators.base import (
    DocumentCreator,
)
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser


class UMLSDocumentCreator(DocumentCreator):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.parser = SimpleNodeParser.from_defaults(
            include_metadata=True,
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 0),
        )

    def create_documents(self):
        id_col = self.config["id_column"]
        embed_cols = self.config["embed_columns"]
        metadata_cols = self.config["metadata_columns"]

        index_nodes = {}

        for col in embed_cols:
            ids = []
            documents = []

            for _, row in self.df.iterrows():
                doc = row[col]
                meta = {val: row[val] for val in metadata_cols}
                document = Document(
                    text=doc,
                    metadata=meta,
                    excluded_embed_metadata_keys=list(meta.keys()),
                    text_template="{content}",
                )
                documents.append(document)

            nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)
            index_nodes[col] = nodes

        return index_nodes
