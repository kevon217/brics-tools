from llama_index import Document, StorageContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    EntityExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.schema import MetadataMode
from llama_index.text_splitter import TokenTextSplitter

from brics_tools.index_tools.document_creators.base import DocumentCreator


class StudyInfoDocumentCreator(DocumentCreator):
    def __init__(self, config):
        super().__init__(config)
        self.documents = []

    def create_documents(self, df_studyinfo):
        doc_id_col = self.config.id_column
        text_col = self.config.text_column

        # Convert DataFrame columns to a list
        all_columns = df_studyinfo.columns.tolist()

        # Determine which metadata columns to include
        if self.config.metadata_include.method == "auto":
            metadata_cols_include = list(set(all_columns) - set([text_col, doc_id_col]))
        else:
            metadata_cols_include = list(self.config.metadata_include.columns)

        # Determine which metadata columns to exclude
        if self.config.metadata_exclude.method == "auto":
            metadata_cols_exclude = list(
                set(all_columns).difference(set([text_col] + metadata_cols_include))
            )
        else:
            metadata_cols_exclude = list(self.config.metadata_exclude.columns)

        documents = []
        for idx, row in df_studyinfo.iterrows():
            doc = row[text_col]
            meta = {val: row[val] for val in all_columns if val not in text_col}
            document = Document(
                text=doc,
                metadata=meta,
                excluded_embed_metadata_keys=metadata_cols_exclude,
                excluded_llm_metadata_keys=metadata_cols_exclude,
                metadata_seperator=self.config.metadata_separator,
                metadata_template=self.config.metadata_template,
                text_template=self.config.text_template,
            )
            document.id_ = row[doc_id_col]
            documents.append(document)

        return documents
