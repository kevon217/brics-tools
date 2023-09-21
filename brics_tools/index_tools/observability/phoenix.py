import pandas as pd
import numpy as np

from dotenv import load_dotenv
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
import phoenix as px

from brics_tools.utils import helper


cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


def vector_index_to_dataframe(storage_path_root, index_id) -> pd.DataFrame:
    storage_context = StorageContext.from_defaults(persist_dir=storage_path_root)
    index = load_index_from_storage(storage_context)
    docs = vec_db.get(include=["embeddings", "documents"])
    document_ids = docs["ids"]
    document_texts = docs["documents"]

    embeddings = [np.array(x) for x in docs["embeddings"]]
    document_embeddings = embeddings
    return pd.DataFrame(
        {
            "document_id": document_ids,
            "text": document_texts,
            "text_vector": document_embeddings,
        }
    )


index_id = cfg.index_managers.studyinfo_vectorstore_index.index_id
storage_path_root = (
    cfg.index_managers.studyinfo_vectorstore_index.storage_context.storage_path_root
)


database_df = vector_index_to_dataframe(
    storage_path_root=storage_path_root, index_id=index_id
)
# database_df = database_df.drop_duplicates(subset=["text"])
database_df.head()

database_schema = px.Schema(
    prediction_id_column_name="document_id",
    prompt_column_names=px.EmbeddingColumnNames(
        vector_column_name="text_vector",
        raw_data_column_name="text",
    ),
)
database_ds = px.Dataset(
    dataframe=database_df,
    schema=database_schema,
    name="fitbir-data-repository",
)

session = px.launch_app(database_ds)
