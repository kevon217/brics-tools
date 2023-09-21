import chromadb
from brics_tools.utils import helper

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


client = chromadb.PersistentClient(
    path=cfg.index_managers.studyinfo_vectorstore_index.storage_context.storage_path_root
)
chroma_collection = client.get_or_create_collection(
    cfg.index_managers.studyinfo_vectorstore_index.index_id
)
print(f"Collection: {chroma_collection}")
print(f"Count: {chroma_collection.count()}")

docs = chroma_collection.get(include=["documents", "metadatas"])


md = docs["metadatas"]

import pandas as pd


def list_of_dicts_to_dataframe(list_of_dicts):
    df = pd.DataFrame()
    for entry in list_of_dicts:
        temp_df = pd.DataFrame(
            [entry]
        )  # Convert the dictionary to a single-row DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)

    return df

    return df


df = list_of_dicts_to_dataframe(md)

df_ids = list(df_studyinfo["id"].unique())
ids = [doc["doc_id"] for doc in docs["metadatas"]]
# find ids not in df_ids
missing_ids = [int(id) for id in ids if int(id) not in df_ids]

doc_to_update = chroma_collection.get()
doc_to_update["metadatas"][0] = {
    **doc_to_update["metadatas"][0],
    **{"author": "Paul Graham"},
}
chroma_collection.update(
    ids=[doc_to_update["ids"][0]], metadatas=[doc_to_update["metadatas"][0]]
)
updated_doc = chroma_collection.get(limit=1)
print(updated_doc["metadatas"][0])

# delete the last document
print("count before", chroma_collection.count())
chroma_collection.delete(ids=[doc_to_update["ids"][0]])
print("count after", chroma_collection.count())
