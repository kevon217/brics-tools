from pathlib import Path
from llama_index import (
    LangchainEmbedding,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from llama_index import load_index_from_storage


class StudyInfoVectorStoreIndexLoader:
    def __init__(self, config):
        self.config = config

    def load_index(self, index_type):
        # Validate index_type
        if index_type not in self.config.index_managers.keys():
            raise ValueError(
                f"Invalid index_type: {index_type}. Must be one of {list(self.config.index_managers.keys())}"
            )

        # Get the specific configuration for this index_type
        index_config = self.config.index_managers.get(index_type)

        # Initialize a StorageContext for the given persist_dir
        storage_context = StorageContext.from_defaults(
            persist_dir=index_config.storage_context.storage_path_root
        )

        # Load index from storage
        return load_index_from_storage(storage_context=storage_context)
