from brics_tools.brics_modules.data_dictionary.crossmap.llamaindex_setup.index_managers.base import (
    DocumentCreator,
)


class UMLSIndexManager(IndexManager):
    def __init__(self, config, service_context):
        self.config = config
        self.service_context = service_context
        self.indices = {}

    def initialize_vector_store(self, storage_type, storage_path, col):
        if storage_type == "chromadb":
            client = chromadb.PersistentClient(path=storage_path)
            chroma_collection = client.get_or_create_collection(
                col, metadata=dict(self.config["distance_metric"])
            )
            return ChromaVectorStore(chroma_collection=chroma_collection)
        elif storage_type == "pinecone":
            # initialize PineconeVectorStore
            pass

    def create_index(self, index_nodes):
        storage_path = self.config["storage_path_root"]
        storage_type = self.config.get("storage_type", "chromadb")

        for col, nodes in index_nodes.items():
            vector_store = self.initialize_vector_store(storage_type, storage_path, col)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                service_context=self.service_context,
                show_progress=True,
            )

            # Add metadata
            index.summary = f"{self.config['summary']} {col}"
            index.set_index_id(col)

            # Persist index
            storage_path_index = Path(storage_path, col).as_posix()
            index.storage_context.persist(storage_path_index)

            self.indices[col] = index

        return self.indices
