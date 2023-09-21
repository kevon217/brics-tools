from abc import ABC, abstractmethod


class DocumentCreator(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def create_documents(self, data_frame):
        """
        Abstract method for creating documents for a specific type of index.
        Should be implemented by each subclass.

        Parameters:
        - data_frame: The DataFrame containing the data for which documents will be created.

        Returns:
        A list of documents (or any other suitable data structure) ready for indexing.
        """
        pass
