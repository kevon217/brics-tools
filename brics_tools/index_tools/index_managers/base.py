from abc import ABC, abstractmethod


class IndexManager(ABC):
    @abstractmethod
    def create_index(self, index_nodes):
        pass
