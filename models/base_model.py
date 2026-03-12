from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def print_results(self, y_true, y_pred):
        pass
