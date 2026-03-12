from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from models.base_model import BaseModel

class RandomForestModel(BaseModel):

    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def print_results(self, y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
