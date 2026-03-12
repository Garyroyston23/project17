from preprocessing.preprocess import preprocess_data
from models.random_forest_model import RandomForestModel

def main():

    # load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    model = RandomForestModel()

    # Train model for Type2
    model.train(X_train, y_train["Type2"])

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    model.print_results(y_test["Type2"], predictions)

if __name__ == "__main__":
    main()
