import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data():

    df = pd.read_csv("emails.csv")

    X = df["email_text"]
    y = df[["Type2","Type3","Type4"]]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
