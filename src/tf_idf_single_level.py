"""This file does a single level logistic regression."""
import click
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, accuracy_score


# Load and preprocess
def load_and_preprocess(input_filepath):
    input_filepath = Path(input_filepath).resolve()
    print(f"{input_filepath=}")
    data = pd.read_json(input_filepath)
    print(f"{data.columns=}")
    data = data[["text", "tags"]]
    print(f"{data.head(2)=}")
    return data


def train(data, threshold=0.5, print_conf=5):
    # Split the data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Extract text and tags
    X_train, y_train = train_df["text"], train_df["tags"]
    X_val, y_val = val_df["text"], val_df["tags"]
    X_test, y_test = test_df["text"], test_df["tags"]

    # Binarize
    mlbin = MultiLabelBinarizer()
    y_train_bin = mlbin.fit_transform(y_train)
    y_val_bin = mlbin.transform(y_val)
    y_test_bin = mlbin.transform(y_test)

    # Pipeline with tf-idf and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced'), n_jobs=-1))
    ])

    # Train the model
    pipeline.fit(X_train, y_train_bin)

    # # Predict the text
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # # Calculate probabilities
    # y_prob = pipeline.predict_proba(X_test)
    # # Thresholding to get confidence scores
    # y_pred_bin = (y_prob >= threshold).astype(int)
    #
    # # Calculate confidence scores for each tag
    # confidence_scores = {tag: y_prob[:, i] for i, tag in enumerate(mlbin.classes_)}
    #
    # # Convert binary predictions to multilabel format
    # y_pred = mlbin.transform(y_pred_bin)
    # y_test_labels = mlbin.inverse_transform(y_test_bin)

    # Evaluation
    print(f"Val Classification Report")
    print(classification_report(y_val_bin, y_val_pred, target_names=mlbin.classes_))
    print(f"Test Classification Report")
    print(classification_report(y_test_bin, y_test_pred, target_names=mlbin.classes_))

    # Accuracy
    val_accuracy = accuracy_score(y_val_bin, y_val_pred)
    test_accuracy = accuracy_score(y_test_bin, y_test_pred)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Hamming Loss
    val_hamming_loss = hamming_loss(y_val_bin, y_val_pred)
    test_hamming_loss = hamming_loss(y_test_bin, y_test_pred)
    print(f"Validation Hamming Loss: {val_hamming_loss:.2f}")
    print(f"Test Hamming Loss: {test_hamming_loss:.2f}")

    def get_confidence_estimates(pipeline, X, threshold=0.5):
        probabilities = pipeline.decision_function(X)
        confidence_scores = np.max(probabilities, axis=1)
        predictions = (probabilities > threshold).astype(int)
        return predictions, confidence_scores

    y_test_pred, confidence_scores = get_confidence_estimates(pipeline, X_test)
    print(f"{confidence_scores=}")


@click.command()
@click.option(
    "--input-filepath",
    type=str,
    default="/Users/ananya.lahiri/PycharmProjects/SigmaProject/Users/ananya.lahiri/output_sigma/selected_fields_driver_load/rules/windows/driver_load/extracted_data.json",
    help="Location of input datafile",
)
@click.option(
    "--threshold",
    default=0.5,
    help="Threshold",
)
def run_main(
        input_filepath,
        threshold,
):
    data = load_and_preprocess(input_filepath)

    train(data=data, threshold=threshold)


if __name__ == "__main__":
    run_main()




