import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss


# Load and preprocess
def load_and_preprocess():
    data = pd.read_json('path_to_sigma_rules.json') # complete this
    data['text'] = data['title'] + ' ' + data['description'] + ' ' + data['detection']
    return data


data = load_and_preprocess()  # TODO


# Split the data
def train(data, print_conf=5, threshold=0.5):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['tags'], test_size=0.2, random_state=42)
    # Multi|LabelBinarizer for multi label classification
    mlbin = MultiLabelBinarizer()
    y_train_bin = mlbin.fit_transform(y_train)
    y_test_bin = mlbin.transform(y_test)

    # Pipeline with tf-idf and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Train the model
    pipeline.fit(X_train, y_train_bin)

    # Predict the text
    y_pred_bin = pipeline.predict(X_test)

    # Calculate probabilities
    y_prob = pipeline.predict_proba(X_test)
    # Thresholding to get confidence scores
    y_pred_bin = (y_prob >= threshold).astype(int)

    # Calculate confidence scores for each tag
    confidence_scores = {tag: y_prob[:, i] for i, tag in enumerate(mlbin.classes_)}

    # Convert binary predictions to multilabel format
    y_pred = mlbin.transform(y_pred_bin)
    y_test_labels = mlbin.inverse_transform(y_test_bin)

    # Evaluation
    print(classification_report(y_test_bin, y_pred_bin, target_names=mlbin.classes_))
    print(f"Hamming loss: {hamming_loss(y_test_bin, y_pred_bin)}")
    for i in range(print_conf):
        print(f"For sample {i}: predicted_tags {y_pred[i]}")
        print(f"Confidence scores: {[confidence_scores[tag][i] for tag in y_pred[i]]}")







