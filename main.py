import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    "spam.csv",
    sep="\t",
    names=["label", "message"]
)

# Remove missing values
data = data.dropna()

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Features and target
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test new messages
while True:
    msg = input("\nEnter message (or type 'exit' to quit): ")

    if msg.lower() == "exit":
        print("Program Closed.")
        break

    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("Result: SPAM")
    else:
        print("Result: NOT SPAM")