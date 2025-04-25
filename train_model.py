import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import os

# Load data
data = pd.read_csv("spam.csv", encoding='latin-1')

# Print actual column names (for debugging)
print(data.columns)

# Select only needed columns and rename
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Encode labels: spam = 1, ham = 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the trained model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
