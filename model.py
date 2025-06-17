import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

print("🔄 Step 1: Reading the CSV file...")
try:
    df = pd.read_csv("movies.csv")
    print("✅ CSV loaded successfully.")
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

print("🔄 Step 2: Splitting data...")
X = df['description']
y = df['genre']

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data split done.")
except Exception as e:
    print("❌ Error in train_test_split:", e)
    exit()

print("🔄 Step 3: Creating pipeline...")
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

print("🔄 Step 4: Training model...")
try:
    model.fit(X_train, y_train)
    print("✅ Model trained successfully.")
except Exception as e:
    print("❌ Error during model training:", e)
    exit()

print("💾 Step 5: Saving model...")
try:
    joblib.dump(model, "genre_model.pkl")
    print("✅ Model saved as 'genre_model.pkl'.")
except Exception as e:
    print("❌ Error saving model:", e)

print("📊 Step 6: Checking accuracy...")
try:
    accuracy = model.score(X_test, y_test)
    print(f"🎯 Model Accuracy: {accuracy:.2f}")
except Exception as e:
    print("❌ Error calculating accuracy:", e)

