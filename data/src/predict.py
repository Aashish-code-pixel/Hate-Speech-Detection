import joblib

# Load saved model and vectorizer
model = joblib.load("models/hate_speech_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Function to predict text
def predict_hate_speech(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

# Test prediction
text = input("Enter text: ")
print("Prediction:", predict_hate_speech(text))
