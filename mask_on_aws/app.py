from flask import Flask, render_template, request
import joblib
from data_processing_and_features import text_data_cleaning, tfidf_features_transform

app = Flask(__name__)

# Load artifacts from models/ (inside container or local run)
try:
    model = joblib.load('models/model_classifier.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf_vectorizer = None

@app.route('/')
def index():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('Review', '')
    cleaned_review = text_data_cleaning(review)
    if model and tfidf_vectorizer:
        vectorized = tfidf_features_transform(cleaned_review, tfidf_vectorizer)
        prediction = model.predict(vectorized)
        output = prediction[0]
        return render_template('index.html', prediction_text=f'Sentiment is {output}')
    return render_template('index.html', prediction_text='Model or Vectorizer not found.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
