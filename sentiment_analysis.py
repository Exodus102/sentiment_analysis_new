from flask import Flask, request, jsonify
from flask_cors import CORS
import tweetnlp
import os  


model = tweetnlp.load_model('sentiment', multilingual=True)


app = Flask(__name__)
CORS(app)  

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    user_input = data['text'].strip()
    if not user_input:
        return jsonify({'error': 'Text cannot be empty'}), 400

    try:
     
        result = model.sentiment(user_input, return_probability=True)
        return jsonify({'text': user_input, 'sentiment': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

