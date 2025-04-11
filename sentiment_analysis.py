import gradio as gr
from flask import Flask, request, jsonify
import tweetnlp
import os

# Initialize Flask app
app = Flask(__name__)

# Load the sentiment analysis model
model = tweetnlp.load_model('sentiment', multilingual=True)

# Define the sentiment analysis function
def analyze_sentiment(text):
    if not text.strip():
        return "Error: Text cannot be empty"
    try:
        result = model.sentiment(text.strip(), return_probability=True)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="json",
    title="Multilingual Sentiment Analyzer"
)

# Run the Flask app with Gradio
if __name__ == "__main__":
    iface.launch(share=True)