import gradio as gr
import tweetnlp

# Load the sentiment analysis model
model = tweetnlp.load_model('sentiment', multilingual=True)

# Define function to run prediction
def analyze_sentiment(text):
    if not text.strip():
        return "Error: Text cannot be empty"
    try:
        result = model.sentiment(text.strip(), return_probability=True)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Create a Gradio Interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="json",
    title="Multilingual Sentiment Analyzer"
)

iface.launch()