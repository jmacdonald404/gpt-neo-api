from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Initialize the model and tokenizer as None, to load them lazily
model = None
tokenizer = None
model_loaded = False  # Flag to ensure model is only loaded once

def load_model():
    """Load the model and tokenizer if they are not already loaded."""
    global model, tokenizer, model_loaded
    if not model_loaded:
        print("Loading model...")  # Debug log to confirm the model is being loaded
        model_name = "distilgpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model loaded!")  # Debug log
        model_loaded = True

@app.before_request
def ensure_model_loaded():
    """Ensure the model is loaded before handling any request."""
    load_model()

@app.route("/chat", methods=["POST"])
def chat():
    global model, tokenizer

    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not loaded"}), 500

    # Get the user input from the request
    user_message = request.json.get("message", "")

    # Tokenize the input and generate a response
    inputs = tokenizer.encode(user_message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # Decode the output tokens and return as a response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
