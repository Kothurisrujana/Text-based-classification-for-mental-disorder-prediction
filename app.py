from flask import Flask, render_template_string, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# -------------------------------
# 1️⃣ Load Model & Tokenizer
# -------------------------------
model = tf.keras.models.load_model("mental_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------------
# 2️⃣ Clean Text Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# 3️⃣ Prediction Function
# -------------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(pad, verbose=0)[0][0]

    if pred >= 0.5:
        label = "😔 Distressed / Negative"
        color = "#ff4d4d"
    else:
        label = "🙂 Normal / Positive"
        color = "#4CAF50"

    confidence = round(float(pred), 3)
    return label, confidence, color

# -------------------------------
# 4️⃣ Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 5️⃣ HTML Template (Modern UI)
# -------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LSTM Emotion Analyzer 💭</title>
    <style>
        body {
            margin: 0;
            height: 100vh;
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            animation: gradientShift 8s infinite alternate;
        }
        @keyframes gradientShift {
            0% { background: linear-gradient(135deg, #89f7fe, #66a6ff); }
            100% { background: linear-gradient(135deg, #667eea, #764ba2); }
        }
        .container {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 40px;
            width: 450px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            text-align: center;
            color: white;
            transition: 0.3s;
        }
        .container:hover {
            transform: scale(1.03);
        }
        h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        textarea {
            width: 90%;
            height: 120px;
            padding: 10px;
            border-radius: 10px;
            border: none;
            outline: none;
            resize: none;
            font-size: 16px;
            margin-top: 20px;
            color: #333;
        }
        button {
            background: linear-gradient(45deg, #43cea2, #185a9d);
            border: none;
            color: white;
            padding: 10px 25px;
            font-size: 16px;
            border-radius: 10px;
            margin-top: 15px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            transform: scale(1.05);
            background: linear-gradient(45deg, #6dd5ed, #2193b0);
        }
        .result {
            margin-top: 25px;
            font-size: 20px;
            font-weight: bold;
        }
        .confidence {
            font-size: 16px;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💬 Mental Health Sentiment Analyzer</h1>
        <form method="POST">
            <textarea name="text" placeholder="Type your thoughts here..." required></textarea><br>
            <button type="submit">Analyze 💡</button>
        </form>

        {% if label %}
        <div class="result" style="color: {{ color }};">
            Prediction: {{ label }}
        </div>
        <div class="confidence">
            Confidence: {{ confidence }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# -------------------------------
# 6️⃣ Flask Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    label, confidence, color = None, None, None
    if request.method == "POST":
        user_text = request.form["text"]
        label, confidence, color = predict_sentiment(user_text)
    return render_template_string(HTML_TEMPLATE, label=label, confidence=confidence, color=color)

# -------------------------------
# 7️⃣ Run the App
# -------------------------------
import webbrowser

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)

