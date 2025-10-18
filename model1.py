import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pickle
from flask import Flask, request, jsonify
# Load data
df = pd.read_csv("mental_health (2).csv")

# Basic cleaning
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"[^A-Za-z\s]", "", str(text))
    text = text.lower().strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

text = df["clean_text"].values
labels = df["label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    SpatialDropout1D(0.3),
    LSTM(128, dropout=0.3, recurrent_dropout=0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, 100))
model.summary()

history = model.fit(
    X_train_pad,
    y_train,
    epochs=10,                # increased epochs
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    class_weight=class_weights,
    verbose=2
)

def predict_sentiment(text):
    text = clean_text(text)
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=100, padding='post')
    sentiment = model.predict(text_pad, verbose=0)[0][0]
    label = "Distressed / Negative" if sentiment >= 0.6 else "Normal / Positive"
    print(f"Predicted: {label} ({sentiment:.3f} confidence)")
    return label
predict_sentiment("I’m so frustrated and angry about the situation.")
# Save the model (architecture + weights)
model.save("mental_model.h5")  # or "mental_model.keras" for newer format
import pickle

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
