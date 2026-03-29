import pandas as pd
import numpy as np
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dl/spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0,'spam':1})

# URL Detection Feature
def detect_url(text):
    url_pattern = r'https?://\S+|www\.\S+'
    return 1 if re.search(url_pattern, text) else 0

data['url_feature'] = data['message'].apply(detect_url)

X = data['message']
y = data['label']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

# Padding
X_pad = pad_sequences(sequences, maxlen=100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# LSTM Model
model = Sequential()
model.add(Embedding(5000,64,input_length=100))
model.add(LSTM(64))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(X_train,y_train,epochs=5,batch_size=32)

# Evaluate model
loss,accuracy = model.evaluate(X_test,y_test)
print("Model Accuracy:",accuracy)

# Prediction Function
def predict_message(msg):

    url = detect_url(msg)

    seq = tokenizer.texts_to_sequences([msg])
    padded = pad_sequences(seq,maxlen=100)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5 or url == 1:
        print("Result: SPAM MESSAGE")
    else:
        print("Result: NOT SPAM")


# -------- Real-Time Message Input --------

print("\nSpam Detection System Ready!")
print("Type a message to check if it is spam.")
print("Type 'exit' to stop.\n")

while True:

    msg = input("Enter Message: ")

    if msg.lower() == "exit":
        print("Program stopped.")
        break

    print("\nMessage:", msg)
    predict_message(msg)