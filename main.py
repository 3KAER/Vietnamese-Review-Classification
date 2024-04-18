import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def predict_sentiment(text, negative_threshold=0.5):
    model = tf.keras.models.load_model('C:/Users/nguye/Downloads/Vietnamese-Review-Classification/models/review_model.keras')

    tokenizer = None
    with open("C:/Users/nguye/Downloads/Vietnamese-Review-Classification/models/review_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    max_len = 200
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    prediction = model.predict(padded_sequence)
    class_probabilities = prediction[0]

    # Lấy phần trăm tiêu cực
    for i, prob in enumerate(class_probabilities):
        if i == 0:
            print(f"Positive: {prob * 100:.2f}%")
        elif i == 1:
            print(f"Negative: {prob * 100:.2f}%")
        elif i == 2:
            print(f"Neutral: {prob * 100:.2f}%")
    predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]
    if predicted_class == 1:
        print("Predicted Sentiment: Negative")
    elif predicted_class == 0:
        print("Predicted Sentiment: Positive")
    elif predicted_class == 2:
        print("Predicted Sentiment: Neutral")

predict_sentiment('nứng')


