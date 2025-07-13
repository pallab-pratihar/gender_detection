from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('my_model.h5') # Update with your model path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        if file:
            # Read the image using OpenCV
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (200, 200)) # Resize to match model input

            # Preprocess the image
            img = img.reshape((1, 200, 200, 3))
            img = img / 255.0  # Normalize

        prediction = model.predict(img)
        print("Raw Model Output:", prediction)  # Debugging step

        # Convert logits to probabilities (if necessary)
        if prediction.shape[-1] == 1:  # Binary classification case
            probability = tf.nn.sigmoid(prediction).numpy()[0][0]
            print("Sigmoid Output:", probability)  # Debugging step
            result = 'Male' if probability < 0.5 else 'Female'  # Switched threshold
        else:  # If it's categorical classification
            result = 'Male' if np.argmax(prediction) == 1 else 'Female'

        print("Final Prediction:", result)  # Debugging step
        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)