from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join('model', 'disease_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Retrieve form data
        features = [float(request.form[key]) for key in request.form]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)