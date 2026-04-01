from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask import render_template
 
# Load the trained model
model = joblib.load('iris_model.pkl')
 
#initialize the Flask application
app = Flask(__name__)
 
@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get json data from the request
        data = request.get_json(force=True)

        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Invalid input. Exactly 4 features required."}), 400
 
        # extract features from the json data
        features = np.array(data['features']).reshape(1, -1)
 
        # make a prediction using the loaded model
        prediction = model.predict(features)[0]
 
        # map the prediction with the corresponding iris class
        classes = ['setosa', 'versicolor', 'virginica']
        result = {"prediction": classes[prediction]}
 
        # return the prediction as a json response
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
 
if __name__ == "__main__":
    app.run(debug=True)