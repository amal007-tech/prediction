from flask import Flask, render_template, request
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

app = Flask(__name__)

# Load the diabetes dataset
diabetes_data = pd.read_csv("diabetes.csv")  # Replace with the actual path to your diabetes dataset

# Sample AdaBoost classifier for diabetes
diabetes_features = diabetes_data.drop("Outcome", axis=1)
diabetes_target = diabetes_data["Outcome"]
diabetes_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
diabetes_classifier.fit(diabetes_features, diabetes_target)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/disease_selection")
def disease_selection():
    return render_template("disease_selection.html")

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        # Update the features list to include all the features used during training
        features = ["pregnancy", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "Age"]

        # Verify that the length of the features list matches the number of columns in diabetes_features
        if len(features) != diabetes_features.shape[1]:
            raise ValueError(f"Number of features in the input does not match the model's expectations. Expected {diabetes_features.shape[1]} features.")

        input_data = [float(request.form[feature]) for feature in features]

        # Print debugging information
        print("Received request data:", request.form)
        print("Length of input_data:", len(input_data))
        print("Columns of diabetes_features:", diabetes_features.columns)

        # Verify that the length of the input_data matches the number of features
        if len(input_data) != len(features):
            raise ValueError(f"Length of input_data does not match the number of features. Expected {len(features)} features.")

        prediction = diabetes_classifier.predict([input_data])[0]
        

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        print("Error:", str(e))
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
