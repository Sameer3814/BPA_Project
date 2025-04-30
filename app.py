from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and preprocessor
model = joblib.load("model/logistic_model.pkl")       # or use "random_forest_model.pkl"
preprocessor = joblib.load("model/preprocessor.pkl")  # this is your preprocessing pipeline

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "❌ No file part in the request."

        file = request.files["file"]
        if file.filename == "":
            return "❌ No file selected."

        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read the uploaded CSV
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()  # clean up column names

            try:
                # Select only the raw features the preprocessor expects
                df_selected = df[preprocessor.feature_names_in_]

                # Apply preprocessing
                X_prepared = preprocessor.transform(df_selected)

                # Predict tax classes
                predictions = model.predict(X_prepared)

                # Map predicted class labels back to original tax classes
                label_map = {0: 1, 1: 2, 2: 3, 3: 4}
                df["Predicted_Tax_Class"] = [label_map[p] for p in predictions]

                # Save output
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], "predicted_" + filename)
                df.to_csv(output_path, index=False)

                return send_file(output_path, as_attachment=True)

            except Exception as e:
                return f"❌ Error: {str(e)}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
