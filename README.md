# Placement Prediction using Machine Learning

This project predicts whether a student will be placed based on features such as CGPA and IQ. It uses machine learning classification techniques and provides a trained model for future predictions.

# 📂 Project Structure
├── placement.csv              # Dataset used for training
├── placement_data_ml.ipynb    # Jupyter Notebook with training code
├── model.pkl                  # Saved trained model (pickle file)
├── README.md                  # Project documentation

# 🚀 Key Features

1. Exploratory Data Analysis (EDA): Visualizes trends between CGPA, IQ, and placement outcomes.
2. Machine Learning Model: Trained classifier (e.g., Logistic Regression / SVM / Decision Tree).
3. Model Persistence: Saves the trained model using pickle for reuse.
4. Evaluation Metrics: Model is tested with accuracy scores and confusion matrix.
5. Quick Predictions: The model.pkl can instantly predict placement for new student data.
6. Extendable: Can be deployed as a Flask, FastAPI, or Streamlit app.

# 📊 Dataset

The dataset (placement.csv) contains student details including:

1. CGPA
2. IQ
3. Placement status (Placed / Not Placed)

# 🛠️ Workflow

1. Load Dataset → Import and clean placement data (placement.csv).
2. Data Visualization → Scatter plots of CGPA vs IQ, color-coded by placement.
3. Model Training → Train a classifier on training data.
4. Evaluation → Test accuracy and visualize decision boundaries.
5. Save Model → Export trained model as model.pkl for future use.
6. Prediction → Load model and predict placement for new inputs.
   
# 📖 Usage
1️⃣ Train the Model
1. Open the notebook and run all cells:
   jupyter notebook placement_data_ml.ipynb

2️⃣ Use the Saved Model
1. You can load the trained model (model.pkl) and make predictions:
   import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Example prediction
print(model.predict([[8.5, 120]]))  # [1] means placed, [0] means not placed

# 📦 Requirements

1. Python 3.x
2. numpy
3. pandas
4. matplotlib
5. scikit-learn
6. mlxtend (optional, for decision boundary plots)
Install them with:

pip install numpy pandas matplotlib scikit-learn mlxtend

🔮 Future Work

1. Deploy the model using Flask/FastAPI/Streamlit for a web app
2. Improve accuracy with hyperparameter tuning
3. Try different ML algorithms
