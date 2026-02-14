# Heart-Disease-ML-model
Trained a model to predict the Heart Disease.

🎯 Problem Statement
Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention.
In this project, a machine learning model is trained to classify whether a patient is at risk of heart disease.

🧠 Algorithm Used
K-Nearest Neighbors (KNN)
Value of K = 10
Distance metric: Euclidean (default in sklearn)

📂 Dataset
Dataset: Heart Disease Risk Dataset
Type: Binary Classification (0 = No Disease, 1 = Disease)
Features: Medical and demographic attributes

🛠️ Tech Stack
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

⚙️ Project Workflow
Data Loading
Data Preprocessing
Train-Test Split
Model Training using KNN
Model Evaluation
Performance Visualization

📊 Model Performance
Metric	Score
Test Accuracy	99.30%
Precision	0.99
Recall	0.99
F1-Score	0.99

✅ Train and test accuracy are very close, indicating low overfitting.
📈 Evaluation Metrics Used
Accuracy Score
Classification Report
Confusion Matrix

Example:
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, knn_pred))

🙌 Acknowledgement
This project was created for learning and educational purposes to strengthen machine learning fundamentals.

📬 Connect with Me
If you found this project helpful, feel free to ⭐ the repo and connect with me on LinkedIn!

⭐ Don't forget to star the repository if you like it!
