# Chronic-Kidney-Disease-Detection
Early Prediction for Chronic Kidney Disease Detection: A Progressive Approach to Health Management

**📌 Objective**

This project is a complete end-to-end Machine Learning application that predicts whether a patient is likely to have Chronic Kidney Disease (CKD) based on medical input features. The project combines data science, model training, evaluation, and web deployment using Flask.

---

**What I did**

  - Comprehensive data cleaning and preprocessing of CKD dataset
  - In-depth Exploratory Data Analysis (EDA) with visualizations
  - Implementation and comparison of multiple ML models:
      - Logistic Regression
      - Decision Tree
      - Random Forest
      - Artificial Neural Network (ANN using Keras + TensorFlow)
  - Model evaluation using metrics such as:
      - Accuracy
      - Confusion Matrix
  - Selection and serialization of the best-performing model using pickle
  - Flask web application for real-time prediction
  - Modular, readable, and extensible code structure

---

**Model Training Pipeline**

1. Data Preprocessing
   - Handling missing values
   - Label encoding categorical variables
   - Feature scaling where necessary
2. Exploratory Data Analysis
   - Pairwise relationships
   - Correlation heatmaps
   - Visual feature inspection
3. Model Training
   - Trained on four classification models
   - Evaluated using accuracy & confusion matrix
   - Neural Network built using Keras for advanced prediction
4. Model Saving
   - The best-performing model is saved using pickle
5. Web Deployment
   - Flask app receives user inputs through a web form
   - Predicts and displays output to the user in real-time

---

**🛠️ Technologies Used**

  - Python
  - Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras, TensorFlow
  - Machine Learning (Classification models)
  - Flask
  - Pickle
  - Jupyter Notebook

---

**Dataset**

  - File: kidney_disease.csv
  - Features include clinical measurements like:
      - Age, Blood Pressure, Albumin, Hemoglobin, Serum Creatinine, etc.


