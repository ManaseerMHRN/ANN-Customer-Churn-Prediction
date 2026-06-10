Customer Churn Prediction with Artificial Neural Network
A deep learning project that predicts customer churn for a telecom company using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

📋 Table of Contents

Overview
Dataset
Project Workflow
Model Architecture
Results
Technologies Used
Getting Started
Project Structure


Overview
Customer churn — when a customer stops using a service — is a critical business problem for telecom companies. This project builds a binary classification model to predict whether a customer will churn (Yes) or stay (No) based on their account information, services subscribed, and billing details.

Dataset
File: Customer_Churn.csv
The dataset contains customer-level information for a telecom provider. Key features include:
CategoryFeaturesDemographicsgender, SeniorCitizen, Partner, DependentsAccount Infotenure, Contract, PaperlessBilling, PaymentMethodServicesPhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMoviesBillingMonthlyCharges, TotalChargesTargetChurn (Yes / No)

Note: The customerID column is dropped as it carries no predictive value.


Project Workflow
1. 🔍 Data Cleaning

Removed the customerID column
Identified and dropped rows where TotalCharges contained blank/whitespace values (11 rows removed)
Converted TotalCharges from object to float

2. 📊 Exploratory Data Analysis (EDA)

Plotted histograms comparing churned vs. retained customers by:

tenure — customers with shorter tenure are more likely to churn
MonthlyCharges — higher charges correlate with higher churn



3. 🛠️ Feature Engineering & Encoding

Replaced "No phone service" and "No internet service" with "No" for consistency
Label-encoded binary Yes/No columns to 1/0
Label-encoded gender (Female=1, Male=0)
One-hot encoded multi-class columns: InternetService, Contract, PaymentMethod

4. ⚖️ Feature Scaling

Applied MinMaxScaler to normalize continuous features: tenure, MonthlyCharges, TotalCharges

5. 🤖 Model Training

80/20 train-test split (random_state=5)
Trained an ANN for 50 epochs using the Adam optimizer

6. 📈 Evaluation

Evaluated using accuracy, classification report (precision, recall, F1-score), and a confusion matrix heatmap


Model Architecture
Input Layer:  26 features
      ↓
Dense(20, activation='relu')
      ↓
Dense(15, activation='relu')
      ↓
Dense(1,  activation='sigmoid')   ← Binary output (Churn probability)
Compilation settings:

Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy


Results
The model is evaluated on the held-out test set (20% of data). Performance metrics include:

Accuracy — from model.evaluate()
Classification Report — precision, recall, and F1-score per class
Confusion Matrix — visualized as a heatmap using Seaborn


Predictions are thresholded at 0.5: probability > 0.5 → Churn (1), otherwise No Churn (0).


Technologies Used
ToolPurposePython 3Core languagePandasData loading & manipulationNumPyNumerical operationsMatplotlib & SeabornData visualizationScikit-learnPreprocessing, splitting, evaluationTensorFlow / KerasANN model building & trainingGoogle ColabDevelopment environment

Getting Started
Prerequisites
bashpip install pandas numpy matplotlib seaborn scikit-learn tensorflow
Running the Notebook

Clone this repository:

bash   git clone https://github.com/your-username/churn-prediction.git
   cd churn-prediction

Upload Customer_Churn.csv to your Google Drive (if using Colab), or update the file path in the notebook to your local path:

python   df = pd.read_csv("Customer_Churn.csv")  # local path

Open and run Churn_Prediction.ipynb in Jupyter Notebook or Google Colab.


Project Structure
churn-prediction/
│
├── Churn_Prediction.ipynb   # Main notebook
├── Customer_Churn.csv       # Dataset (add your own copy)
└── README.md                # Project documentation

📌 Notes

The dataset used is the popular Telco Customer Churn dataset (available on Kaggle).
The model was developed and trained on Google Colab using Google Drive for data storage.
