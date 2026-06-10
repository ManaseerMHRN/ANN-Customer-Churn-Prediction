# 📉 Customer Churn Prediction with Artificial Neural Network

A deep learning project that predicts customer churn for a telecom company using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

---

## 📋 Table of Contents

* Overview
* Dataset
* Project Workflow
* Model Architecture
* Results
* Technologies Used
* Getting Started
* Project Structure
* Notes

---

## 📖 Overview

Customer churn occurs when a customer stops using a company's services. For telecom companies, retaining customers is crucial for maintaining revenue and reducing acquisition costs.

This project develops a binary classification model using an Artificial Neural Network (ANN) to predict whether a customer is likely to churn (`Yes`) or remain with the company (`No`) based on demographic information, subscribed services, account details, and billing information.

---

## 📊 Dataset

**File:** `Customer_Churn.csv`

The dataset contains customer-level information from a telecom service provider.

| Category            | Features                                                                                                                                |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Demographics        | gender, SeniorCitizen, Partner, Dependents                                                                                              |
| Account Information | tenure, Contract, PaperlessBilling, PaymentMethod                                                                                       |
| Services            | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Billing             | MonthlyCharges, TotalCharges                                                                                                            |
| Target Variable     | Churn (Yes / No)                                                                                                                        |

> **Note:** The `customerID` column was removed because it does not provide predictive value for the model.

---

## 🔄 Project Workflow

### 1. 🔍 Data Cleaning

* Removed the `customerID` column.
* Identified and removed rows where `TotalCharges` contained blank or whitespace values.
* Converted `TotalCharges` from object type to numeric (`float`).

### 2. 📊 Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand customer behavior and churn patterns.

Key visualizations included:

* Customer tenure distribution
* Monthly charges distribution
* Churn comparison by tenure
* Churn comparison by monthly charges

**Key Insights:**

* Customers with shorter tenure are more likely to churn.
* Customers with higher monthly charges show a greater tendency to churn.

### 3. 🛠️ Feature Engineering & Encoding

* Replaced `"No phone service"` and `"No internet service"` with `"No"` for consistency.
* Converted binary categorical features (`Yes/No`) into numerical values (`1/0`).
* Encoded gender (`Female = 1`, `Male = 0`).
* Applied one-hot encoding to multi-class categorical features:

  * InternetService
  * Contract
  * PaymentMethod

### 4. ⚖️ Feature Scaling

Used **MinMaxScaler** to normalize numerical features:

* tenure
* MonthlyCharges
* TotalCharges

### 5. 🤖 Model Training

* Train-test split: **80% training / 20% testing**
* Random state: **5**
* Optimizer: **Adam**
* Epochs: **50**

### 6. 📈 Model Evaluation

The model was evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix Heatmap

---

## 🧠 Model Architecture

```text
Input Layer: 26 Features
        ↓
Dense(20, activation='relu')
        ↓
Dense(15, activation='relu')
        ↓
Dense(1, activation='sigmoid')
```

### Compilation Settings

| Parameter         | Value               |
| ----------------- | ------------------- |
| Optimizer         | Adam                |
| Loss Function     | Binary Crossentropy |
| Evaluation Metric | Accuracy            |

The sigmoid activation function produces a probability score representing the likelihood of customer churn.

---

## 📈 Results

The trained ANN model was evaluated on the held-out test dataset.

### Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

Predictions were converted into class labels using a threshold of **0.5**:

```python
prediction > 0.5  → Churn (1)
prediction ≤ 0.5  → No Churn (0)
```

The confusion matrix was visualized using a Seaborn heatmap to provide insights into classification performance.

---

## 🛠️ Technologies Used

| Technology         | Purpose                        |
| ------------------ | ------------------------------ |
| Python 3           | Core programming language      |
| Pandas             | Data loading and preprocessing |
| NumPy              | Numerical computations         |
| Matplotlib         | Data visualization             |
| Seaborn            | Statistical visualization      |
| Scikit-learn       | Preprocessing and evaluation   |
| TensorFlow / Keras | ANN model development          |
| Google Colab       | Development environment        |

---

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### Load the Dataset

If using a local environment:

```python
df = pd.read_csv("Customer_Churn.csv")
```

If using Google Colab, upload the dataset to Google Drive and update the file path accordingly.

### Run the Notebook

Open and execute:

```text
Churn_Prediction.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

---

## 📁 Project Structure

```text
churn-prediction/
│
├── Churn_Prediction.ipynb      # Main notebook
├── Customer_Churn.csv          # Dataset
└── README.md                   # Project documentation
```

---

## 📌 Notes

* This project is based on the popular **Telco Customer Churn Dataset**.
* The model was developed and trained using **Google Colab**.
* Google Drive was used for dataset storage and access.
* This project demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, neural network training, and model evaluation.

---

## 👨‍💻 Author

**Manaseer Maharutheen**

Information Technology Undergraduate | Aspiring Data Scientist

Feel free to fork this repository, explore the code, and contribute improvements.
