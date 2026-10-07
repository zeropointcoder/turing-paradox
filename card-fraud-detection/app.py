import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Data Generator class
class DataGenerator:
    def __init__(self, n_samples=5000, fraud_ratio=0.05, random_state=42):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state

    def generate(self):
        np.random.seed(self.random_state)
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_legit = self.n_samples - n_fraud

        # Features: amount, transaction_time, customer_age, etc.
        amount_legit = np.random.normal(loc=50, scale=30, size=n_legit)
        amount_fraud = np.random.normal(loc=200, scale=100, size=n_fraud)
        amount = np.concatenate([amount_legit, amount_fraud])

        time_legit = np.random.uniform(0, 24, n_legit)
        time_fraud = np.random.uniform(0, 24, n_fraud)
        time = np.concatenate([time_legit, time_fraud])

        age_legit = np.random.normal(35, 10, n_legit)
        age_fraud = np.random.normal(40, 12, n_fraud)
        age = np.concatenate([age_legit, age_fraud])

        label = np.array([0]*n_legit + [1]*n_fraud)

        data = pd.DataFrame({
            'amount': amount,
            'transaction_time': time,
            'customer_age': age,
            'fraud': label
        })

        return data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)


# Model Trainer class
class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_data(self):
        X = self.data.drop('fraud', axis=1)
        y = self.data['fraud']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        return X_test, y_test


# Evaluator class
class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        return report, matrix, roc_auc


# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")

st.markdown("""
This project demonstrates a credit card fraud detection system using a Random Forest Classifier.
All data is **synthetically generated** and processed offline.
""")

# Generate Data
generator = DataGenerator()
data = generator.generate()

if st.checkbox("Show raw data"):
    st.dataframe(data.head(20), width=800)

# Train Model
trainer = ModelTrainer(data)
X_test, y_test = trainer.train()

# Evaluate
evaluator = Evaluator(trainer.model, X_test, y_test)
report, matrix, roc_auc = evaluator.evaluate()

st.subheader("Model Performance")
st.write("Confusion Matrix:")
st.write(matrix)
st.write("ROC AUC Score:", round(roc_auc, 4))
st.write("Classification Report:")
st.json(report)