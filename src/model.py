import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/trafic.csv")

# Clean columns
df.columns = df.columns.str.strip()

# Handle missing/infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode labels
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# Features + target
X = df.drop(columns=["Label"], errors="ignore")
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, pred))

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("CV Accuracy:", scores.mean())

# Save model + feature columns
joblib.dump(model, "ddos_model.pkl")
joblib.dump(X.columns, "features.pkl")