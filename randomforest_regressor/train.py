import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("data/insurance.csv")

# ---------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------
le = LabelEncoder()

categorical_columns = df.select_dtypes(
    include=['object']
).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df.drop("charges", axis=1)

y = df["charges"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------
model = RandomForestRegressor(

    n_estimators=200,

    criterion="squared_error",

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    max_features="sqrt",

    bootstrap=True,

    random_state=42,

    n_jobs=-1
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("MSE :", mse)

print("MAE :", mae)

print("R2 Score :", r2)

# ---------------------------------------------------
# CREATE MODELS FOLDER
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
pickle.dump(
    model,
    open(
        "models/random_forest_model.pkl",
        "wb"
    )
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

print("Model Saved Successfully")