# bgmi_plus_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("bgmi_ranking_final_with_crown_and_type.csv")

# Features and target
X = data.drop("plus_minus_points", axis=1)
y = data["plus_minus_points"]

# Categorical and numerical columns
categorical_cols = ["tier", "game_type"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
print("Training model on dataset with", len(data), "rows...")
model.fit(X_train, y_train)

# Evaluate
print("\nModel Evaluation:")
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # <-- FIXED

# Save model
joblib.dump(model, "bgmi_plus_minus_predictor.pkl")
print("\nModel saved as bgmi_plus_minus_predictor.pkl")

# Prediction using user input
print("\nEnter your game stats for prediction:")
kills = int(input("Kills: "))
damage_dealt = float(input("Damage Dealt: "))
placement = int(input("Placement: "))
healing_used = int(input("Healing Used: "))
revives_given = int(input("Revives Given: "))
headshots = int(input("Headshots: "))
assists = int(input("Assists: "))
tier = input("Tier (Bronze/Silver/Gold/Platinum/Diamond/Crown/Ace/Conqueror): ")
game_type = input("Game Type (Solo/Duo/Squad): ")

# Prepare input for prediction
user_df = pd.DataFrame([{
    "kills": kills,
    "damage_dealt": damage_dealt,
    "placement": placement,
    "healing_used": healing_used,
    "revives_given": revives_given,
    "headshots": headshots,
    "assists": assists,
    "tier": tier,
    "game_type": game_type
}])

# Predict and display result
predicted_points = model.predict(user_df)[0]
print(f"\nPredicted Plus/Minus Points: {predicted_points:.2f}")
