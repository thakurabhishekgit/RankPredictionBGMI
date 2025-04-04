import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("bgmi_ranking_final_with_crown_and_type.csv")

# Features & target
X = data.drop("plus_minus_points", axis=1)
y = data["plus_minus_points"]

# Categorical & numerical columns
categorical_cols = ["tier", "game_type"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# Full ML pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
print("Training model on dataset with", len(data), "rows...")
model.fit(X_train, y_train)

# Evaluate model
print("\nModel Evaluation:")
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  

# Save model
joblib.dump(model, "bgmi_plus_minus_predictor.pkl")
print("\nModel saved as bgmi_plus_minus_predictor.pkl")


### ========================== CLI Prediction ========================== ###
def predict_from_cli():
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

    # Prepare input
    user_df = pd.DataFrame([{
        "kills": kills, "damage_dealt": damage_dealt, "placement": placement,
        "healing_used": healing_used, "revives_given": revives_given,
        "headshots": headshots, "assists": assists,
        "tier": tier, "game_type": game_type
    }])

    # Predict & display
    predicted_points = model.predict(user_df)[0]
    print(f"\nPredicted Plus/Minus Points: {predicted_points:.2f}")


### ========================== Streamlit UI ========================== ###
def run_streamlit_app():
    st.set_page_config(page_title="BGMI Plus/Minus Predictor", page_icon="🎮", layout="centered")

    st.title("🎯 BGMI Plus/Minus Predictor")
    st.markdown("Enter your game stats to predict your **plus/minus points**!")

    # Input fields
    kills = st.number_input("💀 Kills", min_value=0, max_value=50, value=5)
    damage_dealt = st.number_input("🔥 Damage Dealt", min_value=0, value=500)
    placement = st.number_input("🏆 Placement", min_value=1, max_value=100, value=10)
    healing_used = st.number_input("🩹 Healing Used", min_value=0, value=3)
    revives_given = st.number_input("⚕️ Revives Given", min_value=0, value=1)
    headshots = st.number_input("🎯 Headshots", min_value=0, value=2)
    assists = st.number_input("🤝 Assists", min_value=0, value=2)

    # Dropdowns for categorical inputs
    tier = st.selectbox("🏅 Tier", ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Crown", "Ace", "Conqueror"])
    game_type = st.selectbox("🎮 Game Type", ["Solo", "Duo", "Squad"])

    # Prediction button
    if st.button("🔮 Predict Score"):
        # Load trained model
        model = joblib.load("bgmi_plus_minus_predictor.pkl")

        # Prepare input for prediction
        user_data = pd.DataFrame([{
            "kills": kills, "damage_dealt": damage_dealt, "placement": placement,
            "healing_used": healing_used, "revives_given": revives_given,
            "headshots": headshots, "assists": assists,
            "tier": tier, "game_type": game_type
        }])

        # Make prediction
        predicted_points = model.predict(user_data)[0]

        # Format gaming-style output
        if predicted_points > 0:
            st.success(f"🎉 **+{predicted_points:.2f} 🔥** (Great Game!)")
        else:
            st.error(f"❄️ **{predicted_points:.2f} ❄️** (Better Luck Next Time!)")


### ========================== Run as CLI or Web App ========================== ###
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        predict_from_cli()
    else:
        run_streamlit_app()
