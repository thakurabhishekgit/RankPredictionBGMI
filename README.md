# BGMI Rank Prediction Using Machine Learning
## 📌 Introductionn
This project is an attempt to **understand and predict how ranking systems work in BGMI (Battlegrounds Mobile India) and other competitive games**. The curiosity behind this project came from the question:

> **"How does BGMI calculate ranking points, and what factors influence rank gain/loss?"**

Many competitive games have hidden ranking systems that depend on various gameplay metrics such as kills, damage dealt, placement, assists, and other factors. This project uses **Machine Learning (ML)** to analyze these metrics and predict the player's **Plus/Minus Ranking Points** after each match.
## 🎯 Objective
- **Analyze** the factors affecting rank gain/loss in BGMI.
- **Predict** how many ranking points a player will gain or lose after a match.
- **Provide insights** into what gameplay actions influence ranking the most.
- **Make predictions accessible** through a **Streamlit web app** for easy user interaction.

## 📊 Dataset

The dataset used for this project is **bgmi_ranking_final_with_crown_and_type.csv**, which contains player match statistics and their corresponding **Plus/Minus Ranking Points**. The key features include:

### 🔹 Features (Inputs)
1. `kills`: Number of enemy kills in the match
2. `damage_dealt`: Total damage dealt in the match
3. `placement`: Final placement position in the match
4. `healing_used`: Amount of healing items used
5. `revives_given`: Number of times teammates were revived
6. `headshots`: Number of headshot kills
7. `assists`: Number of assists in the match
8. `tier`: Player's current rank (Bronze, Silver, Gold, Platinum, Diamond, Crown, Ace, Conqueror)
9. `game_type`: Type of match (Solo, Duo, Squad)

### 🔸 Target Variable (Output)
- `plus_minus_points`: The number of ranking points a player gains or loses after the match.

## ⚙️ Model Details
The model is built using **XGBoost Regressor**, which is well-suited for structured tabular data and provides accurate predictions. The key steps in model training include:

1. **Data Preprocessing**: One-hot encoding for categorical features (`tier`, `game_type`), normalization of numerical features.
2. **Train-Test Split**: The dataset is split into **90% training data** and **10% test data**.
3. **Model Training**: The **XGBRegressor** is trained with optimized hyperparameters:
   - `n_estimators=300`
   - `learning_rate=0.05`
   - `max_depth=6`
4. **Evaluation Metrics**:
   - **R2 Score**: 0.99 (Indicates a highly accurate model)
   - **MAE (Mean Absolute Error)**: 1.06
   - **RMSE (Root Mean Squared Error)**: 1.54

## 🚀 Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- Pip
- Git
- Virtual Environment (Optional)

### Clone the Repository
```sh
git clone https://github.com/thakurabhishekgit/RankPredictionBGMI.git
cd RankPredictionBGMI
```
### Install Dependencies
```sh

pip install -r requirements.txt
```

### Train the Model
Run the following command to train the model and save it:
```sh
python bgmi_plus_predictor.py
```

### Run the Streamlit Web App
```sh
streamlit run app.py
```

## 🖥️ Usage Guide

1. Open the Streamlit web interface.
2. Enter your **game stats** (kills, damage, placement, etc.).
3. Click **Predict** to see how many ranking points you will gain or lose.
4. Get insights on how to improve your ranking!

## 📌 Future Enhancements
- Add more features like **survival time**, **accuracy**, **win rate**.
- Optimize the model for **real-time predictions**.
- Deploy the model as a **web API**.

---

## 📜 License
This project is open-source and available under the **MIT License**.

## 👨‍💻 Author
**[Thakur Abhishek](https://github.com/thakurabhishekgit)**

If you like this project, don't forget to ⭐ it on GitHub!

