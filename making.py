# Re-execute refined scoring logic with previously defined data and settings

# Recreate tier and game type mappings
tier_penalty_updated = {
    "Bronze": 0,
    "Silver": -2,
    "Gold": -4,
    "Platinum": -6,
    "Diamond": -8,
    "Crown": -10,
    "Ace": -12,
    "Conqueror": -15
}

game_type_multiplier = {
    "Solo": 1.0,
    "Duo": 1.1,
    "Squad": 1.2
}

# Refined scoring function
def refined_scoring(row):
    base_points = 0

    # Placement logic (only high reward for #1)
    if row["placement"] == 1:
        base_points += 25
    elif row["placement"] <= 5:
        base_points += 10
    elif row["placement"] <= 10:
        base_points += 5
    elif row["placement"] <= 50:
        base_points -= 5
    else:
        base_points -= 20

    # Performance contributions
    base_points += row["kills"] * 1.5
    base_points += row["headshots"] * 1.0
    base_points += row["assists"] * 0.5

    # Bonus for #1 with 10+ kills
    if row["placement"] == 1 and row["kills"] >= 10:
        base_points += 15

    # Tier penalty
    base_points += tier_penalty_updated[row["tier"]]

    # Game type multiplier
    base_points *= game_type_multiplier[row["game_type"]]

    return max(-30, min(30, round(base_points)))

# Apply refined scoring
df_final["plus_minus_points"] = df_final.apply(refined_scoring, axis=1)

# Save final refined dataset
refined_dataset_path = "/mnt/data/bgmi_ranking_final_refined.csv"
df_final.to_csv(refined_dataset_path, index=False)

refined_dataset_path
