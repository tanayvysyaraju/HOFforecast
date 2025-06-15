import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine

# --- PostgreSQL setup (no password, random host) ---
db_user = "tanayvysyaraju"
db_host = "localhost"  # replace with actual host if needed
db_port = "5432"
db_name = "full_db"
engine = create_engine(f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}")

# --- STEP 1: Load and clean HOF + non-HOF datasets ---
try:
    non_hof_df = pd.read_sql("SELECT * FROM ml_nonhof", engine)
    hof_df = pd.read_sql("SELECT * FROM ml_hof", engine)
except:
    non_hof_df = pd.read_csv("data/ML_nonHOF.csv")
   
    hof_df = pd.read_csv("data/ML_HOF.csv")
    non_hof_df.to_sql("ml_nonhof", engine, if_exists="replace", index=False)
    hof_df.to_sql("ml_hof", engine, if_exists="replace", index=False)

rename_map = {
    "player": "Name", "position": "Position",
    "tackles": "Total Tackles", "sacks": "Sacks", "interception": "Ints",
    "receptions": "Receptions", "receiving_yards": "Receiving Yards", "receiving_td": "Receiving TDs",
    "rush_yards": "Rushing Yards", "rush_td": "Rushing TDs",
    "pass_yards": "Passing Yards", "pass_td": "TD Passes", "int_thrown": "Ints",
    "pass_attempts": "Passes Attempted", "completions": "Passes Completed"
}
hof_df.rename(columns=rename_map, inplace=True)
hof_df["HOF"] = 1
non_hof_df["HOF"] = 0

all_columns = set(hof_df.columns).union(non_hof_df.columns)
for col in all_columns:
    if col not in hof_df:
        hof_df[col] = pd.NA
    if col not in non_hof_df:
        non_hof_df[col] = pd.NA

hof_df = hof_df.loc[:, ~hof_df.columns.duplicated()]
non_hof_df = non_hof_df.loc[:, ~non_hof_df.columns.duplicated()]

unique_columns = []
seen = set()
for col in list(hof_df.columns) + list(non_hof_df.columns):
    if col not in seen:
        unique_columns.append(col)
        seen.add(col)

hof_aligned = hof_df.reindex(columns=unique_columns)
non_hof_aligned = non_hof_df.reindex(columns=unique_columns)
combined_df = pd.concat([hof_aligned, non_hof_aligned], ignore_index=True)

# Save to SQL
combined_df.to_sql("hof_combined", engine, if_exists="replace", index=False)

# --- STEP 2: Define position-specific relevant features ---
position_feature_map = {
    "QB": ["Passes Attempted", "Passes Completed", "Passing Yards", "TD Passes", "Ints"],
    "WR": ["Receptions", "Receiving Yards", "Receiving TDs"],
    "TE": ["Receptions", "Receiving Yards", "Receiving TDs"],
    "RB": ["Rushing Yards", "Rushing TDs", "Receptions", "Receiving Yards"],
    "LB": ["Total Tackles", "Sacks", "Ints"],
    "DE": ["Sacks", "Total Tackles"],
    "DT": ["Sacks", "Total Tackles"],
    "DB": ["Ints", "Total Tackles", "Passes Defended"],
    "SS": ["Ints", "Total Tackles", "Passes Defended"],
    "FS": ["Ints", "Total Tackles", "Passes Defended"],
}

# --- STEP 3: Train and evaluate models by position ---
positions = combined_df['Position'].dropna().unique()
all_y_true = []
all_y_pred = []

for position in positions:
    print(f"\n=== Training for Position: {position} ===")
    features = position_feature_map.get(position)
    if not features:
        print(f"Skipping {position} — no relevant features defined.")
        continue

    pos_df = combined_df[combined_df['Position'] == position]
    if pos_df['HOF'].nunique() < 2 or len(pos_df) < 10:
        print(f"Skipping {position} — insufficient class diversity or samples.")
        continue

    X = pos_df[features].fillna(0)
    y = pos_df["HOF"]

    hof_count = sum(y == 1)
    if hof_count <= 5:
        print(f"Reducing features and applying SMOTE for {position} (only {hof_count} HOF samples)...")
        k = min(2, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected = [features[i] for i in selector.get_support(indices=True)]
        print(f"Selected Features: {selected}")
        X = X[selected]
    else:
        selected = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    if hof_count <= 5:
        minority_class_count = y_train.value_counts().get(1, 0)
        if minority_class_count > 1:
            k_neighbors = min(5, minority_class_count - 1)
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"Applied SMOTE with k_neighbors={k_neighbors}")
        else:
            print("Skipping SMOTE — not enough HOF samples to resample.")

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"F1 Cross-Validation Scores: {cv_scores}")
    print(f"Average F1 Score: {cv_scores.mean():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, 'feature_importances_') and len(selected) == len(model.feature_importances_):
        print("Feature Importances:")
        for feat, importance in zip(selected, model.feature_importances_):
            print(f"  {feat}: {importance:.4f}")

# --- Final Overall Accuracy ---
print("\n=== OVERALL MODEL ACCURACY ACROSS ALL POSITIONS ===")
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

import numpy as np

n_runs = 20
total_correct = 0
total_samples = 0

for run in range(n_runs):
    for position in positions:
        features = position_feature_map.get(position)
        if not features:
            continue

        pos_df = combined_df[combined_df["Position"] == position].copy()
        if pos_df["HOF"].nunique() < 2 or len(pos_df) < 20:
            continue

        test_sample = pos_df.sample(n=min(5, len(pos_df) // 4), random_state=np.random.randint(0, 10000))
        train_df = pos_df.drop(index=test_sample.index)

        X_train = train_df[features].fillna(0)
        y_train = train_df["HOF"]
        X_test = test_sample[features].fillna(0)
        y_test = test_sample["HOF"]

        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        correct = (y_pred == y_test).sum()
        total_correct += correct
        total_samples += len(y_test)

        for i, row in test_sample.iterrows():
            name = row.get("Name", f"Player {i}")
            actual = row["HOF"]
            predicted = y_pred[X_test.index.get_loc(i)]

# --- Overall Accuracy Across All Runs ---
overall_fresh_accuracy = total_correct / total_samples if total_samples > 0 else 0
print("\n=== OVERALL FRESH DATA ACCURACY (across runs) ===")
print(f"Total Samples: {total_samples}")
print(f"Correct Predictions: {total_correct}")
print(f"Fresh Accuracy: {overall_fresh_accuracy:.4f}")

from sklearn.preprocessing import StandardScaler

print("\n=== INTERACTIVE PLAYER EVALUATION ===")
while True:
    user_input = input("Do you want to evaluate a player? (Y/N): ").strip().upper()
    if user_input != "Y":
        print("Exiting interactive evaluation.")
        break

    # Position selection
    print("\nAvailable Positions:")
    for idx, pos in enumerate(position_feature_map.keys(), start=1):
        print(f"{idx}. {pos}")
    pos_choice = input("Enter the number for the player's position: ").strip()

    try:
        pos_index = int(pos_choice) - 1
        position = list(position_feature_map.keys())[pos_index]
    except (IndexError, ValueError):
        print("Invalid position selection. Try again.")
        continue

    features = position_feature_map[position]
    print(f"\nEnter the following stats for a {position}:")
    input_data = {}
    for feat in features:
        while True:
            try:
                value = float(input(f"{feat}: "))
                input_data[feat] = value
                break
            except ValueError:
                print("Please enter a numeric value.")

    # Convert to DataFrame
    user_df = pd.DataFrame([input_data])

    # Train model using all available position data
    pos_df = combined_df[combined_df["Position"] == position]
    X = pos_df[features].fillna(0)
    y = pos_df["HOF"]

    if y.nunique() < 2:
        print(f"Not enough data to evaluate players at position {position}.")
        continue

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X, y)

    # Predict probability
    prob = model.predict_proba(user_df)[0][1]  # Probability of HOF
    print(f"\nPredicted Probability of Hall of Fame: {prob * 100:.2f}%\n")
