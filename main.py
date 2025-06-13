from data_extraction import filterHOF, filterNonHOF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Combine the data
hof_df, non_hof_df = filterHOF(), filterNonHOF()
hof_df['HOF'] = 1
non_hof_df['HOF'] = 0
df = pd.concat([hof_df, non_hof_df], ignore_index=True)

# List of positions to model
positions = df['Position'].unique()

# Loop through each position and train a model
for position in positions:
    if pd.isna(position):
        continue

    pos_df = df[df['Position'] == position]
    if len(pos_df) < 2:  # or whatever threshold makes sense
        print(f"Skipping {position} — not enough data.")
        continue
    print(f"\n=== Training for Position: {position} ===")
    
    # Filter data for the current position
    pos_df = df[df['Position'] == position]
    
    # Define features and target
    X = pos_df.drop(columns=['HOF', 'Name', 'Player Id', 'Position'], errors='ignore')
    y = pos_df['HOF']

    # Drop columns with all NaNs or low variance
    X = X.dropna(axis=1, how='all').fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))



