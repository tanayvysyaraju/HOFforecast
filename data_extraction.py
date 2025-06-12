import pandas as pd
from sqlalchemy import create_engine

# Load and process your data
df = pd.read_csv("data/nflhofplayers.csv")

# Filter positions and rename
valid_positions = ['DB', 'LB', 'QB', 'DE', 'RB', 'WR', 'DT', 'HB', 'E', 'FB', 'TB', 'TE']
df = df[df["position"].isin(valid_positions)]
df["position"] = df["position"].replace({'HB': 'RB', 'FB': 'RB', 'TB': 'RB'})

# Filter by 'to' year
df = df[df["to"] >= 1985]

df.to_csv("data/HOFfiltered_players.csv", index=False)
