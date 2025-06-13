import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def filterHOF():
  #reading csv
  df = pd.read_csv("data/nflhofplayers.csv")

  # filter positions, rename, and filter by year player's career ended
  valid_positions = ['DB', 'LB', 'QB', 'DE', 'RB', 'WR', 'DT', 'HB', 'E', 'FB', 'TB', 'TE']
  df = df[df["position"].isin(valid_positions)]
  df["position"] = df["position"].replace({'HB': 'RB', 'FB': 'RB', 'TB': 'RB'})
  df = df[df["to"] >= 1985]

  #saving to a csv file
  df.to_csv("HOF.csv", index=False)

#FILTERING DEFENSIVE DATA SET
#reading csv
df = pd.read_csv("data/Career_Stats_Defensive.csv")

#aggregating players seasons into career statistics and replacing empty entries
df.replace('--', pd.NA, inplace=True)
numeric_cols = [
    'Games Played', 'Total Tackles', 'Solo Tackles', 'Assisted Tackles',
    'Sacks', 'Safties', 'Passes Defended', 'Ints', 'Ints for TDs',
    'Int Yards', 'Yards Per Int', 'Longest Int Return'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

#filtering out non-defensive positions and grouping positions into HOF positions
valid_positions = ['NT', 'SS', 'DB', 'LB', 'MLB', 'DE', 'FS', 'ILB', 'OLB', 'CB', 'DT']
df = df[df["Position"].isin(valid_positions)]
df["Position"] = df["Position"].replace({
    'SS': 'DB', 
    'CB': 'DB', 
    'FS': 'DB', 
    'MLB': 'LB', 
    'ILB': 'LB', 
    'OLB': 'DE', 
    'NT': 'DT'
})

# Load HOF data and normalize name format
hof_df = pd.read_csv("data/HOF.csv")
hof_df["Formatted Name"] = hof_df["player"].apply(
    lambda x: f"{x.split()[1]}, {x.split()[0]}" if len(x.split()) >= 2 else x
)
hof_names = set(hof_df["Formatted Name"])

# Keep only players NOT in the HOF
df = df[~df["Name"].isin(hof_names)]

#build defensive back data frame
db_df = df[df["Position"] == "DB"]
db_df = db_df[db_df["Games Played"] >= 48]

#build linebacker data frame
lb_df = df[df["Position"] == "LB"]
lb_df = lb_df[lb_df["Games Played"] >= 20]

#build defensive end data frame
de_df = df[df["Position"] == "DE"]
de_df = de_df[de_df["Games Played"] >= 64]

#build defensive tackle data frame
dt_df = df[df["Position"] == "DE"]
dt_df = dt_df[dt_df["Games Played"] >= 100]


#FILTERING PASSING DATASET
df = pd.read_csv("data/Career_Stats_Passing.csv")

# Replace '--' with NaN and convert columns to numeric
df.replace('--', pd.NA, inplace=True)

numeric_cols = [
    'Games Played', 'Passes Attempted', 'Passes Completed', 'Passing Yards', 'TD Passes', 'Ints'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Group by player to get career totals
df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

# Derived stats
df['Completion Percentage'] = (df['Passes Completed'] / df['Passes Attempted']) * 100
df['Pass Attempts Per Game'] = df['Passes Attempted'] / df['Games Played']
df['Passing Yards Per Attempt'] = df['Passing Yards'] / df['Passes Attempted']
df['Passing Yards Per Game'] = df['Passing Yards'] / df['Games Played']
df['Percentage of TDs per Attempts'] = (df['TD Passes'] / df['Passes Attempted']) * 100
df['Int Rate'] = (df['Ints'] / df['Passes Attempted']) * 100

# Replace any infinite or NaN results from division by zero
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#filtering out names that are in the hall of fame
df = df[~df["Name"].isin(hof_names)]

# Keep only rows where Position is 'QB' or NaN
df = df[df['Position'].isna() | (df['Position'] == 'QB')]
# Replace NaN with 'QB'
df['Position'] = df['Position'].fillna('QB')

#filtering based on games played
qb_df = df[df["TD Passes"] >= 50]


#FILTERING RECEIVING DATASET
df = pd.read_csv("data/Career_Stats_Receiving.csv")

# Replace '--' with NaN
df.replace('--', pd.NA, inplace=True)

# Define numeric columns
numeric_cols = [
    'Games Played', 'Receptions', 'Receiving Yards',
    'Receiving TDs', 'Receptions Longer than 20 Yards',
    'Receptions Longer than 40 Yards', 'First Down Receptions', 'Fumbles'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Group by player to aggregate career totals
df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

# Derived metrics
df['Yards Per Reception'] = df['Receiving Yards'] / df['Receptions']
df['Yards Per Game'] = df['Receiving Yards'] / df['Games Played']

# Replace infinite/NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Filter out HOF players and keep certain players
df = df[~df["Name"].isin(hof_names)]
df = df[df['Position'].isna() | df['Position'].isin(['TE', 'WR'])]
df['Position'] = df['Position'].fillna('WR')

wr_df = df[df["Position"] == 'WR']
wr_df = wr_df[(wr_df["Receiving Yards"] > 3250) & (wr_df["Receiving TDs"] > 20)]

te_df = df[df["Position"] == 'TE']
te_df = te_df[(te_df["Receiving Yards"] > 1000) & (te_df["Receiving TDs"] > 5)]

df = pd.read_csv("data/Career_Stats_Rushing.csv")

# Replace '--' with NaN
df.replace('--', pd.NA, inplace=True)

# Define numeric columns to convert
numeric_cols = [
    'Games Played', 'Rushing Attempts', 'Rushing Attempts Per Game',
    'Rushing Yards', 'Yards Per Carry', 'Rushing Yards Per Game',
    'Rushing TDs', 'Rushing First Downs', 'Percentage of Rushing First Downs',
    'Rushing More Than 20 Yards', 'Rushing More Than 40 Yards', 'Fumbles'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Filter for running back-related positions
rb_positions = ['RB', 'FB', 'HB', 'TB']
df = df[df['Position'].isna() | df['Position'].isin(rb_positions)]

# Normalize position to just 'RB'
df['Position'] = 'RB'

# Group by player for career totals
df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

# Filter out HOF players
hof_df = pd.read_csv("data/HOF.csv")
hof_df["Formatted Name"] = hof_df["player"].apply(
    lambda x: f"{x.split()[1]}, {x.split()[0]}" if len(x.split()) >= 2 else x
)
hof_names = set(hof_df["Formatted Name"])
df = df[~df["Name"].isin(hof_names)]

# Optional: filter only productive backs
rb_df = df[(df["Rushing Yards"] > 3500) | (df["Rushing TDs"] > 25)]

nonHOF_df = pd.concat([db_df, wr_df, rb_df, lb_df, de_df, qb_df, dt_df, te_df], ignore_index=True)
nonHOF_df.to_csv("data/nonHOF.csv", index=False)
print(nonHOF_df)
