import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# PostgreSQL connection setup
db_user = "tanayvysyaraju"
db_host = "localhost"
db_port = "5431"
db_name = "intial_db"
engine = create_engine(f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}")

def filterHOF():
    df = pd.read_csv("data/nflhofplayers.csv")

    valid_positions = ['DB', 'LB', 'QB', 'DE', 'RB', 'WR', 'DT', 'HB', 'E', 'FB', 'TB', 'TE']
    df = df[df["position"].isin(valid_positions)]
    df["position"] = df["position"].replace({'HB': 'RB', 'FB': 'RB', 'TB': 'RB'})
    df = df[df["to"] >= 1985]

    df.to_sql("ml_hof_raw", engine, if_exists="replace", index=False)
    return df

def filterNonHOF():
    # DEFENSIVE
    df = pd.read_csv("data/Career_Stats_Defensive.csv")
    df.replace('--', pd.NA, inplace=True)
    numeric_cols = [
        'Games Played', 'Total Tackles', 'Solo Tackles', 'Assisted Tackles',
        'Sacks', 'Safties', 'Passes Defended', 'Ints', 'Ints for TDs',
        'Int Yards', 'Yards Per Int', 'Longest Int Return'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

    valid_positions = ['NT', 'SS', 'DB', 'LB', 'MLB', 'DE', 'FS', 'ILB', 'OLB', 'CB', 'DT']
    df = df[df["Position"].isin(valid_positions)]
    df["Position"] = df["Position"].replace({
        'SS': 'DB', 'CB': 'DB', 'FS': 'DB', 'MLB': 'LB',
        'ILB': 'LB', 'OLB': 'DE', 'NT': 'DT'
    })

    hof_df = pd.read_sql("SELECT * FROM ml_hof_raw", engine)
    hof_df["Formatted Name"] = hof_df["player"].apply(
        lambda x: f"{x.split()[1]}, {x.split()[0]}" if len(x.split()) >= 2 else x
    )
    hof_names = set(hof_df["Formatted Name"])
    df = df[~df["Name"].isin(hof_names)]

    db_df = df[(df["Position"] == "DB") & (df["Games Played"] >= 48)]
    lb_df = df[(df["Position"] == "LB") & (df["Games Played"] >= 20)]
    de_df = df[(df["Position"] == "DE") & (df["Games Played"] >= 64)]
    dt_df = df[(df["Position"] == "DT") & (df["Games Played"] >= 100)]

    # PASSING
    df = pd.read_csv("data/Career_Stats_Passing.csv")
    df.replace('--', pd.NA, inplace=True)
    numeric_cols = ['Games Played', 'Passes Attempted', 'Passes Completed', 'Passing Yards', 'TD Passes', 'Ints']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

    df['Completion Percentage'] = (df['Passes Completed'] / df['Passes Attempted']) * 100
    df['Pass Attempts Per Game'] = df['Passes Attempted'] / df['Games Played']
    df['Passing Yards Per Attempt'] = df['Passing Yards'] / df['Passes Attempted']
    df['Passing Yards Per Game'] = df['Passing Yards'] / df['Games Played']
    df['Percentage of TDs per Attempts'] = (df['TD Passes'] / df['Passes Attempted']) * 100
    df['Int Rate'] = (df['Ints'] / df['Passes Attempted']) * 100

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[~df["Name"].isin(hof_names)]
    df = df[df['Position'].isna() | (df['Position'] == 'QB')]
    df['Position'] = df['Position'].fillna('QB')
    qb_df = df[df["TD Passes"] >= 50]

    # RECEIVING
    df = pd.read_csv("data/Career_Stats_Receiving.csv")
    df.replace('--', pd.NA, inplace=True)
    numeric_cols = [
        'Games Played', 'Receptions', 'Receiving Yards', 'Receiving TDs',
        'Receptions Longer than 20 Yards', 'Receptions Longer than 40 Yards',
        'First Down Receptions', 'Fumbles'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()

    df['Yards Per Reception'] = df['Receiving Yards'] / df['Receptions']
    df['Yards Per Game'] = df['Receiving Yards'] / df['Games Played']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[~df["Name"].isin(hof_names)]
    df = df[df['Position'].isna() | df['Position'].isin(['TE', 'WR'])]
    df['Position'] = df['Position'].fillna('WR')

    wr_df = df[(df["Position"] == 'WR') & (df["Receiving Yards"] > 3250) & (df["Receiving TDs"] > 20)]
    te_df = df[(df["Position"] == 'TE') & (df["Receiving Yards"] > 1000) & (df["Receiving TDs"] > 5)]

    # RUSHING
    df = pd.read_csv("data/Career_Stats_Rushing.csv")
    df.replace('--', pd.NA, inplace=True)
    numeric_cols = [
        'Games Played', 'Rushing Attempts', 'Rushing Attempts Per Game', 'Rushing Yards',
        'Yards Per Carry', 'Rushing Yards Per Game', 'Rushing TDs',
        'Rushing First Downs', 'Percentage of Rushing First Downs',
        'Rushing More Than 20 Yards', 'Rushing More Than 40 Yards', 'Fumbles'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df[df['Position'].isna() | df['Position'].isin(['RB', 'FB', 'HB', 'TB'])]
    df['Position'] = 'RB'
    df = df.groupby(['Player Id', 'Name', 'Position'], dropna=False)[numeric_cols].sum().reset_index()
    df = df[~df["Name"].isin(hof_names)]
    rb_df = df[(df["Rushing Yards"] > 3500) | (df["Rushing TDs"] > 25)]

    nonHOF_df = pd.concat(
        [db_df, wr_df, rb_df, lb_df, de_df, qb_df, dt_df, te_df],
        ignore_index=True
    )
    nonHOF_df.to_sql("ml_nonhof_raw", engine, if_exists="replace", index=False)
    return nonHOF_df