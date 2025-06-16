# HOFforecast

**HOFforecast** is a machine learning-driven system for predicting whether an NFL player has a strong likelihood of being inducted into the Hall of Fame. It leverages historical data, position-specific metrics, feature engineering, and real-time interactive prediction to enable analysts and fans to evaluate career excellence.

## Overview

This project was developed to:
- Extract and clean raw NFL career statistics from multiple CSV sources
- Normalize, combine, and classify players as HOF or non-HOF
- Train position-specific ML models to maximize predictive accuracy
- Allow interactive user evaluation of player profiles via command-line
- Integrate PostgreSQL for scalable data handling

**Final model accuracy on unseen player samples: 93.00%**

---

## Features

- Feature selection using `SelectKBest` for low-sample positions
- Class balancing using `SMOTE` for undersampled Hall of Famers
- Per-position model training (e.g., QB, WR, LB, DB, TE, etc.)
- PostgreSQL integration for scalable data persistence
- Cross-validation + fresh accuracy testing
- Interactive mode for user-input predictions

---

## Technologies Used

- **Python 3.11**
- **Pandas & NumPy** – Data loading, aggregation, feature engineering
- **scikit-learn** – Model training (`RandomForestClassifier`), evaluation, cross-validation, feature selection
- **imbalanced-learn (SMOTE)** – Oversampling rare HOF samples
- **SQLAlchemy** – PostgreSQL ORM integration
- **PostgreSQL** – Relational storage for normalized career data
- **Custom CSV preprocessing** – From public NFL stat datasets

---

## Project Structure

```bash
HOFforecast/
├── data/                       # Raw career stats CSVs
│   ├── Career_Stats_Defensive.csv
│   ├── Career_Stats_Passing.csv
│   ├── Career_Stats_Receiving.csv
│   └── Career_Stats_Rushing.csv
│   └── ML_HOF.csv           # cleaned and categorized hall of fame data
│   └── ML_nonHOF.csv        # cleaned and categorized non hall of fame data
├── data_extraction.py         # Cleans & normalizes data, saves to PostgreSQL
├── main.py                    # Trains models, evaluates accuracy, supports user input
├── requirements.txt
├── README.md
└── .gitignore
````

---

## Data Pipeline

### 1. Data Extraction & Cleaning (`data_extraction.py`)

* Raw CSVs processed by position type: QB, WR, RB, DB, LB, DE, DT, TE
* Grouped by player for career-level stats
* Derived metrics created (e.g., Yards/Game, Completion %, TD%, etc.)
* HOF player names are excluded from non-HOF group
* Final datasets stored in **PostgreSQL** tables: `ml_hof_raw` and `ml_nonhof_raw`

### 2. Model Training & Evaluation (`main.py`)

* Loads HOF and non-HOF data from database or CSV fallback
* Trains a separate model per position using only relevant features
* Uses `SelectKBest` if few positive samples exist
* Applies `SMOTE` to oversample rare HOF cases when needed
* Performs 5-fold cross-validation (F1-score), then trains/test splits
* Runs fresh data validation by randomly sampling new unseen rows
* Offers live user predictions based on stat entry

---

## Sample Prediction

```plaintext
=== INTERACTIVE PLAYER EVALUATION ===
Do you want to evaluate a player? (Y/N): Y

Available Positions:
1. QB
2. WR
3. TE
4. RB
5. LB
6. DE
7. DT
8. DB
9. SS
10. FS

Enter the number for the player's position: 1

Enter the following stats for a QB:
Passes Attempted: 10000
Passes Completed: 9500
Passing Yards: 150000
TD Passes: 500
Ints: 150

Predicted Probability of Hall of Fame: 87.00%
```

---

## Installation

```bash
git clone https://github.com/<your-username>/HOFforecast.git
cd HOFforecast

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Make sure you have PostgreSQL installed and running on your local machine (default port 5432 or 5431). You can change the DB connection string in `main.py` and `data_extraction.py`.

---

## Running the Project

```bash
# Step 1: Extract and store data into PostgreSQL
python data_extraction.py

# Step 2: Train models and enable user evaluation
python main.py
```

---

## Requirements

All Python dependencies are in `requirements.txt`:

```
pandas
numpy
scikit-learn
imblearn
sqlalchemy
psycopg2-binary
```

---

## Author

**Tanay Vysyaraju**
Project: HOFforecast
Domain: Sports Analytics, Machine Learning

