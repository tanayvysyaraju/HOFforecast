Here's the updated `README.md` with all emojis removed and a sample CLI interaction using your provided QB stats:

---

```markdown
# HOFforecast: Predicting NFL Hall of Fame Induction

**HOFforecast** is a machine learning-powered analytics platform designed to forecast the likelihood of NFL players being inducted into the Pro Football Hall of Fame. By leveraging historical performance metrics, position-specific modeling, SMOTE oversampling, and user-interactive prediction inputs, HOFforecast aims to provide insightful, data-driven predictions for players across all major positions.

## Project Highlights

- Position-specific model training using Random Forests
- SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance for rare Hall of Fame classes
- Dynamic feature selection based on player's position
- Interactive CLI for user-submitted player predictions
- PostgreSQL integration for full data persistence and reproducibility
- Cross-validation and test set evaluation for model robustness

## Project Structure

```

HOFforecast/
│
├── data/                       # Raw CSV data files
│   ├── Career\_Stats\_Defensive.csv
│   ├── Career\_Stats\_Passing.csv
│   ├── Career\_Stats\_Receiving.csv
│   ├── Career\_Stats\_Rushing.csv
│   ├── nflhofplayers.csv
│   ├── ML\_HOF.csv              # Generated HOF player data
│   └── ML\_nonHOF.csv           # Generated non-HOF player data
│
├── main.py                     # Full ML pipeline with model training, evaluation, and user input
├── filter\_data.py              # Cleans, aggregates, and saves filtered HOF and non-HOF data to PostgreSQL
├── README.md                   # Project overview and documentation
├── requirements.txt            # Dependencies
└── ...

```

## Features & ML Design

### Feature Engineering

- Features are carefully selected per position using a domain-specific `position_feature_map`.
- Derived stats like Yards per Game, TDs per Attempt, and Int Rate enrich quarterback and receiver metrics.

### Modeling Per Position

Each position (e.g., QB, RB, WR, LB, DE, DB, etc.) is modeled separately:
- Uses `RandomForestClassifier` for robustness and feature importance extraction
- Automatically reduces features for underrepresented HOF positions (≤ 5 HOF samples)
- Applies SMOTE when minority HOF samples are insufficient for reliable prediction

### Evaluation

- Cross-validation with F1 score for class balance sensitivity
- Confusion matrix and classification report per position
- Accuracy is evaluated using:
  - Internal test split
  - Randomly selected "unseen" test rows
  - Optional live user input

## User Prediction CLI

After training, users are prompted to input custom players for prediction:

```



````


## Setup & Usage

### Prerequisites

- Python 3.8+
- PostgreSQL (local or remote)
- `virtualenv` recommended

### Installation

```bash
git clone https://github.com/yourusername/HOFforecast.git
cd HOFforecast
pip install -r requirements.txt
````

### PostgreSQL Setup

Ensure PostgreSQL is running locally:

```bash
# Create DB (if not already)
createdb full_db
```

Update credentials in `main.py` and `filter_data.py`:

```python
db_user = "your_postgres_user"
db_host = "localhost"
db_port = "5432"
db_name = "full_db"
```

### Run the Data Pipeline

```bash
python filter_data.py
```

This loads and filters the raw CSV data, then saves HOF and non-HOF players to PostgreSQL.

### Train & Predict

```bash
python main.py
```

This performs:

* Model training for all positions
* Cross-validation and feature importance output
* Interactive prediction mode

## Example Use Case

* Predict if a quarterback with 10,000 pass attempts, 150,000 yards, and 500 touchdowns has a Hall of Fame probability over 85%
* Analyze what features (e.g., interceptions, TD rate, sacks) matter most by position
* Use it to benchmark active NFL players for HOF trajectory

## Technologies Used

* Python 3.11
* pandas / numpy – Data processing
* scikit-learn – ML modeling
* imblearn – SMOTE oversampling
* SQLAlchemy – PostgreSQL ORM
* psycopg2 – PostgreSQL driver

## Sample Output

```
=== Training for Position: WR ===
Applied SMOTE with k_neighbors=4
Performing 5-fold cross-validation...
F1 Cross-Validation Scores: [0.78 0.80 0.76 0.79 0.82]
Average F1 Score: 0.7900

Feature Importances:
  Receptions: 0.42
  Receiving Yards: 0.38
  Receiving TDs: 0.20

Enter the following stats for a QB:
Passes Attempted: 10000
Passes Completed: 9500
Passing Yards: 150000
TD Passes: 500
Ints: 150

Predicted Probability of Hall of Fame: 87.00%
```

