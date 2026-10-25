from pathlib import Path
from src.features import process_application_data

DATA_DIR = Path("home-credit-default-risk")
train_path = DATA_DIR / "application_train.csv"

# Run the lazy feature engineering function
print("Processing application data lazily...")
lazy_df = process_application_data(train_path)

# Preview the first few rows
print(lazy_df.select(["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_ANNUITY_RATIO"]).head(5).collect())
print("\nSetup verified successfully! You're ready to build out the rest of your pipeline.")