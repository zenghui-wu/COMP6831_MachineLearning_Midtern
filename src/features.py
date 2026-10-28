import polars as pl

def process_application_data(file_path: str) -> pl.LazyFrame:
    """Loads and cleans application_train lazily."""
    return (
        pl.scan_csv(file_path)
        .with_columns([
            # Feature engineering steps here
            (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY")).alias("CREDIT_ANNUITY_RATIO")
        ])
    )