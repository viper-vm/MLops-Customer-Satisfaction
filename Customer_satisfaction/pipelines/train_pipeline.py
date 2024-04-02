from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)  # change cached version to true to implement faster as it uses data from previous cached version
def train_pipeline(data_path: str):
    """
    Args:
        ingest_df: DataClass
        clean_df: DataClass
        train_model: DataClass
        evaluate_model: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, r2_score = evaluate_model(model, x_test, y_test)

