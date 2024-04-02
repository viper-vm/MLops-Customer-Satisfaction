import logging
import pandas as pd
from zenml import step
from model.evaluation import RMSE, R2Score, MSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, 
                   x_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluates the model on the ingested data
    Args:
        df: the ingested data
    """
    try:    
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse_score = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse",mse_score)

        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("R2",r2_score)

        rmse_class = RMSE()
        rmse_score = rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse",rmse_score)

        return r2_score, rmse_score
    except Exception as e:
        logging.error("Error in Evaluating model: {}".format(e))
        raise e
