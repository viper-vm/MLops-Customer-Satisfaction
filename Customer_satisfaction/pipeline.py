from pipelines.train_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # run the pipeline
    print("Tracking URI:",Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/vivek/Desktop/codes/MLops/data/olist_customers_dataset.csv")

