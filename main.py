import json
from pathlib import Path

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

PARENT_DIR = Path(__file__).parent

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_path=str(PARENT_DIR), config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Get original directory script was called from
    cwd = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # Clean the raw data
            _ = mlflow.run(
                str(Path(cwd) / "src" / "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": config["data_cleaning"]["raw_data"],
                    "output_artifact_name": config["data_cleaning"]["cleaned_data"],
                    "output_artifact_type": "cleaned_data",
                    "output_artifact_description": "Cleaned data",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            # Run data tests on the cleaned data
            _ = mlflow.run(
                str(Path(cwd) / "src" / "data_check"),
                "main",
                parameters={
                    "csv": config["data_check"]["sample_data"],
                    "ref": config["data_check"]["reference_data"],
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                    "min_rows": config["data_check"]["min_rows"],
                    "max_rows": config["data_check"]["max_rows"]
                },
            )

        if "data_split" in active_steps:
            # Split data into a test dataset and a train/validation dataset
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version='main',
                parameters={
                    "input": config["data_split"]["input_data"],
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            # Train a random forest regression model
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
                str(Path(cwd) / "src" / "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": config["modeling"]["train_val_data"],
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["modeling"]["output_artifact"]
                },
            )

        if "test_regression_model" in active_steps:
            # Evaluate the trained model on test data
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version='main',
                parameters={
                    "mlflow_model": config["test_model"]["model"],
                    "test_dataset": config["test_model"]["data"]
                },
            )


if __name__ == "__main__":
    go()
