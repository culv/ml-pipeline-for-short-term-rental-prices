#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the result in Weights and Biases
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    logger.info(f"Dataframe size before deduping: {df.shape}")
    start_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dataframe size after deduping: {df.shape}")
    logger.info(f"Dropped {len(df) - start_len} rows")

    # # A minimal feature engineering step: a new feature
    # logger.info("Feature engineering")
    # df['title'].fillna(value='', inplace=True)
    # df['song_name'].fillna(value='', inplace=True)
    # df['text_feature'] = df['title'] + ' ' + df['song_name']

    # filename = "processed_data.csv"
    # df.to_csv(filename)

    # artifact = wandb.Artifact(
    #     name=args.artifact_name,
    #     type=args.artifact_type,
    #     description=args.artifact_description,
    # )
    # artifact.add_file(filename)

    # logger.info("Logging artifact")
    # run.log_artifact(artifact)

    # os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean the raw data")


    parser.add_argument(
        "--input-artifact", 
        type=str,
        help="Fully qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--artifact-name", 
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--artifact-type", 
        type=str,
        help="Type of the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--artifact-description", 
        type=str,
        help="Description of the W&B artifact that will be created",
        required=True
    )


    args, unknown_args = parser.parse_known_args()
    logger.info(f"Args: {args}")
    logger.info(f"Unknown args: {unknown_args}")

    go(args)
