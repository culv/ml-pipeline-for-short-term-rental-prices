#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the result in Weights and Biases
"""
import argparse
import logging
import os

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
    logger.info(f"Dropped {start_len - len(df)} duplicate rows")


    # Filter out price outliers
    logger.info("Removing price outliers")
    logger.info(f"Dataframe size before removing outliers: {df.shape}")
    start_len = len(df)

    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    logger.info(f"Dataframe size after removing price outliers: {df.shape}")
    logger.info(f"Dropped {start_len - len(df)} price outlier rows")


    # Correct the datatype of the last_reviewed column from str to datetime
    logger.info("Converting 'last_reviewed' from str to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])


    # Temporarily save csv file of cleaned data
    filename = "clean_data.csv"
    logger.info(f"Temporarily saving cleaned data locally to {filename}")
    df.to_csv(filename)


    # Create artifact and log to Weights and Biases
    logger.info("Setting up wandb artifact for cleaned data")
    artifact = wandb.Artifact(
        name=args.output_artifact_name,
        type=args.output_artifact_type,
        description=args.output_artifact_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    artifact.wait()

    logger.info(f"Cleaning up local files (deleting {filename})")
    os.remove(filename)
    logger.info(f"{filename} deleted")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean the raw data")


    parser.add_argument(
        "--input-artifact", 
        type=str,
        help="Fully qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output-artifact-name", 
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--output-artifact-type", 
        type=str,
        help="Type of the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--output-artifact-description", 
        type=str,
        help="Description of the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--min-price", 
        type=float,
        help="Minimum price for Airbnb listings, any smaller prices are removed as outliers",
        required=True
    )

    parser.add_argument(
        "--max-price", 
        type=float,
        help="Maximum price for Airbnb listings, anything larger prices are removed as outliers",
        required=True
    )


    args, unknown_args = parser.parse_known_args()
    logger.info(f"Args: {args}")
    logger.info(f"Unknown args: {unknown_args}")

    go(args)
