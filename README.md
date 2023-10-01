# ML Pipeline for short-term rental prices in NYC
This repo contains an MLflow pipeline for training a random forest regression model to predict the
prices of short-term Airbnb rentals in New York City. This pipeline makes it easy to retrain
the model on new data or experiment with hyperparameters to find better performing models

The model can be easily retrained on new data by overriding the Hydra config using this command:
```bash
> mlflow run . -P hydra_options='<new data>'
```

## Pipeline Overview
The pipeline currently has the following components:
1. Download data
    * This step downloads the data and uploads it to Weights and Biases to be used later
2. Clean data
    * This step performs basic data cleaning like removing duplicate rows, price outliers, and
    properties located outside of NYC. Cleaned data is uploaded to Weights and Biases
3. Check data
    * This step performs several data tests on cleaned data (such as if prices are within a
    given range, if all properties are located in NYC, etc.)
4. Split data
    * This step splits cleaned data into a test dataset, and a training/validation dataset. All
    datasets are uploaded to Weights and Biases
5. Train model
    * This step trains a random forest regression model and exports the trained sklearn pipeline
    along with other info such as R^2, MAE, and feature importance to Weights and Biases
6. Test model
    * This step will test a regression model (NOTE: must be tagged as ``prod`` in Weights and Biases).
    This step also will not be run automatically and must be manually run using this command:
    ```bash
    > mlflow run -P steps='test_regression_model'
    ```


## Weights and Biases
This pipeline saves all models, datasets, and metrics to Weights and Biases. All experiments done so
far can be found [here in this public W&B project](https://wandb.ai/culv/nyc_airbnb/overview)


## Project Setup
### Create environment
You can set up a conda environment based on ``environment.yml` file with all the necessary packages
by running the following:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
All datasets, models, and metrics are logged to Weights and Biases in this pipeline, so you'll
need to make sure you are logged in. You can get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and clicking on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### EDA (Exploratory Data Analysis)
Before running the pipeline to train a model, you may wish to examine the data. A Jupyter notebook
can be found at [src/eda/EDA.ipynb](src/eda/EDA.ipynb) that explores the data using ydata_profiling 
(formerly known as pandas_profiling)


### Hydra and the Configuration
The parameters of the model and other aspects of the pipeline are defined in ``config.yaml``. These
can either be updated manually by editing the file, or by overriding values using Hydra. For example,
to save runs under a different experiment name than ``development`` in Weights and Biases, you could
override config["main"]["experiment_name"] like so:
```bash
> mlflow run . -P hydra_options='main.experiment_name=my_experiment'
```

Hydra is also useful for doing hyperparameter sweeps by varying the model parameters. For example:
```bash
> mlflow run . -P hydra_options='modeling.max_tf_idf_features=5,10,25 modeling.random_forest.max_features=0.25,0.5,0.75,1 -m'
```

NOTE: It's highly recommended to update ``config.yaml`` with the best hyperparameters you've found
so that the default is always provides the best model

[You can learn more about Hydra here](https://hydra.cc/)


### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```
