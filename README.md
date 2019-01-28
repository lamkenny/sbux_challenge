# Starbucks Challenge 

## Project Definition

In this project, we'll be looking at a simulated dataset that reflect the behavior of customers on Starbucks rewards mobile application. The dataset contains coffee offers, which are basically advertisement for a drink or an actual discount or BOGO. Some users receive these offers and some don't. Offers are sent out multiple times. The dataset captures when a user receive an offer, when the offer is viewed and when the offer is completed, and when a purchase is made. 

Our task is to perform data exploration and analysis the see how offers influenced purchases by demographics.

Our final task is to build a recommender. Given a customer of a demographic group, the recommender returns an offer that it thinks would get the most response from that demographic group. The recommender takes the form of a web application that comprises of a frontend UI and backend ML component. The frontend UI allows users of the application to select a demographic group. The backend component responds to HTTP requests made by the frontend browser and makes offer recommendation with the ML model.

## Methodology

Our method is threefold. First is data exploration and analysis. Second is developing an ML model for recommendation. Third is employing the trained model on a web application.

We take the data exploration step to learn about the data. Missing values in the dataset are either inferred or removed. The process of cleaning up prepares us for the analysis where we aim to answer the question, which demographic groups are influenced by the offers they received?

In the second step, we define our ML problem and work to develop a model for offer recommendation. Starting with a benchmark model we can quickly define the starting point for training a better and more accurate model. 

The final step is to expose the model behind a web application. The web application can be used by an internal marketing team or an analyst to evaluate the offer they'd like to send out.


## System

The repository contains two notebooks, `sbux_challenge_data_analysis.ipynb` and `sbux_challenge_ml_recommender.ipynb`. They are for the data exploration and analysis step and the ML development step described above, respectively. In `sbux_challenge_ml_recommender.ipynb`, we defined a model and obtain the best hyperparameters the model.

The web application is a Flask application. It uses [Dash](https://plot.ly/products/dash/) to create the frontend HTML components. On the backend, it uses the trained model to make offer recommendation.

The system component can be illustrated as follow:
![Offer Recommender System Diagram](system_diagram.png?raw=true "Offer Recommender System Diagram")

## Future Improvement

The system we're building for this project is a simple one. This is small project and we're running a standalone web application, after all. 

The ML learning architecture can be improved to use stack of models instead of a single model. 
The web application can expose a REST endpoint so that it can be used programmatically by other web applications.


## File Description

`sbux_challenge_data_analysis.ipynb`

Python notebook where data exploration, data transformation/cleaning and analysis can be found. This notebook illustrate how we gain insight into our dataset.

`sbux_challenge_ml_recommender.ipynb`

Python notebook where ML models are developed. This documents how the final recommender ML model is derived, the metrics used for working towards obtaining the best model. 

`train.py`

Train the model using (X_train, y_train) data and evaluate based on (Y_train, y_test). The purpose is to train and __save__ the model for the offer recommender web application via `pickle`. Due to file size limitation on Github, we're not able to save and commit large `pickle` outputs. Use this script before running `serve`.

`serve.py`

Driver application for launching offer recommender web application. 

`recommenderapp\`

The files that makes up the web application are stored in this module. The web app is built using Dash and Flask server. The routes, HTML, and business logic for recommending the best offer can be found here.

`data\`

Directory containing the data files saved from the python notebooks and required for training and serving. It includes the simulated dataset, cleaned transaction data, and data for training/testing the ML model.

`models\`

Directory for storing saved models. This is where the web application looks for the saved model.


## Installation

[Python 3.6, Anaconda distribution](https://www.anaconda.com/download/) is recommended. 
The requirements file is generated for Anaconda environment. Packages that are required include Scikit-learn, NumPy and Pandas. 

Install the conda environment:
    
    conda env create -f conda_env.yaml

Activate the conda environment:

    conda activate starbucks_challenge_x193k

Before luanching the web application, train the model and save it:
    
    python train.py

Starting the recommender web application:

    python serve.py

On a browser, open: 

    http://localhost:3001
