# Starbucks Challenge 

## Project Overview

In this project, we'll be looking at a simulated dataset that reflect the behavior of customers on Starbucks rewards mobile application. The dataset contains coffee offers, which are basically advertisement for a drink (informational only) or an actual discount offer or BOGO offer. Some users receive these offers and some don't. Offers are sent out multiple times. The dataset captures events of when a user receives, views and complete an offer and when a user makes purchase. These events as a whole tell a story of how users respond to these offers. 

## Dataset

The dataset is divided into three files:

* portfolio.json - contains metadata about offers that get sent out to customers.

* profile.json - demographic data of every customer that includes became member date, age, income and gender.

* transcript.json - event logs of when offers are received, viewed, completed, and event logs of purchases.

## Problem Statement

We want to know how user responded to the offers they received. There are ten offers in the portfolio. Certain offers will definitely get a stronger response from the users than other offers. We want to look at the data and find out how each demographic group respond to the offers.

Our first task is to perform data exploration and analysis the see how offers influence purchases by demographics.

Our second task is to build a recommender. Given a customer of a demographic group, the recommender returns an offer that it thinks would get the most response from that demographic group. The recommender is an ML model trained from our dataset.

Our final task is to build a web application that takes the trained model and provide a UI layer that allows Starbucks marketing staff to interact with it. 

## Metrics

For the ML model, we want it to rank the top three offers and evaluate whether or not those top three offers match the actual top three offers. We defer to NDCG (normalized discounted cumulative gain) @k to generate a score between 0.0 and 1.0 indicating how relevant our top three offers are compared to the actual top three offers.

![](images/ndcg.png?raw=true "")

All normalized DCG (NDCG) values are on the interval 0.0 to 1.0, where higher is better.

More information about NDCG can be found at https://en.wikipedia.org/wiki/Discounted_cumulative_gain.

The number of outcomes for our dataset is small, and a user don't receive very many offers, and thus, we limit our evaluation to only the top 3 offers and thus consider k=3.

In other words, our models are graded based on whether or not they get the top 3 offers right. It doesn't matter what order the top 3 items are in as long as they are in the top 3.

## Data Preprocessing

Gender and income in the demographic data have missing values. For missing gender, we created a category for it and called it 'Unspecified (U)'. Missing income values are forward filled using adjacent values.

The transcript data contains offer metadata that were in a JSON encoded string. That was expanded into columns in the dataframe so that they can be joined with the portfolio data.
![](images/json_transformation.png?raw=true "")

We took many transformation steps to join the transcript data, portfio data and profile data into a single data frame for analysis. Each transaction is properly attributed with the received offer.
![](images/transaction_dataframe.png?raw=true "")

## Data Exploration

As documented in [sbux\_challenge\_data\_analysis.ipynb](sbux_challenge_data_analysis.ipynb), we learned the following information from the data exploration and analysis:
    - There are 6 campaign periods where offers get sent out to customers.
    - A customer can receive the same offers as much as 5 times.
    - A customer only receive 1 offer per campaign period.
    - There are 10 offers in the portfolio and in each campaign, all 10 offers get sent out.
    - There are three types of event for the offer: received, viewed, and completed.
    - An informational offer does not have a completed event and doesn't have any reward.
    - An offer can be completed multiple times in the same campaign. This is like completing many purchases or stacking multiple items to complete multiple bogo/discount offers.

After joining the offers to their corresponding transactions, we're able to determine whether or not each transaction was influenced and by which offer. We define influence as the transaction is viewed before a purchase is made. If the purchase is made after the offer is viewed during the offer's validity period, then customer is responding to the offer, and therefore, influenced.

For each gender group, we can see from the plots that there are more influenced transactions (orange) than not-influenced transactions (blue) consistently for each offer type. 
![](images/influence_by_gender.png?raw=true "")

For each age group, the number of influenced transactions is higher but proportion varies from group to group.
![](images/influence_by_age.png?raw=true "")

For each income group, the influenced transactions are most pronounced for those making approximately $60K.
![](images/influence_by_income.png?raw=true "")

When grouping the users by all demographic features (age, gender, age, income), the average response rate is __0.60__ for informational offer, __0.66__ for bogo and __0.65__ for discount.


## Method & Results

Let's summarize how we create the ML model and obtain our results. For more details, the ML development task is documented in [sbux\_challenge\_ml\_recommender.ipynb](sbux_challenge_ml_recommender.ipynb). 

Each demographic group receives multiple offers and each group responds to the offers they received differently. Our data contains many combination of demographic features (gender, age, income and became_member_on date). For each of these demographic group, we want to recommend the offer that gets the best response. This is an infeasible task to do with data analysis. So, we turn to machine learning.

What we need is the ability to rank the offers and pick the offer with the best rank. Using machine learning, we should be able assign a probability, for likely to get a response, to each offer. Offers with the highest probability gets returned as the recommended offers. ML classifiers such as `DecisionTreeClassifier` and `RandomForestClassifier` naturally do this with the `predict()` method and `predict_proba()` method. 
    - `predict()` returns the offer with the highest probability. 
    - `predict_proba()` returns all probabilities, one for each offer.

The ML model's task is to predict the offer_id. That is the target label (y). Each transaction has the user demographic features. These make up the input features (X). 

For users who do not need an offer but would purchase on the app anyway, we define 'no_offer' as the label. This is for the negative case. Altogether, there are 11 possible labels: ten offer_ids and 'no_offer'. 

We split our (X, y) data 80/20 for train and test, respectively.

We use a `DecisionTreeClassifier` as our benchmark model. This model has a nDCG@3 score 0.59.

Then, we train `RandomForestClassifier` for the recommender. This model gets an nDCG@3 score of 0.62, which is better than the `DecisionTreeClassifier`. 

Hyperparameters tuning for the `RandomForestClassifier` involves performing 2-fold RandomSearchCV so that we can get an idea of where the best hyperparameter values are in hyperparameters space. RandomSearchCV determines that the hyperparameter values are around:
    ```
        {'n_estimators': 1836,
         'min_samples_split': 8,
         'min_samples_leaf': 1,
         'max_features': 'auto',
         'max_depth': 20,
         'bootstrap': True}
    ```

Finally, we use 3-fold GridSearchCV to look for the best hyperparameters. The best values are:
    ```
        {'bootstrap': True,
         'max_depth': 25,
         'max_features': 'auto',
         'min_samples_leaf': 2,
         'min_samples_split': 10,
         'n_estimators': 2000}
    ```

The tuned model has an nDCG@3 score of __0.68__. 

The tuned model is then saved for Offer Recommender web application. The web application is a Flask application. It uses [Dash](https://plot.ly/products/dash/) to create the frontend HTML components. On the backend, it uses the trained model to make offer recommendation. The web application will be useful for a marketing team or an analyst when it comes time to evaluate which offer to send out.
![Offer Recommender Web Application](images/offer_rec_web_app.png?raw=true "Offer Recommender Web Application")

The system component can be illustrated as follow:
![Offer Recommender System Diagram](images/system_diagram.png?raw=true "Offer Recommender System Diagram")

## Conclusion

We started with a benchmark model, `DecisionTreeClassifier`, that has a nDCG@3 score of 0.59. 
Our first `RandomForestClassifier` has a nDCG@3 score of 0.62. 
Our tuned `RandomForestClassifier` has a DCG@3 score of 0.68. 
We have built a web application based on the tuned classifier.

### Reflection 
The final `RandomForestClassifier` model is a 15% improvement over the `DecisionTreeClassifier`. That's not a bad improvement. However, nDCG@3 score of 0.68 feels like there is room for improvement. We haven't exhausted the learning and I believe we can further explore other algorithms, such as, XGBoost and multi-layer perceptron neural networks. As a future task, I suggest we train multiple models, stack their learning and combine their predictions. Instead of using one model, we can use a network of models.

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

Before launching the web application, train the model and save it:
    
    python train.py

Starting the recommender web application:

    python serve.py

On a browser, open: 

    http://localhost:3001
