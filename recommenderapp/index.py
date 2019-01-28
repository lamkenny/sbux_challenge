import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import pickle
import bz2
import gzip

from recommenderapp import app

app.css.append_css({'external_url': dbc.themes.BOOTSTRAP})

# Load the save model.
model_filename = 'models/rf.pkl.bz2'
print('Loading {}...this may take awhile'.format(model_filename))
with bz2.BZ2File(model_filename, 'r') as file:
     model = pickle.load(file)

offer_df = pd.read_json('data/app/offers.json', orient='records', lines=True)
offer_df['offer_type'].fillna('', inplace=True)
offer_df['channels'].fillna('', inplace=True)
offer_df.fillna(0, inplace=True)
demographics_df = pd.read_json('data/app/demographics.json', orient='records', lines=True)
gender_values = sorted(demographics_df['gender'].unique())
became_member_on_values = sorted(demographics_df['became_member_on'].unique())
age_values = sorted(demographics_df['age'].unique())
income_values = sorted(demographics_df['income'].unique())

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Offer Recommender</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        </footer>
    </body>
</html>
'''

navbar = html.Div(
    [
        html.A('Offer Recommender', href='#', className='navbar-brand')
    ],
    className='navbar navbar-expand-sm navbar-dark bg-dark sticky-top'
)

header_bg = html.Div(
    [
        html.Img(src='assets/header-bg-slim.png',
            style={
                'width': '100%',
                'height': '100%',
                'margin-bottom': '50px'
            }
            )
    ]
)


became_member_on_dropdown = dbc.FormGroup(
    [
        dbc.Label("Became member on:"),
        dcc.Dropdown(
            id="became_member_on_dropdown",
            options=[{"label": n, "value": n} for n in became_member_on_values],
        ),
    ]
)

age_dropdown = dbc.FormGroup(
    [
        dbc.Label("Age:"),
        dcc.Dropdown(
            id="age_dropdown",
            options=[{"label": n, "value": n} for n in age_values],
        ),
    ]
)

income_dropdown = dbc.FormGroup(
    [
        dbc.Label("Income:"),
        dcc.Dropdown(
            id="income_dropdown",
            options=[{"label": "${}".format(n), "value": n} for n in income_values],
        ),
    ]
)

def to_label_html_label(gender):
    if gender=='M':
        return 'Male'
    if gender=='F':
        return 'Female'
    if gender=='O':
        return 'Other'
    if gender=='U':
        return 'Unspecified'

gender_dropdown = dbc.FormGroup(
    [
        dbc.Label("Gender:"),
        dcc.Dropdown(
            id="gender_dropdown",
            options=[{"label": to_label_html_label(n), "value": n} for n in gender_values]
        ),
    ]
)

inputs = dbc.Row(
            [
                dbc.Col(gender_dropdown, align='center'),
                dbc.Col(age_dropdown, align='center'),
                dbc.Col(income_dropdown, align='center'),
                dbc.Col(became_member_on_dropdown, align='center'),
            ]
        )

offer = html.Div(
    [
        html.Div([html.H2("", id='offer_id')], className="text-lg-center text-success"),
        html.Div([html.P("", id='offer_type')], className="text-lg-center font-weight-bold"),
        html.Div([html.P("", id='reward')], className="text-lg-center"),
        html.Div([html.P("", id='difficulty')], className="text-lg-center"),
        html.Div([html.P("", id='channels')], className="text-lg-center")
    ]
    
    )

container = dbc.Container(
    [
        html.P("Enter demographic data for a user", className='font-weight-bold'),
        html.P("This tool returns an offer that gets the highest response from the user. May return no offer if no offer is found."),
        html.P("Try one of the followings:"),
        html.P("Female 65 $67000 20170907", style={'padding-left': '50px'}),
        html.P("Male 23 $32000 20180722", style={'padding-left': '50px'}),
        html.P("Female 49 $73000 20131026", style={'padding-left': '50px'}),
        html.P("Male 31 $38000 20180222", style={'padding-left': '50px'}),
        html.P("Other 93 $35000 20170904", style={'padding-left': '50px'}),
        html.Hr(),
        inputs,
        html.Br(),
        html.Hr(),
        offer,
    ])

app.layout = html.Div(
    [
        navbar,
        header_bg,
        container
    ])

def is_no_offer(offer):
    """ Check or not the recommended item is not an offer. """
    return (str(offer['offer_id']) == 'no_offer')

@app.callback(
    Output("offer_id", "children"),
    [
        Input("gender_dropdown", "value"), 
        Input("income_dropdown", "value"), 
        Input("age_dropdown", "value"), 
        Input("became_member_on_dropdown", "value")
    ]
    )
def get_offer_id(gender, income, age, became_member_on):
    """ Callback for getting the offer ID of the recommended offer. """
    offer = get_offer(gender, income, age, became_member_on)
    if isinstance(offer, type(None)):
        return ""

    if is_no_offer(offer):
        return 'No offer is required for this demographic group.'
    else:
        return offer['offer_id']

@app.callback(
    Output("offer_type", "children"),
    [
        Input("gender_dropdown", "value"), 
        Input("income_dropdown", "value"), 
        Input("age_dropdown", "value"), 
        Input("became_member_on_dropdown", "value")
    ]
    )
def get_offer_type(gender, income, age, became_member_on):
    """ Callback for getting the offer type label of the recommended offer. """
    offer = get_offer(gender, income, age, became_member_on)
    if isinstance(offer, type(None)):
        return ""
    if is_no_offer(offer):
        return ""

    offer_type = offer['offer_type']
    if str(offer_type) == 'None':
        return ""
    return offer_type

@app.callback(
    Output("reward", "children"),
    [
        Input("gender_dropdown", "value"), 
        Input("income_dropdown", "value"), 
        Input("age_dropdown", "value"), 
        Input("became_member_on_dropdown", "value")
    ]
    )
def get_offer_reward(gender, income, age, became_member_on):
    """ Callback for getting the reward label of the recommended offer. """
    offer = get_offer(gender, income, age, became_member_on)
    if isinstance(offer, type(None)):
        return ""
    if is_no_offer(offer):
        return ""

    reward = offer['reward']
    return ('Reward: {}'.format(reward))

@app.callback(
    Output("difficulty", "children"),
    [
        Input("gender_dropdown", "value"), 
        Input("income_dropdown", "value"), 
        Input("age_dropdown", "value"), 
        Input("became_member_on_dropdown", "value")
    ]
    )
def get_offer_difficulty(gender, income, age, became_member_on):
    """ Callback for getting the difficulty label of the recommended offer. """
    offer = get_offer(gender, income, age, became_member_on)
    if isinstance(offer, type(None)):
        return ""
    if is_no_offer(offer):
        return ""
    difficulty = offer['difficulty']
    return ('Difficulty: {}'.format(difficulty))

@app.callback(
    Output("channels", "children"),
    [
        Input("gender_dropdown", "value"), 
        Input("income_dropdown", "value"), 
        Input("age_dropdown", "value"), 
        Input("became_member_on_dropdown", "value")
    ]
    )
def get_offer_channels(gender, income, age, became_member_on):
    """ Callback for getting the channel label for the recommended offer. """
    offer = get_offer(gender, income, age, became_member_on)
    if isinstance(offer, type(None)):
        return ""
    if is_no_offer(offer):
        return ""
        
    channels = offer['channels']
    if str(channels) == 'None':
        return ""
    return ('Channels: {}'.format(channels))

def get_offer(gender, income, age, became_member_on):
    """ Recommend an offer for the selected demograhic group."""
    if (gender == None or income == None or age == None or became_member_on == None):
        return None

    print("gender {}".format(gender))
    print("income {}".format(income))
    print("age {}".format(age))
    print("became_member_on {}".format(became_member_on))

    offer_label = predict_offer(gender, income, age, became_member_on)
    print('predicted label {}'.format(offer_label))
    print('predicted {}'.format(offer_df.loc[offer_label]))
    return offer_df.loc[offer_label]

def predict_offer(gender, income, age, became_member_on):
    """ From demographic data, predict the offer label. """
    gender_features = to_gender_features(gender)
    # features = ['age', 'income', 'became_member_on', 'gender_M', 'gender_O', 'gender_U']
    X = [[age, income, became_member_on, 1, 0, 0]]
    offer_label = model.predict(X)[0]
    return offer_label

def to_gender_features(gender):
    """Construct input values for the model."""

    # the model accepts ['gender_M'   'gender_O'    'gender_U'];
    gender_M = 0
    gender_O = 0
    gender_U = 0

    if gender == 'Male':
        gender_M = 1

    if gender == 'Other':
        gender_O = 1

    if (gender_U == 'Unspecified'):
        gender_U = 1

    return [gender_M, gender_O, gender_U]

