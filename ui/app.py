import os

import numpy as np
import pandas as pd

import requests
from flask import request
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import config


def get_random_company():
    row = company_df.sample(1)
    name = row.iat[0, 1]
    url = row.iat[0, 2]
    logo = row.iat[0, 0]
    return name, url, logo


API_ENDPOINT = config.API_URL
company_df = pd.read_csv("companies_forbes.csv",
                         usecols=["company_logo", "company_name", "company_website"])
suggested_rating = None

company_name, company_url, company_logo = get_random_company()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

body = dbc.Container([
    dbc.Col([
        html.Div([
            html.A(html.Img(src=company_logo,
                            id="company_logo",
                            style={"width": "auto", "maxHeight": "100%"}),
                   id="company_link",
                   href=company_url)
        ], style={"height": "100px"}),
        html.H2([
            "What do you think of ",
            html.Span(company_name, id="company_name"),
            "?"
        ]),
        dbc.Textarea(id="text_input",
                     rows="5",
                     style={"width": "100%", "marginBottom": "5px"}),
        html.H5("Sentiment analysis"),
        dbc.Progress(html.Span(id='score_bar_label',
                               style={"color": "black",
                                      "marginLeft": "5px",
                                      "overflow": "visible",
                                      "position": "relative"}),
                     id="score_bar", value=0,
                     animated=False,
                     style={"textAlign": "center"}),
        html.H5("Propose a rating"),
        html.Div([
            dcc.Slider(id='rating_slider', min=1, max=5, value=5,
                       marks={i: str(i) for i in range(1, 6)})
        ], style={"width": "100%"}),
        dbc.Button("Submit review", color="primary",
                   id="submit_button",
                   style={"marginBottom": "5px", "width": "100%"}),
        html.Br(),
        dbc.Button("Review another brand", color="secondary",
                   id="change_button",
                   style={"marginBottom": "5px", "width": "100%"}),
    ], className='text-center')
], style={"width": "350px", "marginTop": "5%"})

app.layout = html.Div([body])


def score_to_rating(score):
    conds = [
        score < 20, score < 40, score < 60, score < 80, score < 100
    ]
    choices = [1, 2, 3, 4, 5]
    return np.select(conds, choices)


@app.callback(
    [
        Output('score_bar', 'value'),
        Output('score_bar', 'color'),
        Output('score_bar_label', 'children'),
        Output('rating_slider', 'value'),
        Output('submit_button', 'disabled'),
    ],
    [
        Input('text_input', 'value'),
    ],
    [
        State('rating_slider', 'value')
    ]
)
def predict_sentiment(text, current_rating):
    if text is None or text.strip() == "":
        return 0, None, "", current_rating, True
    api = os.path.join(API_ENDPOINT, "predict")
    data = {"review": text}
    r = requests.post(api, json=data)
    score = r.json()['score'] * 100
    if score < 100/3:
        color = "danger"
    elif score < 200/3:
        color = "warning"
    else:
        color = "success"
    rating = score_to_rating(score)
    return score, color, f"{score:.2f}%", rating, False


@app.callback(
    [
        Output('company_name', 'children'),
        Output('company_link', 'href'),
        Output('company_logo', 'src'),
    ],
    [Input('change_button', 'n_clicks')],
)
def change_company(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    print("change company callback invoked")
    return get_random_company()


@app.callback(
    [
        Output('change_button', 'n_clicks'),
        Output('text_input', 'value')
    ],
    [
        Input('submit_button', 'n_clicks')
    ],
    [
        State('text_input', 'value'),
        State('rating_slider', 'value'),
        State('score_bar', 'value'),
        State('company_name', 'children'),
        State('change_button', 'n_clicks')
    ]
)
def submit_review(n_clicks_submit, review, rating, score, company_name, n_clicks_change):
    if n_clicks_submit is None:
        raise PreventUpdate
    print("submit callback invoked ")
    api = os.path.join(API_ENDPOINT, "review")

    data = {
        'review': review,
        'rating': int(rating),
        'suggested_rating': int(score_to_rating(score)),
        'sentiment_score': float(score),
        'brand': company_name,
        'user_agent': request.headers.get('User-Agent'),
        'ip_address': request.remote_addr,
    }
    r = requests.post(api, json=data)
    if r.ok:
        print("Review saved to db")
    else:
        print("Error saving review to db")
    return (1, "") if n_clicks_change is None else (n_clicks_change + 1, "")


if __name__ == '__main__':
    app.run_server(debug=config.DEBUG, host=config.HOST)
