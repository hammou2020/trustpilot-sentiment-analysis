import requests
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

body = dbc.Container([
    dbc.Col([
        html.Div([
            html.Img(src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                     style={"width": "auto", "maxHeight": "100%"}),
        ], style={"height": "100px"}),
        html.H2("What do you think of Github?"),
        dbc.Textarea(id="text_input",
                     rows="5",
                     style={"width": "100%", "marginBottom": "5px"}),
        dbc.Progress(id="score_bar", value=0,
                     style={"textAlign": "center"}),
        html.Div([
            dcc.Slider(id='rating_slider', min=1, max=5, value=5,
                       marks={i: str(i) for i in range(1, 6)})
        ], style={"width": "100%"}),
        dbc.Button("Submit review", color="primary",
                   style={"marginBottom": "5px", "width": "100%"}),
        html.Br(),
        dbc.Button("Review another brand", color="secondary",
                   style={"marginBottom": "5px", "width": "100%"}),
    ], className='text-center')
], style={"width": "350px", "marginTop": "5%"})

app.layout = html.Div([body])


@app.callback(
    [
        Output(component_id='score_bar', component_property='value'),
        Output(component_id='score_bar', component_property='children'),
        Output(component_id='score_bar', component_property='color'),
        Output(component_id='rating_slider', component_property='value')
    ],
    [
        Input(component_id='text_input', component_property='value'),
    ],
    [
        State(component_id='rating_slider', component_property='value')
    ]
)
def predict_sentiment(text, current_rating):
    if text is None or text.strip() == "":
        return 0, "", None, current_rating
    API_ENDPOINT = "http://127.0.0.1:5000/predict"
    data = {"review": text}
    r = requests.post(API_ENDPOINT, json=data)
    score = r.json()['score'] * 100
    if score < 100/3:
        color = "danger"
        rating = 1
    elif score < 200/3:
        color = "warning"
        rating = 3
    else:
        color = "success"
        rating = 5
    return score, f"{score:.2f}%", color, rating


if __name__ == '__main__':
    app.run_server(debug=True)
