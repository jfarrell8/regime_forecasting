import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import datetime
from inference.predictor import RegimePredictor

dash.register_page(__name__, path='/prediction', name="Prediction")

# Predictor instance
predictor = RegimePredictor()

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("📈 Predict Tomorrow's Market Regime"), width=12)
    ], className="my-2"),

    dbc.Row([
        dbc.Col([
            html.P("Click below to generate a prediction using the most recent market data."),
            dbc.Button("Predict", id="predict-button", color="primary", className="me-2"),
            html.Div(id="prediction-output", className="mt-3")
        ], width=6)
    ])
])


@dash.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True
)
def run_prediction(n_clicks):
    try:
        prediction = predictor.predict()
        if prediction == -1:
            return dbc.Alert("Error generating prediction.", color="danger")

        return dbc.Alert(f"📊 Predicted Regime for Tomorrow: {prediction}", color="success")

    except Exception as e:
        return dbc.Alert(f"Unexpected error: {str(e)}", color="danger")
