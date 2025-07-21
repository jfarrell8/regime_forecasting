import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import datetime
from inference.predictor import RegimePredictor
from src.utils.dash_utils import load_cluster_data, load_naive_data, load_forecasting_metrics

dash.register_page(__name__, path='/prediction', name="Prediction")

df, regime_stats, n_clusters = load_cluster_data()
naive_pred, naive_test = load_naive_data()
forecast_metrics, naive_metrics = load_forecasting_metrics()

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
    ]),

    dbc.Card([
        dbc.CardHeader("Model Performance (Out of Sample)"),
        dbc.CardBody([
            html.P(f"Accuracy: {forecast_metrics['test_accuracy']:.2f}"),
            html.P(f"F1 Score: {forecast_metrics['test_f1_score']:.2f}"),
            html.P(f"Precision: {forecast_metrics['test_precision']:.2f}"),
            html.P(f"Recall: {forecast_metrics['test_recall']:.2f}"),
        ])
    ]),
    dbc.Card([
        dbc.CardHeader("Naive Model Performance (Out of Sample)"),
        dbc.CardBody([
            html.P(f"Accuracy: {naive_metrics['accuracy']:.2f}"),
            html.P(f"F1 Score: {naive_metrics['f1-score']:.2f}"),
            html.P(f"Precision: {naive_metrics['precision']:.2f}"),
            html.P(f"Recall: {naive_metrics['recall']:.2f}"),
        ])
    ]),
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
