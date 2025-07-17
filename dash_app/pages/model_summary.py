# import os
# import pandas as pd
# from dash import html, dcc
# import dash
# import dash_bootstrap_components as dbc
# from src.utils.utils import load_object

# dash.register_page(__name__, path='/model', name="Model Summary")


# MODEL_PATH = "final_model/model.pkl"
# METRICS_PATH = "artifacts/regime_forecasting/metrics.csv"

# if os.path.exists(MODEL_PATH):
#     model = load_object(MODEL_PATH)
#     model_name = model.__class__.__name__
# else:
#     model = None
#     model_name = "Model not found"

# if os.path.exists(METRICS_PATH):
#     metrics_df = pd.read_csv(METRICS_PATH)
#     metrics_table = dbc.Table.from_dataframe(metrics_df, striped=True, bordered=True, hover=True)
# else:
#     metrics_table = html.P("No metrics available.")

# layout = html.Div([
#     html.H4("📋 Model Summary"),
#     html.P(f"Best Model: {model_name}"),
#     html.Hr(),
#     html.H5("Evaluation Metrics (Train/Test):"),
#     metrics_table
# ])