import dash
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from src.utils.dash_utils import (load_cluster_data, confidence_interval_plot, 
                                    create_sp500_regime_plot, radar_regime_plot, 
                                    generate_yellowbrick_silhouette_image, compute_regime_ttests)

# Register page
dash.register_page(__name__, path="/clustering", name="Regime Clustering")

# Load some data once at app start
df, regime_stats, n_clusters = load_cluster_data()
image_base64 = generate_yellowbrick_silhouette_image(df)
ttest_results = compute_regime_ttests(df)


layout = dbc.Container([
    html.H2("Market Regime Clustering Summary"),
    html.P(f"Number of regimes detected: {n_clusters}"),

    dbc.Card([
        dbc.CardHeader("Regime-Colored Return Series"),
        dbc.CardBody([
            dcc.Graph(
                id="regime-time-series",
                figure = create_sp500_regime_plot(df),
                style={'width': '100%', 'height': 'auto'}
            )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Regime Summary Statistics"),
        dbc.CardBody([
            dash.dash_table.DataTable(
                data=regime_stats.to_dict("records"),
                columns=[{"name": col, "id": col} for col in regime_stats.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"}
            )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Regime Feature Profiles (Radar Plot)"),
        dbc.CardBody([
            dcc.Graph(id="radar-plot", figure=radar_regime_plot(regime_stats))
        ])
    ], className="mb-4"),


    dbc.Card([
        dbc.CardHeader("T-Test Regime Comparison Between SP500 Mean Returns"),
        dbc.CardBody([
            dash_table.DataTable(
                    columns=[
                        {'name': 'Regime 1', 'id': 'Regime 1'},
                        {'name': 'Regime 2', 'id': 'Regime 2'},
                        {'name': 'T-statistic', 'id': 'T-statistic'},
                        {'name': 'P-value', 'id': 'P-value'},
                        {'name': 'Significant (p < 0.05)', 'id': 'Significant'}
                    ],
                    data=ttest_results,
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{Significant} eq "Yes"',
                                'column_id': 'P-value'
                            },
                            'backgroundColor': '#FFDDDD',
                            'color': 'black'
                        }
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold'}
        )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Mean Return Confidence Intervals by Regime"),
        dbc.CardBody([
            dcc.Graph(id="conf-plot", figure=confidence_interval_plot(df, n_clusters))
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Silhouette Plots (Yellowbrick, Static Image)"),
        dbc.CardBody([
            html.Img(src='data:image/png;base64,{}'.format(image_base64), style={'width': '100%', 'height': 'auto'})
        ]) 
    ], className="mb-4")
])
