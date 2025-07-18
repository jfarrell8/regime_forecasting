import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import io
import base64
from itertools import combinations
from scipy.stats import ttest_ind

def load_cluster_data():
    df = pd.read_csv("artifacts/clustering/regimes.csv", index_col="Date", parse_dates=True)
    stats = pd.read_csv("artifacts/clustering/regime_stats.csv")
    with open("artifacts/clustering/num_clusters.txt") as f:
        n_clusters = int(f.read())
    return df, stats, n_clusters

def load_naive_data():
    naive_pred = pd.read_csv("artifacts/regime_model_forecasting/naive_forecasting.csv")
    naive_test = pd.read_csv("artifacts/regime_model_forecasting/naive_ytest.csv")

    return naive_pred, naive_test

# --- Generate Silhouette plots with Yellowbrick (matplotlib)
def generate_yellowbrick_silhouette_image(data, k_vals=[2, 3, 4, 5]):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['SP500_ret', 'SP500_vol_21d', '^VIX']].dropna())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, k in zip(axes, k_vals):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax)
        visualizer.fit(X)
        visualizer.finalize()
        ax.set_title(f'Silhouette Plot for k={k}')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded


def confidence_interval_plot(data, n_clusters):

    # Define mappings
    features = {
        'SP500_ret': 'Mean Daily Return',
        'SP500_vol_21d': '21-Day Rolling Volatility',
        '^VIX': 'VIX Level'
    }
    
    color_options = ['skyblue', 'orange', 'lightgreen', 'red', 'black']
    colors = [color_options[j] for j in range(n_clusters)]
    # y_labels = ['Regime 0', 'Regime 1', 'Regime 2']
    y_labels = [f'Regime {i}' for i in range(n_clusters)]
    regimes = sorted(data['regime'].unique())
    y_positions = np.arange(len(regimes))

    # Initialize subplot figure
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[v for k, v in features.items()])

    # Build each subplot
    for col_index, (feature_key, x_label) in enumerate(features.items(), start=1):
        means = []
        errors = []

        for i in regimes:
            sample = data[data['regime'] == i][feature_key].dropna()
            m = np.mean(sample)
            se = stats.sem(sample)
            h = se * stats.t.ppf(0.975, len(sample) - 1)
            means.append(m)
            errors.append(h)

        for i, (mean, error, color) in enumerate(zip(means, errors, colors)):
            # CI bar (horizontal line)
            fig.add_trace(go.Scatter(
                x=[mean - error, mean + error],
                y=[y_positions[i]] * 2,
                mode='lines',
                line=dict(color=color, width=6),
                showlegend=False
            ), row=1, col=col_index)

            # Mean vertical marker
            fig.add_trace(go.Scatter(
                x=[mean, mean],
                y=[y_positions[i] - 0.1, y_positions[i] + 0.1],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=1, col=col_index)

        # Optional zero line for returns
        if 'Return' in x_label:
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[-0.5, len(regimes) - 0.5],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ), row=1, col=col_index)

    # Layout tweaks
    fig.update_layout(
        height=400,
        width=1000,
        title_text="Confidence Intervals by Regime",
        template='simple_white',
        margin=dict(t=60),
    )

    # Y-axis labeling only on first subplot
    fig.update_yaxes(
        tickvals=y_positions,
        ticktext=y_labels,
        row=1,
        col=1,
        title='Regime'
    )
    # Hide y-axis ticks on other plots
    for i in range(2, n_clusters+1):
        fig.update_yaxes(showticklabels=False, row=1, col=i)

    return fig


def radar_regime_plot(regime_stats):
    regime_stats['Mean VIX Level'] = regime_stats['Mean VIX Level'] / 1000

    features = ['Mean Return', 'Return Std Dev', 'Mean Volatility (21d)', 'Mean VIX Level']

    fig = go.Figure()

    for i in range(len(regime_stats)):
        fig.add_trace(go.Scatterpolar(
            r=regime_stats.loc[i, features].values,
            theta=features,
            fill='toself',
            name=f'Regime {i}'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            # range=[0, 5]
            ))
    )

    return fig




def create_sp500_regime_plot(data):
    fig = go.Figure()

    # Line for context
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SP500_ret'],
        mode='lines',
        name='S&P 500 Rets',
        line=dict(color='gray'),
        opacity=0.4
    ))

    # Scatter points colored by regime
    regimes = sorted(data['regime'].unique())
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, regime in enumerate(regimes):
        df_regime = data[data['regime'] == regime]
        fig.add_trace(go.Scatter(
            x=df_regime.index,
            y=df_regime['SP500_ret'],
            mode='markers',
            name=f'Regime {regime}',
            marker=dict(color=colors[i % len(colors)], size=5),
            showlegend=True
        ))

    fig.update_layout(
        title='S&P 500 Daily Returns Colored By Regime',
        xaxis_title='Date',
        yaxis_title='Daily Return',
        template='plotly_white'
    )

    return fig


def compute_regime_ttests(data):
    results = []
    regimes = sorted(data['regime'].unique())
    
    for reg1, reg2 in combinations(regimes, 2):
        r1 = data[data['regime'] == reg1]['SP500_ret']
        r2 = data[data['regime'] == reg2]['SP500_ret']
        t_stat, p_val = ttest_ind(r1, r2, equal_var=False)

        results.append({
            'Regime 1': reg1,
            'Regime 2': reg2,
            'T-statistic': round(t_stat, 2),
            'P-value': round(p_val, 4),
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })
    
    return results