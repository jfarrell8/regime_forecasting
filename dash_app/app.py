import dash
from dash import Dash, html, page_registry
from dash_bootstrap_components.themes import BOOTSTRAP
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Market Regime App"


sidebar = html.Div(
    [
        html.H2("Navigation", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(page["name"], href=page["path"], active="exact")
                for page in page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"padding": "1rem"},
)

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(sidebar, width=2),
            dbc.Col(dash.page_container, width=10)
        ])
    ],
    fluid=True
)

# server = app.server  # for gunicorn deployment

if __name__ == '__main__':
    app.run(debug=True)
