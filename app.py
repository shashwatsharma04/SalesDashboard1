from flask import Flask, render_template
from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px

# Initialize Flask App
server = Flask(__name__)

# Initialize Dash App
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Load Dataset
def load_data():
    data = pd.read_csv('supermarket_sales.csv')
    # Convert the 'Date' column to datetime for proper time series analysis
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Dash Layout
app.layout = html.Div([
    html.H1("Supermarket Sales Dashboard"),

    # Product Line Sales Distribution
    html.Div([
        html.H2("Product Line Sales Distribution"),
        dcc.Graph(
            id='sales-by-product-line',
            figure=px.pie(
                data,
                names='Product line',
                values='Total',
                title="Sales Distribution by Product Line"
            )
        )
    ]),

    # Average Rating by Product Line
    html.Div([
        html.H2("Average Rating by Product Line"),
        dcc.Graph(
            id='average-rating-by-product-line',
            figure=px.bar(
                data.groupby('Product line')[['Rating']].mean().reset_index(),
                x='Product line',
                y='Rating',
                color='Product line',
                title="Average Rating by Product Line"
            )
        )
    ]),

    # Monthly Sales Trends
    html.Div([
        html.H2("Monthly Sales Trends"),
        dcc.Graph(
            id='monthly-sales',
            figure=px.line(
                data.groupby('Date').sum().reset_index(),
                x='Date',
                y='Total',
                title="Monthly Sales Trends"
            )
        )
    ]),

    # Payment Method Distribution
    html.Div([
        html.H2("Payment Method Distribution"),
        dcc.Graph(
            id='payment-method-distribution',
            figure=px.pie(
                data,
                names='Payment',
                values='Total',
                title="Payment Method Distribution"
            )
        )
    ]),
])

@server.route('/')
def index():
    return """
    <h1>Welcome to the Supermarket Sales Dashboard</h1>
    <p>Go to <a href="/dashboard/">Dashboard</a></p>
    """

if __name__ == '__main__':
    server.run(debug=True)
