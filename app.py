from flask import Flask
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Initialize Flask App
server = Flask(__name__)

# Initialize Dash App
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Load and preprocess the data
df = pd.read_csv('supermarket_sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Aggregate sales data monthly
monthly_sales = df['Total'].resample('M').sum()

# Generate Forecast
def generate_dynamic_forecast(steps=12):
    model = ARIMA(monthly_sales, order=(1, 1, 1))
    result = model.fit()
    forecast = result.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    zigzag = np.random.normal(0, forecast_mean.std() * 0.1, size=steps)
    upward_trend = np.linspace(0, forecast_mean.mean() * 0.2, steps) 
    dynamic_forecast = forecast_mean + zigzag + upward_trend

    forecast_index = pd.date_range(start=monthly_sales.index[-1], periods=steps + 1, freq='M')[1:]
    forecast_ci = forecast.conf_int()

    return forecast_index, dynamic_forecast, forecast_ci

# Create Forecast Plot
def create_forecast_plot(forecast_index, forecast_mean, forecast_ci):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines+markers', name='Actual Sales'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines+markers', name='Forecasted Sales', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(
        x=list(forecast_index) + list(forecast_index[::-1]),
        y=list(forecast_ci.iloc[:, 0]) + list(forecast_ci.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(255, 182, 193, 0.3)',
        line=dict(color='rgba(255, 182, 193, 0)'),
        name='Confidence Interval'
    ))
    fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Total Sales')
    return fig

# Create Scatter Plot
scatter_fig = px.scatter(df, x='Unit price', y='Total', color='Branch', hover_data=['Product line'], title='Scatter Plot of Sales')

# Create Pie Chart for Payment Methods
payment_fig = px.pie(df, names='Payment', title='Payment Method Distribution')

# Create Pie Chart for Product Line Sales
product_line_fig = px.pie(
    df,
    names='Product line',
    values='Total',
    title="Sales Distribution by Product Line"
)

# Create Bar Chart for Average Ratings by Product Line
rating_fig = px.bar(
    df.groupby('Product line')[['Rating']].mean().reset_index(),
    x='Product line',
    y='Rating',
    color='Product line',
    title="Average Rating by Product Line"
)

# Dash Layout
app.layout = html.Div([
    html.H1("Supermarket Sales Dashboard", style={'text-align': 'center'}),

    # Dropdown for Forecast Period
    html.Div([
        html.H2("Choose Forecast Period (Months)", style={'text-align': 'center'}),
        dcc.Dropdown(
            id='forecast-months-dropdown',
            options=[{'label': f'{i} months', 'value': i} for i in range(1, 25)],
            value=12,
            style={'width': '50%', 'margin': '0 auto'}
        ),
    ]),

    # Sales Forecast
    html.Div([
        html.H2("Monthly Sales Trends with Predictions", style={'text-align': 'center'}),
        dcc.Graph(id='forecast-graph'),
    ]),

    # Scatter Plot
    html.Div([
        html.H2("Scatter Plot of Sales", style={'text-align': 'center'}),
        dcc.Graph(id='scatter-graph', figure=scatter_fig),
    ]),

    # Payment Method Distribution
    html.Div([
        html.H2("Payment Method Distribution", style={'text-align': 'center'}),
        dcc.Graph(id='payment-graph', figure=payment_fig),
    ]),

    # Product Line Sales Distribution
    html.Div([
        html.H2("Product Line Sales Distribution", style={'text-align': 'center'}),
        dcc.Graph(id='product-line-graph', figure=product_line_fig),
    ]),

    # Average Rating by Product Line
    html.Div([
        html.H2("Average Rating by Product Line", style={'text-align': 'center'}),
        dcc.Graph(id='rating-graph', figure=rating_fig),
    ]),
])

# Callbacks
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-months-dropdown', 'value')]
)
def update_forecast(forecast_steps):
    forecast_index, forecast_mean, forecast_ci = generate_dynamic_forecast(forecast_steps)
    return create_forecast_plot(forecast_index, forecast_mean, forecast_ci)

@server.route('/')
def index():
    return """
    <h1>Welcome to the Supermarket Sales Dashboard</h1>
    <p>Go to <a href="/dashboard/">Dashboard</a></p>
    """

if __name__ == '__main__':
    server.run(debug=True)
