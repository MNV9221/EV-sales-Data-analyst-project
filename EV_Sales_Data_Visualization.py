import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import uuid

# Step 1: Load and Preprocess the Dataset
# Load the dataset (assuming EV_Dataset.csv is available)
df = pd.read_csv('EV_Dataset.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert 'Year' to integer
df['Year'] = df['Year'].astype(int)

# Convert categorical columns to category type
categorical_columns = ['Month_Name', 'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
df[categorical_columns] = df[categorical_columns].astype('category')

# Handle missing values
df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Step 2: Create Static Visualizations with Seaborn and Save Them
# Visualization 1: Yearly Sales Trend (Line Plot)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color='b')
plt.title('Yearly Analysis of EV Sales in India')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.savefig('yearly_sales_trend.png')
plt.close()

# Visualization 2: Monthly Sales Trend (Line Plot)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month_Name', y='EV_Sales_Quantity', data=df, marker='o', color='r')
plt.title('Monthly Analysis of EV Sales in India')
plt.xlabel('Month')
plt.ylabel('EV Sales')
plt.xticks(rotation=45)
plt.savefig('monthly_sales_trend.png')
plt.close()

# Visualization 3: State-wise Sales (Bar Plot)
state_sales = df.groupby('State')['EV_Sales_Quantity'].sum().reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(y='State', x='EV_Sales_Quantity', data=state_sales, hue='State', palette='bright')
plt.title('State-Wise Analysis of EV Sales')
plt.xlabel('EV Sales')
plt.ylabel('States')
plt.savefig('state_wise_sales.png')
plt.close()

# Visualization 4: Vehicle Category Distribution (Pie Chart)
vehicle_category_counts = df['Vehicle_Category'].value_counts().reset_index()
vehicle_category_counts.columns = ['Vehicle_Category', 'Count']
plt.figure(figsize=(8, 8))
plt.pie(vehicle_category_counts['Count'], labels=vehicle_category_counts['Vehicle_Category'], autopct='%1.1f%%', colors=sns.color_palette('bright'))
plt.title('Vehicle Category Distribution')
plt.savefig('vehicle_category_pie.png')
plt.close()

# Visualization 5: Sales by Vehicle Type (Bar Plot)
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Type', y='EV_Sales_Quantity', data=df, hue='Vehicle_Type', palette='bright')
plt.title('Analysis by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('EV Sales')
plt.xticks(rotation=90)
plt.savefig('vehicle_type_analysis.png')
plt.close()

# Visualization 6: Scatter Plot of Sales vs Month by State
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Month_Name', y='EV_Sales_Quantity', hue='State', size='EV_Sales_Quantity', data=df)
plt.title('Monthly EV Sales by State')
plt.xlabel('Month')
plt.ylabel('EV Sales')
plt.xticks(rotation=45)
plt.savefig('monthly_sales_by_state_scatter.png')
plt.close()

# Step 3: Create Interactive Dashboard with Plotly Dash
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Electric Vehicle Sales Visualization Dashboard - India", style={'textAlign': 'center', 'color': '#1f77b4'}),

    # Dropdown for selecting year
    html.Label("Select Year:"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in df['Year'].unique()],
        value=df['Year'].max(),
        style={'width': '50%'}
    ),

    # Pie Chart for Vehicle Category Distribution
    html.H3("Vehicle Category Distribution"),
    dcc.Graph(id='pie-chart-vehicle-category'),

    # Scatter Plot for Monthly Sales by State
    html.H3("Monthly EV Sales by State"),
    dcc.Graph(id='scatter-plot-sales-state'),

    # Bar Plot for State-wise Sales
    html.H3("State-wise EV Sales"),
    dcc.Graph(id='bar-plot-state-sales'),

    # Line Plot for Yearly Sales Trend
    html.H3("Yearly EV Sales Trend"),
    dcc.Graph(id='line-plot-yearly-sales'),

    # Bar Plot for Vehicle Type
    html.H3("Sales by Vehicle Type"),
    dcc.Graph(id='bar-plot-vehicle-type')
])

# Callback to update graphs based on year selection
@app.callback(
    [Output('pie-chart-vehicle-category', 'figure'),
     Output('scatter-plot-sales-state', 'figure'),
     Output('bar-plot-state-sales', 'figure'),
     Output('line-plot-yearly-sales', 'figure'),
     Output('bar-plot-vehicle-type', 'figure')],
    [Input('year-dropdown', 'value')]
)
def update_graphs(selected_year):
    # Filter data for selected year
    filtered_df = df[df['Year'] == selected_year]

    # Pie Chart: Vehicle Category Distribution
    vehicle_category_counts = filtered_df['Vehicle_Category'].value_counts().reset_index()
    vehicle_category_counts.columns = ['Vehicle_Category', 'Count']
    pie_fig = px.pie(
        vehicle_category_counts,
        values='Count',
        names='Vehicle_Category',
        title=f'Vehicle Category Distribution in {selected_year}',
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Scatter Plot: Monthly Sales by State
    scatter_fig = px.scatter(
        filtered_df,
        x='Month_Name',
        y='EV_Sales_Quantity',
        color='State',
        size='EV_Sales_Quantity',
        title=f'Monthly EV Sales by State in {selected_year}',
        labels={'Month_Name': 'Month', 'EV_Sales_Quantity': 'EV Sales Quantity'}
    )

    # Bar Plot: State-wise Sales
    state_sales = filtered_df.groupby('State')['EV_Sales_Quantity'].sum().reset_index()
    bar_fig = px.bar(
        state_sales,
        x='EV_Sales_Quantity',
        y='State',
        title=f'State-wise EV Sales in {selected_year}',
        color='EV_Sales_Quantity',
        color_continuous_scale='Viridis'
    )

    # Line Plot: Yearly Sales Trend (for all years, shown for context)
    yearly_sales = df.groupby('Year')['EV_Sales_Quantity'].sum().reset_index()
    line_fig = px.line(
        yearly_sales,
        x='Year',
        y='EV_Sales_Quantity',
        title='Yearly EV Sales Trend',
        markers=True,
        line_shape='linear',
        color_discrete_sequence=['blue']
    )

    # Bar Plot: Vehicle Type
    vehicle_type_sales = filtered_df.groupby('Vehicle_Type')['EV_Sales_Quantity'].sum().reset_index()
    vehicle_type_fig = px.bar(
        vehicle_type_sales,
        x='Vehicle_Type',
        y='EV_Sales_Quantity',
        title=f'Sales by Vehicle Type in {selected_year}',
        color='EV_Sales_Quantity',
        color_continuous_scale='Plasma'
    )
    vehicle_type_fig.update_layout(xaxis_tickangle=45)

    return pie_fig, scatter_fig, bar_fig, line_fig, vehicle_type_fig

# Step 4: Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)