import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import uuid

# Step 1: Data Collection
# Load the dataset (assuming EV_Dataset.csv is available)
df = pd.read_csv('EV_Dataset.csv')

# Step 2: Data Preprocessing
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert 'Year' to integer
df['Year'] = df['Year'].astype(int)

# Convert categorical columns to category type
categorical_columns = ['Month_Name', 'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
df[categorical_columns] = df[categorical_columns].astype('category')

# Check for missing values and handle them
df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Step 3: Feature Engineering
# Extract Month and Day from Date
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)

# Step 4: Machine Learning Model
# Split data into features and target
X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Feature importance
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

# Step 5: Create Dashboard with Plotly Dash
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Electric Vehicle Sales Analysis Dashboard - India", style={'textAlign': 'center', 'color': '#1f77b4'}),

    # Dropdown for selecting year
    html.Label("Select Year:"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in df['Year'].unique()],
        value=df['Year'].max(),
        style={'width': '50%'}
    ),

    # Pie Chart for Vehicle Category Distribution
    dcc.Graph(id='pie-chart-vehicle-category'),

    # Scatter Plot for Sales vs Year by State
    dcc.Graph(id='scatter-plot-sales-state'),

    # Bar Plot for State-wise Sales
    dcc.Graph(id='bar-plot-state-sales'),

    # Feature Importance Plot
    dcc.Graph(id='feature-importance-plot')
])

# Callback to update graphs based on year selection
@app.callback(
    [Output('pie-chart-vehicle-category', 'figure'),
     Output('scatter-plot-sales-state', 'figure'),
     Output('bar-plot-state-sales', 'figure'),
     Output('feature-importance-plot', 'figure')],
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

    # Scatter Plot: Sales vs Year by State
    scatter_fig = px.scatter(
        filtered_df,
        x='Month',
        y='EV_Sales_Quantity',
        color='State',
        size='EV_Sales_Quantity',
        title=f'Monthly EV Sales by State in {selected_year}',
        labels={'Month': 'Month', 'EV_Sales_Quantity': 'EV Sales Quantity'}
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

    # Feature Importance Plot
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.index[:10],
        'Importance': feature_importance.values[:10]
    })
    feature_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        title='Top 10 Feature Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )

    return pie_fig, scatter_fig, bar_fig, feature_fig

# Step 6: Save static plots for documentation
# Yearly Sales Trend
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color='b')
plt.title('Yearly Analysis of EV Sales in India')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.savefig('yearly_sales_trend.png')
plt.close()

# Vehicle Type Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Type', y='EV_Sales_Quantity', data=df, hue='Vehicle_Type', palette='bright')
plt.title('Analysis by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('EV Sales')
plt.xticks(rotation=90)
plt.savefig('vehicle_type_analysis.png')
plt.close()

# Step 7: Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)