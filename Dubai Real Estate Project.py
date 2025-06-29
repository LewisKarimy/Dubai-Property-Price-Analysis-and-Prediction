# Import required libraries
import pandas as pd
import numpy as np
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys

# Check Tkinter availability and set Matplotlib backend
try:
    import tkinter
    matplotlib.use('TkAgg')  # Preferred backend for VS Code
    print("Using TkAgg backend for Matplotlib")
except ImportError:
    try:
        import PyQt5
        matplotlib.use('Qt5Agg')  # Fallback backend
        print("Tkinter not found. Using Qt5Agg backend for Matplotlib")
    except ImportError:
        print("Warning: Neither Tkinter nor PyQt5 found. Plots may not display. Install tkinter or PyQt5.")
        matplotlib.use('Agg')  # Non-interactive fallback to avoid crashes

# Confirm current Matplotlib backend
print("Current Matplotlib backend:", matplotlib.get_backend())

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create a simulated Dubai property dataset
# Features: Size (sqft), Bedrooms, Location, Property Type, Price (AED)
data = {
    'Size_sqft': np.random.randint(500, 4000, size=1000),  # Random sizes between 500-4000 sqft
    'Bedrooms': np.random.randint(1, 6, size=1000),        # 1-5 bedrooms
    'Location': np.random.choice(['Downtown Dubai', 'Dubai Marina', 'Palm Jumeirah', 
                                  'Business Bay', 'Jumeirah'], size=1000),  # Popular Dubai areas
    'Property_Type': np.random.choice(['Apartment', 'Villa', 'Townhouse'], size=1000),
    'Price_AED': np.zeros(1000)  # Placeholder for prices
}

# Generate realistic prices based on features (with some noise)
for i in range(1000):
    base_price = data['Size_sqft'][i] * 1000  # Base price per sqft
    if data['Location'][i] == 'Downtown Dubai':
        base_price *= 1.5
    elif data['Location'][i] == 'Palm Jumeirah':
        base_price *= 1.7
    elif data['Location'][i] == 'Dubai Marina':
        base_price *= 1.3
    if data['Property_Type'][i] == 'Villa':
        base_price *= 1.4
    elif data['Property_Type'][i] == 'Townhouse':
        base_price *= 1.2
    base_price += data['Bedrooms'][i] * 100000  # Additional cost per bedroom
    noise = np.random.normal(0, 50000)  # Add random noise
    data['Price_AED'][i] = round(base_price + noise)

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Data Cleaning
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check data types
print("\nData types:\n", df.dtypes)

# Convert Price_AED to integer
df['Price_AED'] = df['Price_AED'].astype(int)

# Step 3: SQL Integration
# Create SQLite database and table
conn = sqlite3.connect('dubai_properties.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS properties (
        Size_sqft INTEGER,
        Bedrooms INTEGER,
        Location TEXT,
        Property_Type TEXT,
        Price_AED INTEGER
    )
''')

# Insert DataFrame into SQL table
df.to_sql('properties', conn, if_exists='replace', index=False)

# SQL Query 1: Summary statistics
cursor.execute('''
    SELECT 
        AVG(Size_sqft) as avg_size,
        AVG(Bedrooms) as avg_bedrooms,
        AVG(Price_AED) as avg_price
    FROM properties
''')
sql_summary = cursor.fetchone()
print("\nSQL Summary Statistics:")
print(f"Average Size (sqft): {sql_summary[0]:.2f}")
print(f"Average Bedrooms: {sql_summary[1]:.2f}")
print(f"Average Price (AED): {sql_summary[2]:,.2f}")

# SQL Query 2: Average price by location
cursor.execute('''
    SELECT Location, AVG(Price_AED) as avg_price
    FROM properties
    GROUP BY Location
    ORDER BY avg_price
''')
sql_avg_price_location = pd.DataFrame(cursor.fetchall(), columns=['Location', 'avg_price'])
print("\nSQL Average Price by Location:\n", sql_avg_price_location)

# Close SQL connection
conn.close()

# Step 4: Exploratory Data Analysis (EDA) with Pandas
# Summary statistics (for comparison with SQL)
print("\nPandas Summary Statistics:\n", df.describe())

# Average price by location (for comparison with SQL)
avg_price_location = df.groupby('Location')['Price_AED'].mean().sort_values()
print("\nPandas Average Price by Location:\n", avg_price_location)

# Step 5: Visualizations (Matplotlib/Seaborn and Plotly)
# --- Matplotlib/Seaborn Visualizations (for display in VS Code) ---
# Set Seaborn style
plt.style.use('seaborn-v0_8')

# Matplotlib Plot 1: Bar plot of average price by location
plt.figure(figsize=(10, 6))
avg_price_location.plot(kind='bar', color='skyblue')
plt.title('Average Property Price by Location in Dubai')
plt.xlabel('Location')
plt.ylabel('Average Price (AED)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_price_by_location.png')  # Save for portfolio
plt.show()  # Display plot in VS Code

# Matplotlib Plot 2: Scatter plot of Size vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Size_sqft', y='Price_AED', hue='Property_Type', size='Bedrooms')
plt.title('Property Price vs Size by Property Type')
plt.xlabel('Size (sqft)')
plt.ylabel('Price (AED)')
plt.tight_layout()
plt.savefig('price_vs_size.png')  # Save for portfolio
plt.show()  # Display plot in VS Code

# --- Plotly Visualizations (for interactive HTML output) ---
# Plotly Plot 1: Bar plot of average price by location
fig1 = px.bar(sql_avg_price_location, x='Location', y='avg_price', 
              title='Average Property Price by Location in Dubai',
              labels={'avg_price': 'Average Price (AED)', 'Location': 'Location'},
              color='avg_price', color_continuous_scale='Blues')
fig1.update_layout(xaxis_tickangle=45)
fig1.write_html('avg_price_by_location.html')  # Save as interactive HTML

# Plotly Plot 2: Scatter plot of Size vs Price
fig2 = px.scatter(df, x='Size_sqft', y='Price_AED', color='Property_Type', size='Bedrooms',
                  title='Property Price vs Size by Property Type',
                  labels={'Size_sqft': 'Size (sqft)', 'Price_AED': 'Price (AED)'},
                  hover_data=['Location'])
fig2.write_html('price_vs_size.html')  # Save as interactive HTML

# Step 6: Prepare Data for Modeling
# Encode categorical variables (Location, Property_Type)
df_encoded = pd.get_dummies(df, columns=['Location', 'Property_Type'], drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('Price_AED', axis=1)
y = df_encoded['Price_AED']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build and Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make Predictions and Evaluate Model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} AED")
print(f"R-squared (R2): {r2:.2f}")

# Step 9: Visualize Predictions (Matplotlib and Plotly)
# --- Matplotlib Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Property Prices')
plt.xlabel('Actual Price (AED)')
plt.ylabel('Predicted Price (AED)')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')  # Save for portfolio
plt.show()  # Display plot in VS Code

# --- Plotly Visualization ---
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
fig3 = px.scatter(pred_df, x='Actual', y='Predicted',
                  title='Actual vs Predicted Property Prices',
                  labels={'Actual': 'Actual Price (AED)', 'Predicted': 'Predicted Price (AED)'})
fig3.add_trace(go.Scatter(x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                          y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                          mode='lines', line=dict(color='red', dash='dash'),
                          name='Ideal Line'))
fig3.write_html('actual_vs_predicted.html')  # Save as interactive HTML

# Step 10: Save Dataset for Reproducibility
df.to_csv('dubai_property_data.csv', index=False)
print("\nDataset saved as 'dubai_property_data.csv'")