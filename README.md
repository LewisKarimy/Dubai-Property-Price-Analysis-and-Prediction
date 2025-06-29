Dubai Property Price Analysis and Prediction
Overview
This project analyzes a simulated Dubai property dataset to predict prices using features like size, bedrooms, location, and property type. It showcases a data science workflow with data generation, cleaning, SQL storage, EDA, visualization, and linear regression modeling.
Objectives

Generate a realistic dataset of 1,000 Dubai properties.
Store data in a SQLite database and query statistics.
Perform EDA with Pandas and visualize trends using Matplotlib, Seaborn, and Plotly.
Predict prices with a linear regression model and evaluate performance (RMSE, R²).
Save dataset and visualizations for reproducibility.

Features

Dataset: Simulated with Size (sqft), Bedrooms, Location (e.g., Downtown Dubai), Property Type (Apartment, Villa, Townhouse), and Price (AED).
SQL: Stores data in dubai_properties.db for querying.
Visualizations: Bar plots (prices by location), scatter plots (price vs. size), and actual vs. predicted price plots.
Modeling: Linear regression with one-hot encoded features.
Outputs: CSV dataset, PNG static plots, HTML interactive plots.

Requirements
pip install pandas numpy sqlite3 matplotlib seaborn plotly scikit-learn

Optional: tkinter or PyQt5 for Matplotlib plots.
Usage
Run dubai_property_analysis.py to:

Generate and save dubai_property_data.csv.
Store data in dubai_properties.db.
Output EDA, model metrics, and visualizations (PNG: avg_price_by_location.png, price_vs_size.png, actual_vs_predicted.png; HTML: interactive Plotly plots).

Notes

Uses np.random.seed(42) for reproducibility.
Matplotlib backend adapts to available dependencies (TkAgg, Qt5Agg, or Agg).
Simulated data mimics Dubai property trends.

Future Improvements

Use real property data.
Test advanced ML models.
Add more features (e.g., amenities).
Build a web interface for exploration.



