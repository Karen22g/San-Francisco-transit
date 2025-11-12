import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from haversine import haversine, Unit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Database Config
DB_HOST = 'localhost'
DB_NAME = 'transit_streaming'
DB_USER = 'pachonarvaez'
DB_PORT = '5432'

# Connect to Database
engine = create_engine(f'postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Extract Data from vehicle_positions table
query = """
SELECT * FROM vehicle_positions
WHERE timestamp >= NOW() - INTERVAL '48 HOURS'
"""
data = pd.read_sql(query, engine)

# Data Cleaning
data.drop_duplicates(inplace=True)
data.fillna(method='ffill', inplace=True)  # Simple null handling
data = data[data['value'] < 100]  # Filter outliers based on some criteria

# Feature Engineering
def calculate_distance(row):
    return haversine((row['lat1'], row['lon1']), (row['lat2'], row['lon2']), unit=Unit.METERS)

data['acceleration'] = data['velocity'].diff() / data['timestamp'].diff().dt.total_seconds()
data['is_stopped'] = (data['velocity'] == 0).astype(int)
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) | (data['hour'] >= 17) & (data['hour'] <= 19)).astype(int)
data['distance_to_center'] = data.apply(calculate_distance, axis=1)
data['heading_change'] = data['heading'].diff().fillna(0)
data['speed_category'] = pd.cut(data['velocity'], bins=[0, 10, 20, 30, 40, 50], labels=['slow', 'medium', 'fast', 'very_fast'], include_lowest=True)

# Transformations for Machine Learning
features = data.drop(columns=['timestamp', 'value'])  # Drop useless columns
X = features.drop(columns=['target'])  # Exclude target variable if exists
y = features['target']  # Assuming a target column exists

# Encode categorical features
categorical_cols = ['speed_category']
numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed datasets
X_train.to_csv('train_data.csv', index=False)
X_test.to_csv('test_data.csv', index=False)
features.to_csv('features_engineered.csv', index=False)

print("Data preprocessing complete and datasets saved.")