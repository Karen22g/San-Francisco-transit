import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine

# Haversine function to calculate distance between GPS coordinates

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate speed from consecutive GPS positions for each vehicle

def calculate_speed_from_positions(df_vehicles):
    speeds = []
    for i in range(1, len(df_vehicles)):
        dist = haversine(df_vehicles.iloc[i-1]['latitude'], df_vehicles.iloc[i-1]['longitude'], 
                         df_vehicles.iloc[i]['latitude'], df_vehicles.iloc[i]['longitude'])
        time_diff = (df_vehicles.iloc[i]['timestamp'] - df_vehicles.iloc[i-1]['timestamp']).total_seconds() / 3600  # in hours
        speed = dist / time_diff if time_diff > 0 else 0
        # Filter unrealistic speeds over 150 km/h
        speeds.append(speed if speed <= 150 else 0)
    speeds.insert(0, 0)  # Assume speed is 0 for the first position
    df_vehicles['speed_calculated'] = speeds
    print("Calculando velocidades desde posiciones GPS...")
    return df_vehicles

# Modified analysis functions to use speeds_calculated instead of speeds from API

def analyze_data(df_vehicles):
    # Placeholder for data analysis logic using df_vehicles['speed_calculated']
    pass

# New plot function to show speed evolution over time

def plot_speed_over_time(df_vehicles):
    plt.figure(figsize=(12, 6))
    plt.plot(df_vehicles['timestamp'], df_vehicles['speed_calculated'], color='blue')
    plt.title('Speed Over Time')
    plt.xlabel('Time')
    plt.ylabel('Speed (km/h)')
    plt.grid()
    plt.show()

# Function to analyze geographic coverage

def analyze_geographic_coverage(df_vehicles, center_coordinates):
    df_vehicles['distance_from_center'] = df_vehicles.apply(
        lambda row: haversine(row['latitude'], row['longitude'], center_coordinates[0], center_coordinates[1]), axis=1)
    # Further coverage analysis logic here

# Main function

def main():
    # Connect to database
    engine = create_engine('postgresql://pachonarvaez@localhost:5432/transit_streaming')
    df_vehicles = pd.read_sql('SELECT * FROM vehicles', engine)  # Adjust SQL according to your schema
    # Call calculate_speed_from_positions to compute speeds
    df_vehicles = calculate_speed_from_positions(df_vehicles)
    # Run all analysis with calculated speeds
    analyze_data(df_vehicles)
    # Generate 6 visualizations including speed over time
    plot_speed_over_time(df_vehicles)
    # Add calls to other plot functions if necessary

if __name__ == '__main__':
    main()