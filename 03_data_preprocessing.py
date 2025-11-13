"""
PREPROCESAMIENTO DE DATOS - SF TRANSIT
Feature Engineering + Limpieza + Transformación para ML
Autor: Francisco Narvaez M
Fecha: 2025-11-12
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DB_CONFIG = {
    'host': 'karenserver.postgres.database.azure.com',
    'database': 'transit_streaming',
    'user': 'admin_karen',
    'password': 'Tiendala60',
    'port': 5432
}

# Centro geográfico de San Francisco (calculado del análisis exploratorio)
SF_CENTER_LAT = 37.759011
SF_CENTER_LON = -122.358909

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Calcular distancia entre dos puntos GPS en km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def connect_db():
    """Conectar a la base de datos"""
    return psycopg2.connect(**DB_CONFIG)

def get_vehicle_positions(conn, hours=48):
    """Obtener posiciones de vehículos"""
    query = f"""
        SELECT 
            id,
            vehicle_id,
            route_id,
            trip_id,
            agency_id,
            latitude,
            longitude,
            speed,
            heading,
            timestamp,
            created_at
        FROM vehicle_positions
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        ORDER BY vehicle_id, timestamp ASC
    """
    return pd.read_sql(query, conn)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_speed_from_positions(df):
    """Calcular velocidad desde posiciones GPS"""
    print("\n🔄 Calculando velocidades desde GPS...")
    
    df = df.copy().sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    df['speed_calculated'] = np.nan
    df['distance_traveled'] = np.nan
    df['time_diff_seconds'] = np.nan
    
    for vehicle_id in df['vehicle_id'].unique():
        mask = df['vehicle_id'] == vehicle_id
        indices = df[mask].index.tolist()
        
        for i in range(1, len(indices)):
            prev_idx = indices[i-1]
            curr_idx = indices[i]
            
            prev_row = df.loc[prev_idx]
            curr_row = df.loc[curr_idx]
            
            distance = haversine(
                prev_row['latitude'], prev_row['longitude'],
                curr_row['latitude'], curr_row['longitude']
            )
            
            time_diff = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds()
            
            if time_diff > 0:
                speed = (distance / time_diff) * 3600
                if speed <= 150:
                    df.loc[curr_idx, 'speed_calculated'] = speed
                    df.loc[curr_idx, 'distance_traveled'] = distance
                    df.loc[curr_idx, 'time_diff_seconds'] = time_diff
    
    print(f"✅ Velocidades calculadas: {df['speed_calculated'].notna().sum():,} registros")
    return df

def calculate_acceleration(df):
    """Calcular aceleración (cambio de velocidad)"""
    print("\n🔄 Calculando aceleraciones...")
    
    df = df.copy().sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    df['acceleration'] = np.nan
    
    for vehicle_id in df['vehicle_id'].unique():
        mask = df['vehicle_id'] == vehicle_id
        vehicle_data = df[mask].copy()
        
        # Calcular cambio de velocidad / tiempo
        speed_diff = vehicle_data['speed_calculated'].diff()
        time_diff = vehicle_data['time_diff_seconds']
        
        # Aceleración en m/s²
        acceleration = (speed_diff / 3.6) / time_diff  # convertir km/h a m/s
        
        df.loc[mask, 'acceleration'] = acceleration
    
    print(f"✅ Aceleraciones calculadas: {df['acceleration'].notna().sum():,} registros")
    return df

def create_temporal_features(df):
    """Crear features temporales"""
    print("\n🔄 Creando features temporales...")
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Hora pico: 7-9am y 5-7pm
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    
    # Periodo del día
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    df['time_of_day'] = df['hour'].apply(time_of_day)
    
    print("✅ Features temporales creados")
    return df

def create_geographic_features(df):
    """Crear features geográficos"""
    print("\n🔄 Creando features geográficos...")
    
    # Distancia al centro de SF
    df['distance_to_center'] = df.apply(
        lambda row: haversine(SF_CENTER_LAT, SF_CENTER_LON, 
                             row['latitude'], row['longitude']),
        axis=1
    )
    
    # Zona geográfica (basado en latitud/longitud)
    def get_zone(lat, lon):
        if lat > SF_CENTER_LAT + 0.05:
            return 'north'
        elif lat < SF_CENTER_LAT - 0.05:
            return 'south'
        elif lon > SF_CENTER_LON + 0.05:
            return 'east'
        elif lon < SF_CENTER_LON - 0.05:
            return 'west'
        else:
            return 'central'
    
    df['zone'] = df.apply(lambda row: get_zone(row['latitude'], row['longitude']), axis=1)
    
    # ¿Está en downtown? (dentro de 5km del centro)
    df['is_in_downtown'] = (df['distance_to_center'] < 5).astype(int)
    
    print("✅ Features geográficos creados")
    return df

def create_movement_features(df):
    """Crear features de movimiento"""
    print("\n🔄 Creando features de movimiento...")
    
    # Categorías de velocidad
    df['is_stopped'] = (df['speed_calculated'] < 5).astype(int)
    df['is_moving_slow'] = ((df['speed_calculated'] >= 5) & 
                            (df['speed_calculated'] < 30)).astype(int)
    df['is_moving_normal'] = ((df['speed_calculated'] >= 30) & 
                              (df['speed_calculated'] < 60)).astype(int)
    df['is_moving_fast'] = (df['speed_calculated'] >= 60).astype(int)
    
    # Categoría de velocidad
    def speed_category(speed):
        if pd.isna(speed):
            return 'unknown'
        elif speed < 5:
            return 'stopped'
        elif speed < 30:
            return 'slow'
        elif speed < 60:
            return 'normal'
        else:
            return 'fast'
    
    df['speed_category'] = df['speed_calculated'].apply(speed_category)
    
    # Cambio de dirección (heading)
    df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    df['heading_change'] = df.groupby('vehicle_id')['heading'].diff().abs()
    
    print("✅ Features de movimiento creados")
    return df

def create_vehicle_aggregates(df):
    """Crear features agregados por vehículo"""
    print("\n🔄 Creando agregados por vehículo...")
    
    # Agregados por vehículo
    vehicle_stats = df.groupby('vehicle_id').agg({
        'speed_calculated': ['mean', 'std', 'max'],
        'distance_traveled': 'sum',
        'is_stopped': 'sum',
        'timestamp': 'count'
    }).reset_index()
    
    vehicle_stats.columns = [
        'vehicle_id', 
        'avg_speed_vehicle', 
        'std_speed_vehicle', 
        'max_speed_vehicle',
        'total_distance_vehicle',
        'num_stops_vehicle',
        'num_readings_vehicle'
    ]
    
    # Merge con el dataframe original
    df = df.merge(vehicle_stats, on='vehicle_id', how='left')
    
    print("✅ Agregados por vehículo creados")
    return df

# ============================================================================
# LIMPIEZA DE DATOS
# ============================================================================

def remove_duplicates(df):
    """Eliminar duplicados"""
    print("\n🧹 Eliminando duplicados...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['vehicle_id', 'timestamp'], keep='first')
    removed = initial_count - len(df)
    print(f"✅ Duplicados eliminados: {removed:,}")
    return df

def handle_missing_values(df):
    """Manejar valores nulos"""
    print("\n🧹 Manejando valores nulos...")
    
    # Mostrar nulos antes
    print("\nNulos antes:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Estrategias:
    # - route_id, trip_id: llenar con 'unknown'
    # - speed_calculated: ya manejados (NaN para primer registro de cada vehículo)
    # - heading: llenar con forward fill por vehículo
    
    df['route_id'] = df['route_id'].fillna('unknown')
    df['trip_id'] = df['trip_id'].fillna('unknown')
    df['heading'] = df.groupby('vehicle_id')['heading'].fillna(method='ffill')
    
    # Para features calculados, llenar con 0 o valores por defecto
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
    
    print("\nNulos después:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("✅ Valores nulos manejados")
    return df

def filter_outliers(df):
    """Filtrar outliers extremos"""
    print("\n🧹 Filtrando outliers...")
    
    initial_count = len(df)
    
    # Filtrar coordenadas fuera del área de SF Bay
    df = df[
        (df['latitude'] >= 37.0) & (df['latitude'] <= 38.0) &
        (df['longitude'] >= -123.0) & (df['longitude'] <= -121.5)
    ]
    
    # Filtrar velocidades imposibles (ya filtradas a 150 km/h)
    # Filtrar aceleraciones extremas (> 10 m/s²)
    df = df[df['acceleration'].abs() <= 10]
    
    removed = initial_count - len(df)
    print(f"✅ Outliers eliminados: {removed:,}")
    
    return df

# ============================================================================
# TRANSFORMACIONES PARA ML
# ============================================================================

def encode_categorical(df):
    """Encodear variables categóricas"""
    print("\n🔄 Encodeando variables categóricas...")
    
    # Label Encoding para variables con orden
    le_agency = LabelEncoder()
    df['agency_encoded'] = le_agency.fit_transform(df['agency_id'])
    
    # One-hot encoding para variables sin orden (se hará en el modelo)
    # Por ahora solo preparamos las columnas
    
    print("✅ Variables categóricas encodeadas")
    print(f"   Agencias: {dict(zip(le_agency.classes_, le_agency.transform(le_agency.classes_)))}")
    
    return df, le_agency

def normalize_features(df):
    """Normalizar features numéricos"""
    print("\n🔄 Normalizando features...")
    
    # Seleccionar features numéricos para normalizar
    features_to_normalize = [
        'latitude', 'longitude', 'heading',
        'speed_calculated', 'distance_traveled', 'acceleration',
        'distance_to_center', 'heading_change',
        'avg_speed_vehicle', 'std_speed_vehicle', 'max_speed_vehicle',
        'total_distance_vehicle'
    ]
    
    # Crear copia de features originales
    for feat in features_to_normalize:
        if feat in df.columns:
            df[f'{feat}_original'] = df[feat]
    
    # Normalizar
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    print("✅ Features normalizados")
    return df, scaler

def prepare_for_ml(df):
    """Preparar dataset final para ML"""
    print("\n🔄 Preparando dataset para ML...")
    
    # Seleccionar features relevantes
    feature_columns = [
        # Temporales
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        # Geográficos
        'latitude', 'longitude', 'distance_to_center', 'zone',
        # Movimiento
        'speed_calculated', 'acceleration', 'heading', 'heading_change',
        'is_stopped', 'is_moving_slow', 'is_moving_normal', 'is_moving_fast',
        # Agregados
        'avg_speed_vehicle', 'std_speed_vehicle', 'max_speed_vehicle',
        'total_distance_vehicle', 'num_stops_vehicle',
        # Categóricos
        'agency_encoded', 'route_id'
    ]
    
    # Filtrar solo columnas que existen
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    df_ml = df[feature_columns + ['vehicle_id', 'timestamp']].copy()
    
    print(f"✅ Dataset preparado con {len(feature_columns)} features")
    return df_ml

def split_train_test(df, test_size=0.2):
    """Dividir en train y test"""
    print(f"\n🔄 Dividiendo en train ({int((1-test_size)*100)}%) y test ({int(test_size*100)}%)...")
    
    # Dividir por tiempo (más realista para series temporales)
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"✅ Train: {len(train_df):,} registros")
    print(f"✅ Test: {len(test_df):,} registros")
    
    return train_df, test_df

# ============================================================================
# GUARDAR DATASETS
# ============================================================================

def save_datasets(df_original, df_engineered, train_df, test_df):
    """Guardar datasets procesados"""
    print("\n💾 Guardando datasets...")
    
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    # Guardar datos originales con features
    df_original.to_csv('data/processed/data_original.csv', index=False)
    print(f"   ✅ data_original.csv ({len(df_original):,} registros)")
    
    # Guardar datos con feature engineering
    df_engineered.to_csv('data/processed/data_engineered.csv', index=False)
    print(f"   ✅ data_engineered.csv ({len(df_engineered):,} registros)")
    
    # Guardar train y test
    train_df.to_csv('data/processed/train_data.csv', index=False)
    print(f"   ✅ train_data.csv ({len(train_df):,} registros)")
    
    test_df.to_csv('data/processed/test_data.csv', index=False)
    print(f"   ✅ test_data.csv ({len(test_df):,} registros)")
    
    print("\n✅ Todos los datasets guardados en data/processed/")

def generate_report(df_original, df_final):
    """Generar reporte de preprocesamiento"""
    print("\n" + "="*70)
    print("📊 REPORTE DE PREPROCESAMIENTO")
    print("="*70)
    
    print(f"\n📏 Registros:")
    print(f"   • Originales: {len(df_original):,}")
    print(f"   • Finales: {len(df_final):,}")
    print(f"   • Removidos: {len(df_original) - len(df_final):,} ({(len(df_original)-len(df_final))/len(df_original)*100:.1f}%)")
    
    print(f"\n📊 Features:")
    print(f"   • Originales: {len(df_original.columns)}")
    print(f"   • Finales: {len(df_final.columns)}")
    print(f"   • Nuevos: {len(df_final.columns) - len(df_original.columns)}")
    
    print(f"\n📈 Estadísticas de velocidad:")
    print(f"   • Media: {df_final['speed_calculated'].mean():.2f} km/h")
    print(f"   • Mediana: {df_final['speed_calculated'].median():.2f} km/h")
    print(f"   • Desviación: {df_final['speed_calculated'].std():.2f} km/h")
    
    print(f"\n🚦 Distribución de movimiento:")
    print(f"   • Detenidos: {df_final['is_stopped'].sum():,} ({df_final['is_stopped'].sum()/len(df_final)*100:.1f}%)")
    print(f"   • Lento: {df_final['is_moving_slow'].sum():,} ({df_final['is_moving_slow'].sum()/len(df_final)*100:.1f}%)")
    print(f"   • Normal: {df_final['is_moving_normal'].sum():,} ({df_final['is_moving_normal'].sum()/len(df_final)*100:.1f}%)")
    print(f"   • Rápido: {df_final['is_moving_fast'].sum():,} ({df_final['is_moving_fast'].sum()/len(df_final)*100:.1f}%)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🔧 PREPROCESAMIENTO DE DATOS - SF TRANSIT")
    print("="*70)
    
    # 1. Conectar y extraer datos
    print("\n🔌 Conectando a la base de datos...")
    conn = connect_db()
    print("✅ Conexión establecida")
    
    print("\n📥 Extrayendo datos...")
    df = get_vehicle_positions(conn, hours=48)
    conn.close()
    print(f"✅ Datos extraídos: {len(df):,} registros")
    
    df_original = df.copy()
    
    # 2. Feature Engineering
    print("\n" + "="*70)
    print("🛠️  FEATURE ENGINEERING")
    print("="*70)
    
    df = calculate_speed_from_positions(df)
    df = calculate_acceleration(df)
    df = create_temporal_features(df)
    df = create_geographic_features(df)
    df = create_movement_features(df)
    df = create_vehicle_aggregates(df)
    
    # 3. Limpieza
    print("\n" + "="*70)
    print("🧹 LIMPIEZA DE DATOS")
    print("="*70)
    
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = filter_outliers(df)
    
    df_engineered = df.copy()
    
    # 4. Transformaciones para ML
    print("\n" + "="*70)
    print("🤖 TRANSFORMACIONES PARA ML")
    print("="*70)
    
    df, le_agency = encode_categorical(df)
    # df, scaler = normalize_features(df)  # Comentado: normalizar en el modelo
    df_ml = prepare_for_ml(df)
    train_df, test_df = split_train_test(df_ml, test_size=0.2)
    
    # 5. Guardar datasets
    save_datasets(df_original, df_engineered, train_df, test_df)
    
    # 6. Generar reporte
    generate_report(df_original, df_engineered)
    
    print("\n" + "="*70)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   • data/processed/data_original.csv")
    print("   • data/processed/data_engineered.csv")
    print("   • data/processed/train_data.csv")
    print("   • data/processed/test_data.csv")
    print("\n🚀 Listo para entrenar modelos de Machine Learning!")

if __name__ == "__main__":
    main()
