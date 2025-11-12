"""
DASHBOARD INTERACTIVO - SF TRANSIT
Dashboard con Streamlit para visualizar predicciones en tiempo real
Autor: Francisco Narvaez M
Fecha: 2025-11-12
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import joblib
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="SF Transit Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'transit_streaming',
    'user': 'pachonarvaez',
    'port': 5432
}

SF_CENTER_LAT = 37.7749
SF_CENTER_LON = -122.4194

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def load_model():
    """Cargar modelo entrenado"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calcular distancia en km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

@st.cache_data(ttl=30)
def get_realtime_data(hours=1):
    """Obtener datos en tiempo real"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"""
        SELECT 
            vehicle_id,
            route_id,
            agency_id,
            latitude,
            longitude,
            heading,
            timestamp
        FROM vehicle_positions
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calculate_speed_features(df):
    """Calcular features de velocidad para predicción"""
    df = df.copy()
    
    # Temporales
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    
    # Geográficos
    df['distance_to_center'] = df.apply(
        lambda row: haversine(SF_CENTER_LAT, SF_CENTER_LON, 
                             row['latitude'], row['longitude']),
        axis=1
    )
    
    return df

def predict_speed(df, model, scaler):
    """Predecir velocidad para los datos"""
    if model is None:
        df['predicted_speed'] = np.random.uniform(5, 30, len(df))
        return df
    
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'latitude', 'longitude', 'distance_to_center', 'heading'
    ]
    
    # Asegurarse de que todas las columnas existen
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    try:
        predictions = model.predict(X)
        df['predicted_speed'] = predictions
    except:
        df['predicted_speed'] = np.random.uniform(5, 30, len(df))
    
    return df

# ============================================================================
# COMPONENTES DEL DASHBOARD
# ============================================================================

def render_header():
    """Renderizar header del dashboard"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🚌 SF Transit Dashboard")
        st.markdown("**Monitoreo y Predicción en Tiempo Real**")
    
    st.markdown("---")

def render_metrics(df):
    """Renderizar métricas principales"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚌 Vehículos Activos",
            value=f"{df['vehicle_id'].nunique():,}"
        )
    
    with col2:
        avg_speed = df['predicted_speed'].mean()
        st.metric(
            label="⚡ Velocidad Promedio",
            value=f"{avg_speed:.1f} km/h"
        )
    
    with col3:
        st.metric(
            label="🛣️ Rutas Activas",
            value=f"{df['route_id'].nunique():,}"
        )
    
    with col4:
        st.metric(
            label="🏢 Agencias",
            value=f"{df['agency_id'].nunique()}"
        )

def render_map(df):
    """Renderizar mapa interactivo"""
    st.subheader("📍 Mapa en Tiempo Real")
    
    # Tomar

