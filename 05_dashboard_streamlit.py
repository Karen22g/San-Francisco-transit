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
    try:
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
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return pd.DataFrame()

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
            value=f"{df['route_id'].notna().sum():,}"
        )
    
    with col4:
        st.metric(
            label="🏢 Agencias",
            value=f"{df['agency_id'].nunique()}"
        )

def render_map(df):
    """Renderizar mapa interactivo"""
    st.subheader("📍 Mapa en Tiempo Real")
    
    # Tomar último registro de cada vehículo
    df_latest = df.sort_values('timestamp').groupby('vehicle_id').tail(1)
    
    if len(df_latest) == 0:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    # Crear mapa
    fig = px.scatter_mapbox(
        df_latest,
        lat='latitude',
        lon='longitude',
        color='predicted_speed',
        size='predicted_speed',
        hover_data=['vehicle_id', 'route_id', 'agency_id', 'predicted_speed'],
        color_continuous_scale='RdYlGn_r',
        size_max=15,
        zoom=11,
        center={'lat': SF_CENTER_LAT, 'lon': SF_CENTER_LON},
        mapbox_style='open-street-map',
        height=600
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Velocidad<br>(km/h)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_speed_distribution(df):
    """Gráfico de distribución de velocidades"""
    st.subheader("📊 Distribución de Velocidades")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['predicted_speed'],
        nbinsx=50,
        name='Velocidad Predicha',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        xaxis_title="Velocidad (km/h)",
        yaxis_title="Frecuencia",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_speed_by_agency(df):
    """Velocidad por agencia"""
    st.subheader("🏢 Velocidad por Agencia")
    
    agency_stats = df.groupby('agency_id')['predicted_speed'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=agency_stats['agency_id'],
        y=agency_stats['mean'],
        error_y=dict(type='data', array=agency_stats['std']),
        marker_color='teal'
    ))
    
    fig.update_layout(
        xaxis_title="Agencia",
        yaxis_title="Velocidad Promedio (km/h)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_speed_by_hour(df):
    """Velocidad por hora del día"""
    st.subheader("⏰ Velocidad por Hora del Día")
    
    hourly = df.groupby('hour')['predicted_speed'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['predicted_speed'],
        mode='lines+markers',
        marker=dict(size=8, color='orange'),
        line=dict(width=3)
    ))
    
    fig.update_layout(
        xaxis_title="Hora del Día",
        yaxis_title="Velocidad Promedio (km/h)",
        height=400,
        xaxis=dict(tickmode='linear')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_route_comparison(df):
    """Comparación de rutas"""
    st.subheader("🛣️ Comparación de Rutas (Top 10)")
    
    # Filtrar solo rutas válidas
    df_valid = df[df['route_id'].notna()].copy()
    
    if len(df_valid) == 0:
        st.warning("No hay rutas disponibles para comparar")
        return
    
    route_stats = df_valid.groupby('route_id').agg({
        'predicted_speed': ['mean', 'count'],
        'vehicle_id': 'nunique'
    }).reset_index()
    
    route_stats.columns = ['route_id', 'avg_speed', 'readings', 'vehicles']
    route_stats = route_stats.sort_values('readings', ascending=False).head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=route_stats['route_id'].astype(str),
        y=route_stats['avg_speed'],
        marker_color='lightblue',
        text=route_stats['vehicles'],
        texttemplate='%{text} veh',
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title="Ruta",
        yaxis_title="Velocidad Promedio (km/h)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_heatmap_by_zone(df):
    """Mapa de calor por zona"""
    st.subheader("🔥 Mapa de Calor - Velocidad por Zona")
    
    if len(df) == 0:
        st.warning("No hay datos para mostrar en el mapa de calor")
        return
    
    fig = px.density_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        z='predicted_speed',
        radius=10,
        center={'lat': SF_CENTER_LAT, 'lon': SF_CENTER_LON},
        zoom=11,
        mapbox_style='open-street-map',
        color_continuous_scale='RdYlGn_r',
        height=500
    )
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    st.plotly_chart(fig, use_container_width=True)

def render_sidebar_filters(df):
    """Sidebar con filtros"""
    st.sidebar.header("🔧 Filtros")
    
    # Limpiar datos nulos en route_id antes de filtrar
    df = df.copy()
    
    # Filtro de agencia
    agencies = ['Todas'] + sorted(df['agency_id'].dropna().unique().tolist())
    selected_agency = st.sidebar.selectbox("Agencia", agencies)
    
    # Filtro de ruta (manejar valores nulos)
    if selected_agency != 'Todas':
        # Filtrar por agencia primero
        routes_list = df[df['agency_id'] == selected_agency]['route_id'].dropna().unique().tolist()
    else:
        routes_list = df['route_id'].dropna().unique().tolist()
    
    # Ordenar y agregar 'Todas'
    routes = ['Todas'] + sorted([str(r) for r in routes_list if r])
    selected_route = st.sidebar.selectbox("Ruta", routes)
    
    # Filtro de velocidad
    max_speed = int(df['predicted_speed'].max()) if df['predicted_speed'].max() > 0 else 100
    speed_range = st.sidebar.slider(
        "Rango de Velocidad (km/h)",
        min_value=0,
        max_value=max_speed,
        value=(0, max_speed)
    )
    
    # Aplicar filtros
    filtered_df = df.copy()
    
    if selected_agency != 'Todas':
        filtered_df = filtered_df[filtered_df['agency_id'] == selected_agency]
    
    if selected_route != 'Todas':
        filtered_df = filtered_df[filtered_df['route_id'].astype(str) == selected_route]
    
    filtered_df = filtered_df[
        (filtered_df['predicted_speed'] >= speed_range[0]) &
        (filtered_df['predicted_speed'] <= speed_range[1])
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Registros filtrados:** {len(filtered_df):,}")
    
    # Botón para refrescar manualmente
    if st.sidebar.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    return filtered_df

def render_data_table(df):
    """Tabla de datos"""
    with st.expander("📋 Ver Datos Detallados"):
        display_df = df[['vehicle_id', 'route_id', 'agency_id', 'predicted_speed', 
                         'latitude', 'longitude', 'timestamp']].head(100)
        st.dataframe(display_df, use_container_width=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Header
    render_header()
    
    # Cargar modelo
    model, scaler = load_model()
    
    if model is None:
        st.warning("⚠️ Modelo no encontrado. Usando predicciones aleatorias para demostración.")
    
    # Cargar datos
    with st.spinner("📡 Cargando datos en tiempo real..."):
        df = get_realtime_data(hours=2)
        
        if df.empty:
            st.error("❌ No hay datos disponibles. Ejecuta 01_data_ingestion_511.py primero.")
            st.info("💡 Tip: Ejecuta `python 01_data_ingestion_511.py` en otra terminal para comenzar a recolectar datos.")
            return
        
        # Calcular features y predecir
        df = calculate_speed_features(df)
        df = predict_speed(df, model, scaler)
    
    # Sidebar con filtros
    filtered_df = render_sidebar_filters(df)
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        return
    
    # Métricas principales
    render_metrics(filtered_df)
    
    st.markdown("---")
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_map(filtered_df)
    
    with col2:
        render_speed_distribution(filtered_df)
        render_speed_by_agency(filtered_df)
    
    st.markdown("---")
    
    # Gráficos adicionales
    col3, col4 = st.columns(2)
    
    with col3:
        render_speed_by_hour(filtered_df)
    
    with col4:
        render_route_comparison(filtered_df)
    
    st.markdown("---")
    
    # Mapa de calor
    render_heatmap_by_zone(filtered_df)
    
    st.markdown("---")
    
    # Tabla de datos
    render_data_table(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**SF Transit Dashboard** | "
        f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Datos: {len(df):,} registros"
    )

if __name__ == "__main__":
    main()
