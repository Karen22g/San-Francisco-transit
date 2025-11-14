"""
DASHBOARD AVANZADO - SF TRANSIT
Dashboard con ETA, Alertas y Análisis Avanzado
Autor: Francisco Narvaez M
Fecha: 2025-11-12
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pg8000
import joblib
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="SF Transit Advanced Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_CONFIG = {
    'host': 'bigdataanalytics.postgres.database.azure.com',
    'database': 'transit_streaming',
    'user': 'nuevo_usuario',
    'password': 'user',
    'port': 5432
}

SF_CENTER_LAT = 37.7749
SF_CENTER_LON = -122.4194

# Umbrales
SPEED_THRESHOLD_STOPPED = 5
SPEED_THRESHOLD_SLOW = 10
ANOMALY_THRESHOLD = 15

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
def get_realtime_data(hours=0.5):
    """Obtener datos en tiempo real"""
    try:
        conn = pg8000.connect(**DB_CONFIG)
        query = f"""
            SELECT 
                vehicle_id,
                route_id,
                agency_id,
                latitude,
                longitude,
                heading,
                created_at
            FROM vehicle_positions
            WHERE created_at > NOW() - INTERVAL '{hours} hours'
            ORDER BY created_at DESC
        """
        interval_str = f"{hours} hours"
        df = pd.read_sql(query, conn, params=[interval_str])
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al conectar: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_vehicle_history(vehicle_id, hours=1):
    """Obtener historial de un vehículo"""
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
                created_at
            FROM vehicle_positions
            WHERE vehicle_id = %s
              AND created_at > NOW() - INTERVAL '{hours} hours'
            ORDER BY created_at DESC
        """
        df = pd.read_sql(query, conn, params=(vehicle_id,))
        conn.close()

        df.to_csv("vehicle_history.csv", index=False)
        return df
    except:
        return pd.DataFrame()

def calculate_features(df):
    """Calcular features para predicción"""
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    df['distance_to_center'] = df.apply(
        lambda row: haversine(SF_CENTER_LAT, SF_CENTER_LON, 
                             row['latitude'], row['longitude']),
        axis=1
    )
    return df

def predict_speed(df, model, scaler):
    """Predecir velocidad"""
    if model is None:
       #print("warning: Usando predicciones por defecto")
       df['predicted_speed'] = np.random.uniform(10, 25, len(df))
       return df
    #print("info: Usando modelo para predicciones")

    print(df.columns.tolist())
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'latitude', 'longitude', 'distance_to_center', 'heading'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    #print("Tipo de modelo:", type(model))
    print("Columnas esperadas por el modelo:", model.feature_names_in_)
    print("Columnas usadas ahora:", X.columns.tolist()) 

    try:
        predictions = model.predict(X)
        #print("predicción generada es", predictions)
        df['predicted_speed'] = predictions
    except Exception as e:
        print("❌ Error en la predicción:", e)
        #print("predicción generada es", 20)
        df['predicted_speed'] = 20
    
    return df

def calculate_actual_speed(df):
    """Calcular velocidad real desde GPS"""
    if len(df) < 2:
        return None
    
    df = df.sort_values('created_at')
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    distance = haversine(
        prev['latitude'], prev['longitude'],
        last['latitude'], last['longitude']
    )
    
    time_diff = (last['created_at'] - prev['created_at']).total_seconds()
    
    if time_diff > 0:
        speed = (distance / time_diff) * 3600
        return speed if speed <= 150 else None
    
    return None

def detect_alerts(df):
    """Detectar alertas en los datos"""
    df = df.copy()
    df['alert_level'] = 'normal'
    df['alert_message'] = ''
    
    # Vehículos detenidos
    stopped_mask = df['predicted_speed'] < SPEED_THRESHOLD_STOPPED
    df.loc[stopped_mask, 'alert_level'] = 'high'
    df.loc[stopped_mask, 'alert_message'] = 'Vehículo detenido'
    
    # Tráfico lento
    slow_mask = (df['predicted_speed'] >= SPEED_THRESHOLD_STOPPED) & \
                (df['predicted_speed'] < SPEED_THRESHOLD_SLOW)
    df.loc[slow_mask, 'alert_level'] = 'medium'
    df.loc[slow_mask, 'alert_message'] = 'Tráfico lento'
    
    # Hora pico
    rush_mask = df['is_rush_hour'] == 1
    df.loc[rush_mask & (df['alert_level'] == 'normal'), 'alert_level'] = 'low'
    df.loc[rush_mask & (df['alert_message'] == ''), 'alert_message'] = 'Hora pico'
    
    return df

# ============================================================================
# COMPONENTES DEL DASHBOARD
# ============================================================================

def render_header():
    """Header del dashboard"""
    st.title("🚀 SF Transit Advanced Dashboard")
    st.markdown("**Monitoreo en Tiempo Real | Predicción de ETA | Sistema de Alertas**")

    st.markdown("---")

def render_alerts_panel(df):
    """Panel de alertas"""
    st.subheader("🚨 Panel de Alertas en Tiempo Real")
    
    alerts_df = df[df['alert_level'] != 'normal'].copy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_alerts = len(alerts_df[alerts_df['alert_level'] == 'high'])
        st.metric(
            label="🔴 Alertas Críticas",
            value=high_alerts,
            delta=f"{(high_alerts/len(df)*100):.1f}%" if len(df) > 0 else "0%"
        )
    
    with col2:
        medium_alerts = len(alerts_df[alerts_df['alert_level'] == 'medium'])
        st.metric(
            label="🟡 Alertas Medias",
            value=medium_alerts,
            delta=f"{(medium_alerts/len(df)*100):.1f}%" if len(df) > 0 else "0%"
        )
    
    with col3:
        low_alerts = len(alerts_df[alerts_df['alert_level'] == 'normal'])
        st.metric(
            label="🟢 Alertas Bajas",
            value=low_alerts,
            delta=f"{(low_alerts/len(df)*100):.1f}%" if len(df) > 0 else "0%"
        )
    
    # Tabla de alertas
    if len(alerts_df) > 0:
        with st.expander("📋 Ver todas las alertas", expanded=False):
            display_alerts = alerts_df[['vehicle_id', 'route_id', 'agency_id', 
                                       'alert_level', 'alert_message', 'predicted_speed']].head(20)
            st.dataframe(display_alerts, use_container_width=True)

def render_eta_calculator(df, model, scaler):
    """Calculadora de ETA"""
    st.subheader("⏱️ Calculadora de Tiempo de Llegada (ETA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de vehículo
        vehicles = df['vehicle_id'].unique().tolist()
        selected_vehicle = st.selectbox(
            "Seleccionar Vehículo",
            vehicles,
            key='eta_vehicle'
        )
        
        # Obtener posición actual
        vehicle_data = df[df['vehicle_id'] == selected_vehicle].iloc[0]
        
        st.info(f"""
        **Información del Vehículo:**
        - ID: {vehicle_data['vehicle_id']}
        - Ruta: {vehicle_data['route_id']}
        - Agencia: {vehicle_data['agency_id']}
        - Posición: {vehicle_data['latitude']:.4f}, {vehicle_data['longitude']:.4f}
        """)
    
    with col2:
        st.markdown("**Destino:**")
        
        # Opciones predefinidas
        destinations = {
            "Downtown SF (Market St)": (37.7749, -122.4194),
            "Ferry Building": (37.7955, -122.3937),
            "Golden Gate Park": (37.7694, -122.4862),
            "Mission District": (37.7599, -122.4148),
            "Personalizado": None
        }
        
        dest_choice = st.selectbox("Seleccionar destino:", list(destinations.keys()))
        
        if dest_choice == "Personalizado":
            dest_lat = st.number_input("Latitud destino:", value=37.7749, format="%.4f")
            dest_lon = st.number_input("Longitud destino:", value=-122.4194, format="%.4f")
        else:
            dest_lat, dest_lon = destinations[dest_choice]
            st.write(f"Coordenadas: {dest_lat}, {dest_lon}")
    
    if st.button("🔮 Calcular ETA", type="primary"):
        # Calcular distancia
        distance = haversine(
            vehicle_data['latitude'], vehicle_data['longitude'],
            dest_lat, dest_lon
        )
        
        # Predecir velocidad
        predicted_speed = vehicle_data['predicted_speed']
        
        # Calcular ETA
        if predicted_speed > 0:
            eta_minutes = (distance / predicted_speed) * 60
            arrival_time = datetime.now() + timedelta(minutes=eta_minutes)
        else:
            eta_minutes = float('inf')
            arrival_time = None
        
        # Mostrar resultados
        st.success("✅ ETA Calculado")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("📍 Distancia", f"{distance:.2f} km")
        
        with col_b:
            st.metric("⚡ Velocidad Est.", f"{predicted_speed:.1f} km/h")
        
        with col_c:
            st.metric("⏱️ Tiempo Est.", f"{eta_minutes:.0f} min")
        
        if arrival_time:
            st.info(f"🕐 **Hora estimada de llegada:** {arrival_time.strftime('%H:%M:%S')}")

def render_route_comparator(df):
    """Comparador de rutas"""
    st.subheader("🛣️ Comparador de Rutas")
    
    routes = df[df['route_id'].notna()]['route_id'].unique().tolist()
    
    if len(routes) < 2:
        st.warning("Se necesitan al menos 2 rutas activas para comparar")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        route1 = st.selectbox("Ruta 1:", routes, key='route1')
    
    with col2:
        route2 = st.selectbox("Ruta 2:", [r for r in routes if r != route1], key='route2')
    
    if st.button("📊 Comparar Rutas"):
        df1 = df[df['route_id'] == route1]
        df2 = df[df['route_id'] == route2]
        
        stats1 = {
            'vehicles': len(df1),
            'avg_speed': df1['predicted_speed'].mean(),
            'min_speed': df1['predicted_speed'].min(),
            'max_speed': df1['predicted_speed'].max()
        }
        
        stats2 = {
            'vehicles': len(df2),
            'avg_speed': df2['predicted_speed'].mean(),
            'min_speed': df2['predicted_speed'].min(),
            'max_speed': df2['predicted_speed'].max()
        }
        
        # Mostrar comparación
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"### Ruta {route1}")
            st.metric("Vehículos Activos", stats1['vehicles'])
            st.metric("Velocidad Promedio", f"{stats1['avg_speed']:.1f} km/h")
            st.metric("Rango de Velocidad", f"{stats1['min_speed']:.1f} - {stats1['max_speed']:.1f} km/h")
        
        with col_b:
            st.markdown(f"### Ruta {route2}")
            st.metric("Vehículos Activos", stats2['vehicles'])
            st.metric("Velocidad Promedio", f"{stats2['avg_speed']:.1f} km/h")
            st.metric("Rango de Velocidad", f"{stats2['min_speed']:.1f} - {stats2['max_speed']:.1f} km/h")
        
        # Recomendación
        if stats1['avg_speed'] > stats2['avg_speed']:
            faster = route1
            diff = stats1['avg_speed'] - stats2['avg_speed']
        else:
            faster = route2
            diff = stats2['avg_speed'] - stats1['avg_speed']
        
        st.success(f"✅ **Recomendación:** Ruta {faster} es {diff:.1f} km/h más rápida")

def render_map_with_alerts(df):
    """Mapa con alertas"""
    st.subheader("🗺️ Mapa con Sistema de Alertas")
    
    # Tomar último registro de cada vehículo
    df_latest = df.sort_values('created_at').groupby('vehicle_id').tail(1)
    
    if len(df_latest) == 0:
        st.warning("No hay datos para mostrar")
        return
    
    # Color según nivel de alerta
    color_map = {
        'high': 'red',
        'medium': 'orange',
        'low': 'yellow',
        'normal': 'green'
    }
    
    df_latest['color'] = df_latest['alert_level'].map(color_map)
    
    # Crear mapa
    fig = px.scatter_mapbox(
        df_latest,
        lat='latitude',
        lon='longitude',
        color='alert_level',
        size='predicted_speed',
        hover_data=['vehicle_id', 'route_id', 'agency_id', 'predicted_speed', 'alert_message'],
        color_discrete_map={'high': 'red', 'medium': 'orange', 'low': 'yellow', 'normal': 'green'},
        size_max=15,
        zoom=11,
        center={'lat': SF_CENTER_LAT, 'lon': SF_CENTER_LON},
        mapbox_style='carto-positron',
        height=600
    )
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    st.plotly_chart(fig, use_container_width=True)

def render_vehicle_tracker(df):
    """Rastreador de vehículo individual"""
    st.subheader("🔍 Rastreador de Vehículo")

    st.write("Variables consideradas: Hora del día, Día de la semana, Coordenadas, Distancia al centro de la ciudad, Dirección")

    vehicles = df['vehicle_id'].unique().tolist()
    selected_vehicle = st.selectbox("Seleccionar vehículo para rastrear:", vehicles)
    
    if st.button("📡 Rastrear"):
        history_df = get_vehicle_history(selected_vehicle, hours=1)
        
        if history_df.empty:
            st.warning("No hay historial disponible")
            return
        
        history_df = calculate_features(history_df)
        model, scaler = load_model()
        history_df = predict_speed(history_df, model, scaler)
        
        # Información actual
        current = history_df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ruta", current['route_id'])
        
        with col2:
            st.metric("Agencia", current['agency_id'])
        
        with col3:
            st.metric("Velocidad Est.", f"{current['predicted_speed']:.1f} km/h")
        
        with col4:
            actual_speed = calculate_actual_speed(history_df)
            if actual_speed:
                st.metric("Velocidad Real", f"{actual_speed:.1f} km/h")
            else:
                st.metric("Velocidad Real", "N/A")
        
        history_df['created_at'] = pd.to_datetime(history_df['created_at']) - pd.Timedelta(hours=5)

        # Gráfico de trayectoria
        if len(history_df) > 1:
            fig = px.line_mapbox(
                history_df.sort_values('created_at'),
                lat='latitude',
                lon='longitude',
                hover_data=['created_at', 'predicted_speed'],
                zoom=13,
                height=600
            )
            fig.update_layout(mapbox_style='carto-positron')
            st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de velocidad en el tiempo
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=history_df.sort_values('created_at')['created_at'],
            y=history_df.sort_values('created_at')['predicted_speed'],
            mode='lines+markers',
            name='Velocidad Predicha'
        ))
        fig2.update_layout(
            title='Velocidad en el Tiempo',
            xaxis_title='Tiempo',
            yaxis_title='Velocidad (km/h)',
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

def render_map_with_kde(df):
    """Mapa con Detección de Puntos Calientes de Tráfico (KDE)"""
    st.subheader("🗺️ Mapa con Detección de Puntos Calientes (KDE)")

    # Tomar último registro de cada vehículo
    df_latest = df.sort_values('created_at').groupby('vehicle_id').tail(1)

    if len(df_latest) == 0:
        st.warning("No hay datos para mostrar")
        return

    df_latest["inv_speed"] = df_latest["predicted_speed"] * -1
    # Crear mapa con densidad KDE
    
    fig = px.density_mapbox(
        df_latest,
        lat="latitude",
        lon="longitude",
        z="inv_speed",  # se usa como peso
        radius=15,  # controla el suavizado del KDE
        hover_data=["vehicle_id", "route_id", "agency_id", "predicted_speed", "alert_message"],
        center={"lat": SF_CENTER_LAT, "lon": SF_CENTER_LON},
        zoom=11,
        mapbox_style="carto-positron",
        height=600,
        color_continuous_scale="YlOrRd",
        title="Mapa de densidad de tráfico (KDE)"
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)
# ============================================================================
# MAIN
# ============================================================================

def main():
    render_header()
    
    # Cargar modelo
    model, scaler = load_model()
    
    if model is None:
        st.warning("⚠️ Modelo no encontrado. Usando estimaciones por defecto.")
    
    # Cargar datos
    with st.spinner("📡 Cargando datos..."):
        df = get_realtime_data(hours=0.5)
        print(df)
        
        if df.empty:
            st.error("❌ No hay datos disponibles.")
            return
        
        df = calculate_features(df)
        df = predict_speed(df, model, scaler)
        df = detect_alerts(df)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Monitoreo General",
        "⏱️ Calculadora ETA",
        "🛣️ Comparador de Rutas",
        "🔍 Rastreo de Vehículo"
    ])
    
    with tab1:
        render_alerts_panel(df)
        st.markdown("---")
        render_map_with_alerts(df)
        st.markdown("---")
        render_map_with_kde(df)
    
    with tab2:
        render_eta_calculator(df, model, scaler)
    
    with tab3:
        render_route_comparator(df)
    
    with tab4:
        render_vehicle_tracker(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**SF Transit Advanced Dashboard** | "
        f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Datos: {len(df):,} registros | "
        f"Alertas activas: {len(df[df['alert_level'] != 'normal']):,}"
    )

if __name__ == "__main__":
    main()
