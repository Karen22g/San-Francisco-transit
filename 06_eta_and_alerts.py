"""
SISTEMA DE PREDICCIÓN ETA Y ALERTAS - SF TRANSIT
Predicción de tiempos de llegada y detección de anomalías
Autor: Francisco Narvaez M
Fecha: 2025-11-12
"""

import psycopg2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'database': 'transit_streaming',
    'user': 'pachonarvaez',
    'port': 5432
}

SF_CENTER_LAT = 37.7749
SF_CENTER_LON = -122.4194

# Umbrales para alertas
SPEED_THRESHOLD_STOPPED = 5      # km/h - vehículo detenido
SPEED_THRESHOLD_SLOW = 10        # km/h - tráfico muy lento
ANOMALY_THRESHOLD = 15           # km/h - diferencia para anomalía

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

def load_model():
    """Cargar modelo entrenado"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        print("⚠️  Modelo no encontrado. Usando estimaciones por defecto.")
        return None, None

# ============================================================================
# OBTENCIÓN DE DATOS
# ============================================================================

def get_vehicle_current_position(vehicle_id):
    """Obtener posición actual de un vehículo"""
    conn = connect_db()
    query = """
        SELECT 
            vehicle_id,
            route_id,
            agency_id,
            latitude,
            longitude,
            heading,
            timestamp
        FROM vehicle_positions
        WHERE vehicle_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """
    df = pd.read_sql(query, conn, params=(vehicle_id,))
    conn.close()
    return df

def get_vehicle_history(vehicle_id, hours=1):
    """Obtener historial de un vehículo"""
    conn = connect_db()
    query = """
        SELECT 
            vehicle_id,
            route_id,
            agency_id,
            latitude,
            longitude,
            heading,
            timestamp
        FROM vehicle_positions
        WHERE vehicle_id = %s
          AND timestamp > NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn, params=(vehicle_id, hours))
    conn.close()
    return df

def get_route_vehicles(route_id):
    """Obtener todos los vehículos de una ruta"""
    conn = connect_db()
    query = """
        SELECT DISTINCT ON (vehicle_id)
            vehicle_id,
            route_id,
            agency_id,
            latitude,
            longitude,
            heading,
            timestamp
        FROM vehicle_positions
        WHERE route_id = %s
          AND timestamp > NOW() - INTERVAL '30 minutes'
        ORDER BY vehicle_id, timestamp DESC
    """
    df = pd.read_sql(query, conn, params=(route_id,))
    conn.close()
    return df

def get_all_active_vehicles():
    """Obtener todos los vehículos activos"""
    conn = connect_db()
    query = """
        SELECT DISTINCT ON (vehicle_id)
            vehicle_id,
            route_id,
            agency_id,
            latitude,
            longitude,
            heading,
            timestamp
        FROM vehicle_positions
        WHERE timestamp > NOW() - INTERVAL '10 minutes'
        ORDER BY vehicle_id, timestamp DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ============================================================================
# PREDICCIÓN DE VELOCIDAD
# ============================================================================

def calculate_features(df):
    """Calcular features para predicción"""
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
    """Predecir velocidad con el modelo ML"""
    if model is None:
        # Estimación por defecto basada en hora
        df['predicted_speed'] = df['hour'].apply(
            lambda h: 15 if (7 <= h <= 9) or (17 <= h <= 19) else 25
        )
        return df
    
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'latitude', 'longitude', 'distance_to_center', 'heading'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    try:
        predictions = model.predict(X)
        df['predicted_speed'] = predictions
    except:
        df['predicted_speed'] = 20  # Valor por defecto
    
    return df

def calculate_actual_speed(df):
    """Calcular velocidad real desde posiciones GPS"""
    if len(df) < 2:
        return None
    
    df = df.sort_values('timestamp')
    
    # Tomar las dos últimas posiciones
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    distance = haversine(
        prev['latitude'], prev['longitude'],
        last['latitude'], last['longitude']
    )
    
    time_diff = (last['timestamp'] - prev['timestamp']).total_seconds()
    
    if time_diff > 0:
        speed = (distance / time_diff) * 3600  # km/h
        return speed if speed <= 150 else None
    
    return None

# ============================================================================
# PREDICCIÓN DE ETA
# ============================================================================

def predict_eta(vehicle_id, destination_lat, destination_lon):
    """
    Predecir tiempo de llegada estimado (ETA)
    
    Args:
        vehicle_id: ID del vehículo
        destination_lat: Latitud del destino
        destination_lon: Longitud del destino
    
    Returns:
        dict con información de ETA
    """
    print(f"\n🔮 Prediciendo ETA para vehículo {vehicle_id}...")
    
    # Obtener posición actual
    current_df = get_vehicle_current_position(vehicle_id)
    
    if current_df.empty:
        return {
            'status': 'error',
            'message': f'Vehículo {vehicle_id} no encontrado'
        }
    
    current = current_df.iloc[0]
    
    # Cargar modelo
    model, scaler = load_model()
    
    # Calcular features y predecir velocidad
    current_df = calculate_features(current_df)
    current_df = predict_speed(current_df, model, scaler)
    
    predicted_speed = current_df.iloc[0]['predicted_speed']
    
    # Calcular distancia al destino
    distance = haversine(
        current['latitude'], current['longitude'],
        destination_lat, destination_lon
    )
    
    # Calcular ETA
    if predicted_speed > 0:
        eta_hours = distance / predicted_speed
        eta_minutes = eta_hours * 60
    else:
        eta_minutes = float('inf')
    
    # Calcular hora de llegada
    arrival_time = datetime.now() + timedelta(minutes=eta_minutes)
    
    result = {
        'status': 'success',
        'vehicle_id': vehicle_id,
        'route_id': current['route_id'],
        'agency_id': current['agency_id'],
        'current_position': {
            'lat': current['latitude'],
            'lon': current['longitude']
        },
        'destination': {
            'lat': destination_lat,
            'lon': destination_lon
        },
        'distance_km': round(distance, 2),
        'predicted_speed_kmh': round(predicted_speed, 1),
        'eta_minutes': round(eta_minutes, 1),
        'arrival_time': arrival_time.strftime('%H:%M:%S'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Mostrar resultado
    print(f"\n✅ ETA calculado:")
    print(f"   📍 Distancia: {result['distance_km']} km")
    print(f"   ⚡ Velocidad estimada: {result['predicted_speed_kmh']} km/h")
    print(f"   ⏱️  Tiempo estimado: {result['eta_minutes']:.0f} minutos")
    print(f"   🕐 Hora de llegada: {result['arrival_time']}")
    
    return result

# ============================================================================
# SISTEMA DE ALERTAS
# ============================================================================

def detect_anomalies(vehicle_id):
    """
    Detectar anomalías comparando velocidad predicha vs real
    
    Args:
        vehicle_id: ID del vehículo
    
    Returns:
        dict con alertas detectadas
    """
    print(f"\n🚨 Analizando vehículo {vehicle_id}...")
    
    # Obtener historial
    history_df = get_vehicle_history(vehicle_id, hours=1)
    
    if len(history_df) < 2:
        return {
            'status': 'insufficient_data',
            'message': 'No hay suficientes datos históricos'
        }
    
    # Cargar modelo
    model, scaler = load_model()
    
    # Calcular features y predecir
    history_df = calculate_features(history_df)
    history_df = predict_speed(history_df, model, scaler)
    
    # Calcular velocidad real
    actual_speed = calculate_actual_speed(history_df)
    predicted_speed = history_df.iloc[-1]['predicted_speed']
    
    current = history_df.iloc[-1]
    
    alerts = []
    severity = 'normal'
    
    # Detectar vehículo detenido
    if actual_speed is not None and actual_speed < SPEED_THRESHOLD_STOPPED:
        alerts.append({
            'type': 'stopped',
            'severity': 'high',
            'message': f'Vehículo detenido (velocidad: {actual_speed:.1f} km/h)'
        })
        severity = 'high'
    
    # Detectar tráfico lento
    elif actual_speed is not None and actual_speed < SPEED_THRESHOLD_SLOW:
        alerts.append({
            'type': 'slow_traffic',
            'severity': 'medium',
            'message': f'Tráfico lento (velocidad: {actual_speed:.1f} km/h)'
        })
        severity = 'medium'
    
    # Detectar anomalía (diferencia entre predicho y real)
    if actual_speed is not None:
        speed_diff = abs(predicted_speed - actual_speed)
        if speed_diff > ANOMALY_THRESHOLD:
            alerts.append({
                'type': 'anomaly',
                'severity': 'medium',
                'message': f'Anomalía detectada: esperado {predicted_speed:.1f} km/h, real {actual_speed:.1f} km/h'
            })
            if severity == 'normal':
                severity = 'medium'
    
    # Detectar hora pico
    if current['is_rush_hour']:
        alerts.append({
            'type': 'rush_hour',
            'severity': 'low',
            'message': 'Hora pico - espere demoras'
        })
    
    result = {
        'status': 'analyzed',
        'vehicle_id': vehicle_id,
        'route_id': current['route_id'],
        'agency_id': current['agency_id'],
        'severity': severity,
        'actual_speed': round(actual_speed, 1) if actual_speed else None,
        'predicted_speed': round(predicted_speed, 1),
        'alerts': alerts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Mostrar alertas
    if alerts:
        print(f"\n⚠️  {len(alerts)} alerta(s) detectada(s):")
        for alert in alerts:
            emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            print(f"   {emoji.get(alert['severity'], '⚪')} {alert['message']}")
    else:
        print("\n✅ Sin alertas - vehículo operando normalmente")
    
    return result

def monitor_all_vehicles():
    """Monitorear todos los vehículos activos"""
    print("\n" + "="*70)
    print("🔍 MONITOREO DE TODOS LOS VEHÍCULOS")
    print("="*70)
    
    df = get_all_active_vehicles()
    
    if df.empty:
        print("\n❌ No hay vehículos activos")
        return
    
    print(f"\n📊 Analizando {len(df)} vehículos activos...")
    
    # Cargar modelo
    model, scaler = load_model()
    
    # Calcular features y predecir
    df = calculate_features(df)
    df = predict_speed(df, model, scaler)
    
    # Clasificar por velocidad
    stopped = df[df['predicted_speed'] < SPEED_THRESHOLD_STOPPED]
    slow = df[(df['predicted_speed'] >= SPEED_THRESHOLD_STOPPED) & 
              (df['predicted_speed'] < SPEED_THRESHOLD_SLOW)]
    normal = df[df['predicted_speed'] >= SPEED_THRESHOLD_SLOW]
    
    print(f"\n🚦 Estado de los vehículos:")
    print(f"   🔴 Detenidos (< {SPEED_THRESHOLD_STOPPED} km/h): {len(stopped)}")
    print(f"   🟡 Lentos ({SPEED_THRESHOLD_STOPPED}-{SPEED_THRESHOLD_SLOW} km/h): {len(slow)}")
    print(f"   🟢 Normales (> {SPEED_THRESHOLD_SLOW} km/h): {len(normal)}")
    
    # Mostrar vehículos con problemas
    if len(stopped) > 0:
        print(f"\n🔴 Vehículos detenidos:")
        for _, vehicle in stopped.head(5).iterrows():
            print(f"   • {vehicle['vehicle_id']} - Ruta {vehicle['route_id']} - {vehicle['agency_id']}")
    
    # Estadísticas por agencia
    print(f"\n📊 Estadísticas por agencia:")
    agency_stats = df.groupby('agency_id')['predicted_speed'].agg(['mean', 'count']).round(1)
    for agency, stats in agency_stats.iterrows():
        print(f"   • {agency}: {stats['count']} vehículos, velocidad promedio {stats['mean']} km/h")
    
    return df

def compare_routes(route1, route2):
    """Comparar dos rutas"""
    print(f"\n🛣️  Comparando rutas {route1} vs {route2}...")
    
    # Obtener vehículos de cada ruta
    df1 = get_route_vehicles(route1)
    df2 = get_route_vehicles(route2)
    
    if df1.empty or df2.empty:
        print("❌ Una o ambas rutas no tienen vehículos activos")
        return
    
    # Cargar modelo
    model, scaler = load_model()
    
    # Predecir velocidades
    df1 = calculate_features(df1)
    df1 = predict_speed(df1, model, scaler)
    
    df2 = calculate_features(df2)
    df2 = predict_speed(df2, model, scaler)
    
    # Comparar
    result = {
        'route1': {
            'route_id': route1,
            'vehicles': len(df1),
            'avg_speed': df1['predicted_speed'].mean(),
            'min_speed': df1['predicted_speed'].min(),
            'max_speed': df1['predicted_speed'].max()
        },
        'route2': {
            'route_id': route2,
            'vehicles': len(df2),
            'avg_speed': df2['predicted_speed'].mean(),
            'min_speed': df2['predicted_speed'].min(),
            'max_speed': df2['predicted_speed'].max()
        }
    }
    
    print(f"\n📊 Resultados:")
    print(f"\n   Ruta {route1}:")
    print(f"      Vehículos: {result['route1']['vehicles']}")
    print(f"      Velocidad promedio: {result['route1']['avg_speed']:.1f} km/h")
    print(f"      Rango: {result['route1']['min_speed']:.1f} - {result['route1']['max_speed']:.1f} km/h")
    
    print(f"\n   Ruta {route2}:")
    print(f"      Vehículos: {result['route2']['vehicles']}")
    print(f"      Velocidad promedio: {result['route2']['avg_speed']:.1f} km/h")
    print(f"      Rango: {result['route2']['min_speed']:.1f} - {result['route2']['max_speed']:.1f} km/h")
    
    # Recomendación
    if result['route1']['avg_speed'] > result['route2']['avg_speed']:
        faster = route1
        diff = result['route1']['avg_speed'] - result['route2']['avg_speed']
    else:
        faster = route2
        diff = result['route2']['avg_speed'] - result['route1']['avg_speed']
    
    print(f"\n✅ Recomendación: Ruta {faster} es {diff:.1f} km/h más rápida")
    
    return result

# ============================================================================
# GUARDAR ALERTAS EN BD
# ============================================================================

def save_alert_to_db(alert_data):
    """Guardar alerta en la base de datos"""
    conn = connect_db()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO transit_alerts 
        (vehicle_id, route_id, agency_id, alert_type, severity, description, 
         latitude, longitude, detected_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    """
    
    try:
        cursor.execute(query, (
            alert_data.get('vehicle_id'),
            alert_data.get('route_id'),
            alert_data.get('agency_id'),
            alert_data.get('alert_type'),
            alert_data.get('severity'),
            alert_data.get('description'),
            alert_data.get('latitude'),
            alert_data.get('longitude')
        ))
        conn.commit()
        print(f"✅ Alerta guardada en BD")
    except Exception as e:
        print(f"❌ Error al guardar alerta: {e}")
    finally:
        cursor.close()
        conn.close()

# ============================================================================
# MAIN - EJEMPLOS DE USO
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 SISTEMA DE PREDICCIÓN ETA Y ALERTAS - SF TRANSIT")
    print("="*70)
    
    # Ejemplo 1: Monitorear todos los vehículos
    print("\n" + "="*70)
    print("EJEMPLO 1: MONITOREO GENERAL")
    print("="*70)
    monitor_all_vehicles()
    
    # Ejemplo 2: Obtener vehículo específico para ETA
    print("\n" + "="*70)
    print("EJEMPLO 2: PREDICCIÓN DE ETA")
    print("="*70)
    
    df = get_all_active_vehicles()
    if not df.empty:
        # Tomar primer vehículo como ejemplo
        vehicle_id = df.iloc[0]['vehicle_id']
        current_lat = df.iloc[0]['latitude']
        current_lon = df.iloc[0]['longitude']
        
        # Destino ejemplo: 1km al norte
        dest_lat = current_lat + 0.01
        dest_lon = current_lon
        
        eta_result = predict_eta(vehicle_id, dest_lat, dest_lon)
    
    # Ejemplo 3: Detectar anomalías
    print("\n" + "="*70)
    print("EJEMPLO 3: DETECCIÓN DE ANOMALÍAS")
    print("="*70)
    
    if not df.empty:
        # Analizar primeros 3 vehículos
        for i in range(min(3, len(df))):
            vehicle_id = df.iloc[i]['vehicle_id']
            detect_anomalies(vehicle_id)
    
    # Ejemplo 4: Comparar rutas
    print("\n" + "="*70)
    print("EJEMPLO 4: COMPARACIÓN DE RUTAS")
    print("="*70)
    
    routes_df = df[df['route_id'].notna()]
    if len(routes_df) >= 2:
        unique_routes = routes_df['route_id'].unique()
        if len(unique_routes) >= 2:
            compare_routes(unique_routes[0], unique_routes[1])
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()
