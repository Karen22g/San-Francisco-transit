"""
ANÁLISIS EXPLORATORIO DE DATOS - SF TRANSIT (VERSIÓN MEJORADA)
Script para analizar los datos recolectados de la API 511.org
Calcula velocidades a partir de posiciones GPS
Genera visualizaciones y estadísticas descriptivas
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'database': 'transit_streaming',
    'user': 'pachonarvaez',
    'port': 5432
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcular distancia entre dos puntos GPS usando la fórmula de Haversine
    Retorna la distancia en kilómetros
    """
    R = 6371  # Radio de la Tierra en km
    
    # Convertir a radianes
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Diferencias
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance = R * c
    
    return distance

def calculate_speed_from_positions(df):
    """
    Calcular velocidad basada en cambio de posición GPS
    Velocidad = distancia / tiempo
    """
    print("\n🔄 Calculando velocidades desde posiciones GPS...")
    
    # Crear copia y ordenar por vehículo y tiempo
    df = df.copy()
    df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    
    # Inicializar columnas
    df['speed_calculated'] = np.nan
    df['distance_traveled'] = np.nan
    df['time_diff_seconds'] = np.nan
    
    # Calcular para cada vehículo
    vehicles_processed = 0
    for vehicle_id in df['vehicle_id'].unique():
        mask = df['vehicle_id'] == vehicle_id
        indices = df[mask].index.tolist()
        
        for i in range(1, len(indices)):
            prev_idx = indices[i-1]
            curr_idx = indices[i]
            
            prev_row = df.loc[prev_idx]
            curr_row = df.loc[curr_idx]
            
            # Calcular distancia en km
            distance = haversine(
                prev_row['latitude'], prev_row['longitude'],
                curr_row['latitude'], curr_row['longitude']
            )
            
            # Calcular tiempo en segundos
            time_diff = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds()
            
            # Calcular velocidad en km/h (evitar división por cero)
            if time_diff > 0:
                speed = (distance / time_diff) * 3600  # convertir a km/h
                
                # Filtrar velocidades irreales (> 150 km/h probablemente error GPS)
                if speed <= 150:
                    df.loc[curr_idx, 'speed_calculated'] = speed
                    df.loc[curr_idx, 'distance_traveled'] = distance
                    df.loc[curr_idx, 'time_diff_seconds'] = time_diff
        
        vehicles_processed += 1
        if vehicles_processed % 100 == 0:
            print(f"   Procesados {vehicles_processed}/{df['vehicle_id'].nunique()} vehículos...")
    
    print(f"✅ Velocidades calculadas para {vehicles_processed} vehículos")
    print(f"   • Registros con velocidad: {df['speed_calculated'].notna().sum():,}")
    print(f"   • Velocidad promedio: {df['speed_calculated'].mean():.2f} km/h")
    print(f"   • Velocidad máxima: {df['speed_calculated'].max():.2f} km/h")
    
    return df

# ============================================================================
# FUNCIONES DE EXTRACCIÓN DE DATOS
# ============================================================================

def connect_db():
    """Conectar a la base de datos"""
    return psycopg2.connect(**DB_CONFIG)

def get_vehicle_positions(conn, hours=24):
    """Obtener posiciones de vehículos de las últimas N horas"""
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

def get_route_statistics(conn):
    """Obtener estadísticas por ruta"""
    query = """
        SELECT 
            route_id,
            agency_id,
            COUNT(*) as total_readings,
            COUNT(DISTINCT vehicle_id) as unique_vehicles
        FROM vehicle_positions
        WHERE route_id IS NOT NULL
        GROUP BY route_id, agency_id
        HAVING COUNT(*) > 10
        ORDER BY total_readings DESC
    """
    return pd.read_sql(query, conn)

def get_alerts(conn, hours=24):
    """Obtener alertas detectadas"""
    query = f"""
        SELECT 
            id,
            vehicle_id,
            route_id,
            agency_id,
            alert_type,
            severity,
            description,
            latitude,
            longitude,
            detected_at
        FROM transit_alerts
        WHERE detected_at > NOW() - INTERVAL '{hours} hours'
        ORDER BY detected_at DESC
    """
    return pd.read_sql(query, conn)

def get_hourly_activity(conn):
    """Obtener actividad por hora del día"""
    query = """
        SELECT 
            EXTRACT(HOUR FROM timestamp) as hour,
            COUNT(*) as readings,
            COUNT(DISTINCT vehicle_id) as unique_vehicles
        FROM vehicle_positions
        GROUP BY EXTRACT(HOUR FROM timestamp)
        ORDER BY hour
    """
    return pd.read_sql(query, conn)

def get_agency_comparison(conn):
    """Comparar agencias"""
    query = """
        SELECT 
            agency_id,
            COUNT(*) as total_readings,
            COUNT(DISTINCT vehicle_id) as total_vehicles,
            COUNT(DISTINCT route_id) as total_routes
        FROM vehicle_positions
        GROUP BY agency_id
        ORDER BY total_readings DESC
    """
    return pd.read_sql(query, conn)

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def print_dataset_overview(df):
    """Imprimir resumen del dataset"""
    print("\n" + "="*70)
    print("📊 RESUMEN DEL DATASET")
    print("="*70)
    print(f"\n📏 Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"📅 Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"🚌 Vehículos únicos: {df['vehicle_id'].nunique():,}")
    print(f"🛣️  Rutas únicas: {df['route_id'].nunique():,}")
    print(f"🏢 Agencias: {df['agency_id'].nunique()} - {df['agency_id'].unique()}")
    
    print("\n" + "-"*70)
    print("📋 INFORMACIÓN DE COLUMNAS")
    print("-"*70)
    print(df.info())
    
    print("\n" + "-"*70)
    print("📈 ESTADÍSTICAS DESCRIPTIVAS (Coordenadas)")
    print("-"*70)
    print(df[['latitude', 'longitude', 'heading']].describe())
    
    print("\n" + "-"*70)
    print("❓ VALORES NULOS")
    print("-"*70)
    null_counts = df.isnull().sum()
    null_pcts = (df.isnull().sum() / len(df) * 100).round(2)
    null_df = pd.DataFrame({
        'Nulos': null_counts,
        'Porcentaje': null_pcts
    })
    print(null_df[null_df['Nulos'] > 0])

def analyze_speed_distribution(df):
    """Analizar distribución de velocidades CALCULADAS"""
    print("\n" + "="*70)
    print("🏃 ANÁLISIS DE VELOCIDADES (CALCULADAS DESDE GPS)")
    print("="*70)
    
    speeds = df[df['speed_calculated'].notna()]['speed_calculated']
    
    if len(speeds) == 0:
        print("\n⚠️  No hay suficientes datos para calcular velocidades")
        return
    
    print(f"\n📊 Estadísticas de velocidad calculada:")
    print(f"   • Media: {speeds.mean():.2f} km/h")
    print(f"   • Mediana: {speeds.median():.2f} km/h")
    print(f"   • Desviación estándar: {speeds.std():.2f} km/h")
    print(f"   • Mínimo: {speeds.min():.2f} km/h")
    print(f"   • Máximo: {speeds.max():.2f} km/h")
    
    # Percentiles
    print(f"\n📈 Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"   • P{p}: {speeds.quantile(p/100):.2f} km/h")
    
    # Vehículos detenidos vs en movimiento
    stopped = (speeds < 5).sum()
    moving_slow = ((speeds >= 5) & (speeds < 30)).sum()
    moving_normal = ((speeds >= 30) & (speeds < 60)).sum()
    moving_fast = (speeds >= 60).sum()
    
    print(f"\n🚦 Estado de vehículos:")
    print(f"   • Detenidos (< 5 km/h): {stopped:,} ({stopped/len(speeds)*100:.1f}%)")
    print(f"   • Movimiento lento (5-30 km/h): {moving_slow:,} ({moving_slow/len(speeds)*100:.1f}%)")
    print(f"   • Movimiento normal (30-60 km/h): {moving_normal:,} ({moving_normal/len(speeds)*100:.1f}%)")
    print(f"   • Movimiento rápido (>60 km/h): {moving_fast:,} ({moving_fast/len(speeds)*100:.1f}%)")

def analyze_temporal_patterns(df):
    """Analizar patrones temporales"""
    print("\n" + "="*70)
    print("⏰ ANÁLISIS TEMPORAL")
    print("="*70)
    
    # Agregar columnas temporales
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Actividad por hora
    hourly = df.groupby('hour').agg({
        'vehicle_id': 'count',
        'speed_calculated': 'mean'
    }).round(2)
    hourly.columns = ['Lecturas', 'Velocidad Promedio (km/h)']
    
    print("\n📊 Actividad por hora del día:")
    print(hourly)
    
    # Hora pico
    peak_hour = hourly['Lecturas'].idxmax()
    print(f"\n🔝 Hora pico: {int(peak_hour):02d}:00 hrs con {int(hourly.loc[peak_hour, 'Lecturas']):,} lecturas")
    
    # Actividad por día
    daily = df.groupby('date').agg({
        'vehicle_id': 'count',
        'speed_calculated': 'mean'
    }).round(2)
    daily.columns = ['Lecturas', 'Velocidad Promedio (km/h)']
    
    print(f"\n📅 Actividad diaria:")
    print(daily.tail())

def analyze_by_agency(df_agency):
    """Analizar por agencia"""
    print("\n" + "="*70)
    print("🏢 ANÁLISIS POR AGENCIA")
    print("="*70)
    print(df_agency.to_string(index=False))
    
    # Agencia más activa
    top_agency = df_agency.iloc[0]
    print(f"\n🥇 Agencia más activa: {top_agency['agency_id']}")
    print(f"   • Total de lecturas: {top_agency['total_readings']:,}")
    print(f"   • Vehículos: {top_agency['total_vehicles']}")
    print(f"   • Rutas: {top_agency['total_routes']}")

def analyze_routes(df_routes):
    """Analizar rutas"""
    print("\n" + "="*70)
    print("🛣️  ANÁLISIS DE RUTAS (TOP 10)")
    print("="*70)
    print(df_routes.head(10).to_string(index=False))
    
    print(f"\n📊 Resumen de rutas:")
    print(f"   • Total de rutas: {len(df_routes)}")
    print(f"   • Rutas con >100 lecturas: {(df_routes['total_readings'] > 100).sum()}")
    print(f"   • Rutas con >1000 lecturas: {(df_routes['total_readings'] > 1000).sum()}")

def analyze_alerts(df_alerts):
    """Analizar alertas"""
    print("\n" + "="*70)
    print("🚨 ANÁLISIS DE ALERTAS")
    print("="*70)
    
    if len(df_alerts) == 0:
        print("\n✅ No se detectaron alertas")
        return
    
    print(f"\n📊 Total de alertas: {len(df_alerts)}")
    
    # Por tipo
    print("\n📋 Alertas por tipo:")
    type_counts = df_alerts['alert_type'].value_counts()
    for alert_type, count in type_counts.items():
        print(f"   • {alert_type}: {count}")
    
    # Por severidad
    print("\n⚠️  Alertas por severidad:")
    severity_counts = df_alerts['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"   • {severity}: {count}")
    
    # Por agencia
    print("\n🏢 Alertas por agencia:")
    agency_counts = df_alerts['agency_id'].value_counts()
    for agency, count in agency_counts.items():
        print(f"   • {agency}: {count}")

def analyze_geographic_coverage(df):
    """Analizar cobertura geográfica"""
    print("\n" + "="*70)
    print("🗺️  ANÁLISIS DE COBERTURA GEOGRÁFICA")
    print("="*70)
    
    print(f"\n📍 Coordenadas:")
    print(f"   • Latitud: {df['latitude'].min():.6f} a {df['latitude'].max():.6f}")
    print(f"   • Longitud: {df['longitude'].min():.6f} a {df['longitude'].max():.6f}")
    
    # Calcular centro geográfico
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    print(f"\n🎯 Centro geográfico:")
    print(f"   • Latitud: {center_lat:.6f}")
    print(f"   • Longitud: {center_lon:.6f}")
    
    # Calcular distancias máximas desde el centro
    df['dist_from_center'] = df.apply(
        lambda row: haversine(center_lat, center_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    print(f"\n📏 Área de cobertura:")
    print(f"   • Radio máximo desde centro: {df['dist_from_center'].max():.2f} km")
    print(f"   • Radio promedio: {df['dist_from_center'].mean():.2f} km")

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_speed_distribution(df):
    """Gráfico de distribución de velocidades"""
    speeds = df[df['speed_calculated'].notna()]['speed_calculated']
    
    if len(speeds) == 0:
        print("\n⚠️  No hay datos de velocidad para graficar")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    axes[0].hist(speeds, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Velocidad (km/h)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Velocidades Calculadas')
    axes[0].axvline(speeds.mean(), color='red', linestyle='--', label=f'Media: {speeds.mean():.1f}')
    axes[0].axvline(speeds.median(), color='green', linestyle='--', label=f'Mediana: {speeds.median():.1f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(speeds, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[1].set_ylabel('Velocidad (km/h)')
    axes[1].set_title('Box Plot de Velocidades')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/01_speed_distribution.png', dpi=300, bbox_inches='tight')
    print("\n💾 Gráfico guardado: plots/01_speed_distribution.png")
    plt.close()

def plot_hourly_activity(df_hourly):
    """Gráfico de actividad por hora"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Lecturas
    ax1.bar(df_hourly['hour'], df_hourly['readings'], alpha=0.7, label='Lecturas', color='steelblue')
    ax1.set_xlabel('Hora del día')
    ax1.set_ylabel('Número de lecturas', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(range(24))
    ax1.grid(alpha=0.3)
    
    # Vehículos únicos
    ax2 = ax1.twinx()
    ax2.plot(df_hourly['hour'], df_hourly['unique_vehicles'], 'o-', 
             color='orangered', linewidth=2, label='Vehículos únicos', markersize=8)
    ax2.set_ylabel('Vehículos únicos', color='orangered')
    ax2.tick_params(axis='y', labelcolor='orangered')
    
    plt.title('Actividad de Transporte por Hora del Día')
    fig.tight_layout()
    plt.savefig('plots/02_hourly_activity.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/02_hourly_activity.png")
    plt.close()

def plot_agency_comparison(df_agency):
    """Comparación entre agencias"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Lecturas totales
    axes[0].bar(df_agency['agency_id'], df_agency['total_readings'], color='steelblue')
    axes[0].set_title('Total de Lecturas por Agencia')
    axes[0].set_ylabel('Lecturas')
    axes[0].set_xlabel('Agencia')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Vehículos
    axes[1].bar(df_agency['agency_id'], df_agency['total_vehicles'], color='green')
    axes[1].set_title('Total de Vehículos por Agencia')
    axes[1].set_ylabel('Vehículos')
    axes[1].set_xlabel('Agencia')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Rutas
    axes[2].bar(df_agency['agency_id'], df_agency['total_routes'], color='orange')
    axes[2].set_title('Total de Rutas por Agencia')
    axes[2].set_ylabel('Rutas')
    axes[2].set_xlabel('Agencia')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/03_agency_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/03_agency_comparison.png")
    plt.close()

def plot_route_heatmap(df):
    """Mapa de calor de rutas"""
    # Tomar muestra si hay muchos datos
    if len(df) > 10000:
        df_sample = df.sample(10000)
    else:
        df_sample = df
    
    plt.figure(figsize=(12, 10))
    
    # Crear mapa de calor con hexbin
    hb = plt.hexbin(df_sample['longitude'], df_sample['latitude'], 
                    gridsize=30, cmap='YlOrRd', mincnt=1)
    
    plt.colorbar(hb, label='Densidad de vehículos')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa de Calor - Densidad de Vehículos en SF Bay Area')
    plt.grid(alpha=0.3)
    
    plt.savefig('plots/04_route_heatmap.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/04_route_heatmap.png")
    plt.close()

def plot_speed_by_agency(df):
    """Velocidades por agencia"""
    df_speed = df[df['speed_calculated'].notna()].copy()
    
    if len(df_speed) == 0:
        print("\n⚠️  No hay datos de velocidad para graficar por agencia")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Box plot por agencia
    agencies = sorted(df_speed['agency_id'].unique())
    data = [df_speed[df_speed['agency_id'] == agency]['speed_calculated'] for agency in agencies]
    
    bp = plt.boxplot(data, labels=agencies, patch_artist=True)
    
    # Colorear
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(agencies)]):
        patch.set_facecolor(color)
    
    plt.ylabel('Velocidad (km/h)')
    plt.xlabel('Agencia')
    plt.title('Distribución de Velocidades Calculadas por Agencia')
    plt.grid(alpha=0.3, axis='y')
    
    plt.savefig('plots/05_speed_by_agency.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/05_speed_by_agency.png")
    plt.close()

def plot_speed_over_time(df):
    """Velocidad promedio a lo largo del tiempo"""
    df_speed = df[df['speed_calculated'].notna()].copy()
    
    if len(df_speed) == 0:
        print("\n⚠️  No hay datos de velocidad para graficar en el tiempo")
        return
    
    # Agrupar por intervalos de tiempo
    df_speed['timestamp_rounded'] = df_speed['timestamp'].dt.floor('10min')
    speed_over_time = df_speed.groupby('timestamp_rounded')['speed_calculated'].agg(['mean', 'std', 'count'])
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(speed_over_time.index, speed_over_time['mean'], 
             linewidth=2, label='Velocidad promedio', color='steelblue')
    plt.fill_between(speed_over_time.index, 
                     speed_over_time['mean'] - speed_over_time['std'],
                     speed_over_time['mean'] + speed_over_time['std'],
                     alpha=0.3, label='± 1 desviación estándar')
    
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad (km/h)')
    plt.title('Evolución de Velocidad Promedio en el Tiempo')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('plots/06_speed_over_time.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/06_speed_over_time.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS - SF TRANSIT (MEJORADO)")
    print("="*70)
    print("ℹ️  Nota: La API 511.org NO provee velocidades directamente")
    print("   Las velocidades se calculan a partir de posiciones GPS")
    
    # Crear carpeta de plots si no existe
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Conectar a BD
    print("\n🔌 Conectando a la base de datos...")
    conn = connect_db()
    print("✅ Conexión establecida")
    
    # Extraer datos
    print("\n📥 Extrayendo datos...")
    df_vehicles = get_vehicle_positions(conn, hours=48)
    df_routes = get_route_statistics(conn)
    df_alerts = get_alerts(conn, hours=48)
    df_hourly = get_hourly_activity(conn)
    df_agency = get_agency_comparison(conn)
    
    if len(df_vehicles) == 0:
        print("\n⚠️  No hay datos disponibles. Ejecuta primero 01_data_ingestion_511.py")
        conn.close()
        return
    
    print(f"✅ Datos extraídos: {len(df_vehicles):,} posiciones de vehículos")
    
    # CALCULAR VELOCIDADES DESDE POSICIONES GPS
    df_vehicles = calculate_speed_from_positions(df_vehicles)
    
    # Análisis
    print_dataset_overview(df_vehicles)
    analyze_speed_distribution(df_vehicles)
    analyze_temporal_patterns(df_vehicles)
    analyze_geographic_coverage(df_vehicles)
    analyze_by_agency(df_agency)
    analyze_routes(df_routes)
    analyze_alerts(df_alerts)
    
    # Visualizaciones
    print("\n" + "="*70)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*70)
    
    plot_speed_distribution(df_vehicles)
    plot_hourly_activity(df_hourly)
    plot_agency_comparison(df_agency)
    plot_route_heatmap(df_vehicles)
    plot_speed_by_agency(df_vehicles)
    plot_speed_over_time(df_vehicles)
    
    # Cerrar conexión
    conn.close()
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)
    print("\n📁 Visualizaciones guardadas en: plots/")
    print("   • 01_speed_distribution.png")
    print("   • 02_hourly_activity.png")
    print("   • 03_agency_comparison.png")
    print("   • 04_route_heatmap.png")
    print("   • 05_speed_by_agency.png")
    print("   • 06_speed_over_time.png")

if __name__ == "__main__":
    main()
