"""
ANÁLISIS EXPLORATORIO DE DATOS - SF TRANSIT
Script para analizar los datos recolectados de la API 511.org
Genera visualizaciones y estadísticas descriptivas
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
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
        ORDER BY timestamp DESC
    """
    return pd.read_sql(query, conn)

def get_route_statistics(conn):
    """Obtener estadísticas por ruta"""
    query = """
        SELECT 
            route_id,
            agency_id,
            COUNT(*) as total_readings,
            COUNT(DISTINCT vehicle_id) as unique_vehicles,
            AVG(speed) as avg_speed,
            MAX(speed) as max_speed,
            MIN(speed) as min_speed,
            STDDEV(speed) as std_speed
        FROM vehicle_positions
        WHERE speed IS NOT NULL
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
            COUNT(DISTINCT vehicle_id) as unique_vehicles,
            AVG(speed) as avg_speed
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
            COUNT(DISTINCT route_id) as total_routes,
            AVG(speed) as avg_speed,
            STDDEV(speed) as std_speed
        FROM vehicle_positions
        WHERE speed IS NOT NULL
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
    print("📈 ESTADÍSTICAS DESCRIPTIVAS")
    print("-"*70)
    print(df.describe())
    
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
    """Analizar distribución de velocidades"""
    print("\n" + "="*70)
    print("🏃 ANÁLISIS DE VELOCIDADES")
    print("="*70)
    
    speeds = df[df['speed'].notna()]['speed']
    
    print(f"\n📊 Estadísticas de velocidad:")
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
    moving = (speeds >= 5).sum()
    print(f"\n🚦 Estado de vehículos:")
    print(f"   • Detenidos (< 5 km/h): {stopped:,} ({stopped/len(speeds)*100:.1f}%)")
    print(f"   • En movimiento (≥ 5 km/h): {moving:,} ({moving/len(speeds)*100:.1f}%)")

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
        'speed': 'mean'
    }).round(2)
    hourly.columns = ['Lecturas', 'Velocidad Promedio']
    
    print("\n📊 Actividad por hora del día:")
    print(hourly)
    
    # Hora pico
    peak_hour = hourly['Lecturas'].idxmax()
    print(f"\n🔝 Hora pico: {peak_hour}:00 hrs con {hourly.loc[peak_hour, 'Lecturas']:,} lecturas")
    
    # Actividad por día
    daily = df.groupby('date').agg({
        'vehicle_id': 'count',
        'speed': 'mean'
    }).round(2)
    
    print(f"\n📅 Actividad diaria (últimos 5 días):")
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
    print(f"   • Velocidad promedio: {top_agency['avg_speed']:.2f} km/h")

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

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_speed_distribution(df):
    """Gráfico de distribución de velocidades"""
    speeds = df[df['speed'].notna()]['speed']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    axes[0].hist(speeds, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Velocidad (km/h)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Velocidades')
    axes[0].axvline(speeds.mean(), color='red', linestyle='--', label=f'Media: {speeds.mean():.1f}')
    axes[0].axvline(speeds.median(), color='green', linestyle='--', label=f'Mediana: {speeds.median():.1f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(speeds, vert=True)
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
    ax2.plot(df_hourly['hour'], df_hourly['unique_vehicles'], 'o-', color='orangered', linewidth=2, label='Vehículos únicos')
    ax2.set_ylabel('Vehículos únicos', color='orangered')
    ax2.tick_params(axis='y', labelcolor='orangered')
    
    plt.title('Actividad de Transporte por Hora del Día')
    fig.tight_layout()
    plt.savefig('plots/02_hourly_activity.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/02_hourly_activity.png")
    plt.close()

def plot_agency_comparison(df_agency):
    """Comparación entre agencias"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Lecturas totales
    axes[0, 0].bar(df_agency['agency_id'], df_agency['total_readings'], color='steelblue')
    axes[0, 0].set_title('Total de Lecturas por Agencia')
    axes[0, 0].set_ylabel('Lecturas')
    axes[0, 0].grid(alpha=0.3)
    
    # Vehículos
    axes[0, 1].bar(df_agency['agency_id'], df_agency['total_vehicles'], color='green')
    axes[0, 1].set_title('Total de Vehículos por Agencia')
    axes[0, 1].set_ylabel('Vehículos')
    axes[0, 1].grid(alpha=0.3)
    
    # Rutas
    axes[1, 0].bar(df_agency['agency_id'], df_agency['total_routes'], color='orange')
    axes[1, 0].set_title('Total de Rutas por Agencia')
    axes[1, 0].set_ylabel('Rutas')
    axes[1, 0].grid(alpha=0.3)
    
    # Velocidad promedio
    axes[1, 1].bar(df_agency['agency_id'], df_agency['avg_speed'], color='red')
    axes[1, 1].set_title('Velocidad Promedio por Agencia')
    axes[1, 1].set_ylabel('Velocidad (km/h)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/03_agency_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/03_agency_comparison.png")
    plt.close()

def plot_route_heatmap(df):
    """Mapa de calor de rutas"""
    # Tomar solo las últimas 10000 posiciones para que sea manejable
    df_sample = df.head(10000)
    
    plt.figure(figsize=(12, 10))
    
    # Crear mapa de calor con hexbin
    plt.hexbin(df_sample['longitude'], df_sample['latitude'], 
               gridsize=30, cmap='YlOrRd', mincnt=1)
    
    plt.colorbar(label='Densidad de vehículos')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa de Calor - Densidad de Vehículos en SF Bay Area')
    plt.grid(alpha=0.3)
    
    plt.savefig('plots/04_route_heatmap.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/04_route_heatmap.png")
    plt.close()

def plot_speed_by_agency(df):
    """Velocidades por agencia"""
    plt.figure(figsize=(12, 6))
    
    # Filtrar solo datos con velocidad
    df_speed = df[df['speed'].notna()]
    
    # Box plot por agencia
    agencies = df_speed['agency_id'].unique()
    data = [df_speed[df_speed['agency_id'] == agency]['speed'] for agency in agencies]
    
    bp = plt.boxplot(data, labels=agencies, patch_artist=True)
    
    # Colorear
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(agencies)]):
        patch.set_facecolor(color)
    
    plt.ylabel('Velocidad (km/h)')
    plt.xlabel('Agencia')
    plt.title('Distribución de Velocidades por Agencia')
    plt.grid(alpha=0.3)
    
    plt.savefig('plots/05_speed_by_agency.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: plots/05_speed_by_agency.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS - SF TRANSIT")
    print("="*70)
    
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
    
    # Análisis
    print_dataset_overview(df_vehicles)
    analyze_speed_distribution(df_vehicles)
    analyze_temporal_patterns(df_vehicles)
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

if __name__ == "__main__":
    main()