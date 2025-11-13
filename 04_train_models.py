"""
ENTRENAMIENTO DE MODELOS - SF TRANSIT
Entrenar y evaluar modelos de Machine Learning
Autor: Francisco Narvaez M
Fecha: 2025-11-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
MODELS_DIR = 'models'
PLOTS_DIR = 'plots/models'

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def load_datasets():
    """Cargar datasets de entrenamiento y prueba"""
    print("\n📥 Cargando datasets...")
    
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    print(f"✅ Train: {len(train_df):,} registros")
    print(f"✅ Test: {len(test_df):,} registros")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Preparar features para entrenamiento"""
    print("\n🔄 Preparando features...")
    
    # Target: Predecir velocidad futura
    target_col = 'speed_calculated'
    
    # Features a excluir
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'latitude', 'longitude', 'distance_to_center', 'heading'
    ]
     
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"✅ Features seleccionados: {len(feature_cols)}")
    print(f"   {feature_cols[:5]}... (mostrando primeros 5)")
    
    return X_train, y_train, X_test, y_test, feature_cols

def normalize_features(X_train, X_test):
    """Normalizar features"""
    print("\n🔄 Normalizando features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features normalizados")
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# MODELOS DE MACHINE LEARNING
# ============================================================================

def train_linear_regression(X_train, y_train):
    """Entrenar modelo de Regresión Lineal"""
    print("\n📈 Entrenando Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✅ Linear Regression entrenado")
    return model

def train_ridge(X_train, y_train):
    """Entrenar modelo Ridge"""
    print("\n📈 Entrenando Ridge Regression...")
    
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    print("✅ Ridge Regression entrenado")
    return model

def train_lasso(X_train, y_train):
    """Entrenar modelo Lasso"""
    print("\n📈 Entrenando Lasso Regression...")
    
    model = Lasso(alpha=0.1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    print("✅ Lasso Regression entrenado")
    return model

def train_decision_tree(X_train, y_train):
    """Entrenar árbol de decisión"""
    print("\n🌳 Entrenando Decision Tree...")
    
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    print("✅ Decision Tree entrenado")
    return model

def train_random_forest(X_train, y_train):
    """Entrenar Random Forest"""
    print("\n🌲 Entrenando Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    print("✅ Random Forest entrenado")
    return model

def train_gradient_boosting(X_train, y_train):
    """Entrenar Gradient Boosting"""
    print("\n🚀 Entrenando Gradient Boosting...")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    print("✅ Gradient Boosting entrenado")
    return model

# ============================================================================
# EVALUACIÓN DE MODELOS
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluar modelo con métricas"""
    print(f"\n📊 Evaluando {model_name}...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"   • RMSE: {rmse:.2f} km/h")
    print(f"   • MAE: {mae:.2f} km/h")
    print(f"   • R²: {r2:.4f}")
    print(f"   • MAPE: {mape:.2f}%")
    
    return metrics, y_pred

def compare_models(results_df):
    """Comparar todos los modelos"""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Mejor modelo por métrica
    print("\n🏆 Mejores modelos por métrica:")
    print(f"   • Mejor R²: {results_df.loc[results_df['R2'].idxmax(), 'Model']}")
    print(f"   • Menor RMSE: {results_df.loc[results_df['RMSE'].idxmin(), 'Model']}")
    print(f"   • Menor MAE: {results_df.loc[results_df['MAE'].idxmin(), 'Model']}")
    
    return results_df.loc[results_df['R2'].idxmax(), 'Model']

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_predictions_vs_actual(y_test, predictions_dict):
    """Gráfico de predicciones vs valores reales"""
    print("\n📈 Generando gráfico predicciones vs real...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        
        # Línea perfecta
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Predicción perfecta')
        
        ax.set_xlabel('Velocidad Real (km/h)')
        ax.set_ylabel('Velocidad Predicha (km/h)')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {PLOTS_DIR}/predictions_vs_actual.png")
    plt.close()

def plot_model_comparison(results_df):
    """Gráfico de comparación de modelos"""
    print("\n📊 Generando gráfico de comparación...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE
    axes[0, 0].barh(results_df['Model'], results_df['RMSE'], color='steelblue')
    axes[0, 0].set_xlabel('RMSE (km/h)')
    axes[0, 0].set_title('Root Mean Squared Error')
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # MAE
    axes[0, 1].barh(results_df['Model'], results_df['MAE'], color='green')
    axes[0, 1].set_xlabel('MAE (km/h)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # R²
    axes[1, 0].barh(results_df['Model'], results_df['R2'], color='orange')
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_title('R² Score (mayor es mejor)')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # MAPE
    axes[1, 1].barh(results_df['Model'], results_df['MAPE'], color='red')
    axes[1, 1].set_xlabel('MAPE (%)')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {PLOTS_DIR}/model_comparison.png")
    plt.close()

def plot_residuals(y_test, y_pred, model_name):
    """Gráfico de residuales"""
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuales vs predicciones
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicciones (km/h)')
    axes[0].set_ylabel('Residuales (km/h)')
    axes[0].set_title(f'Residuales vs Predicciones - {model_name}')
    axes[0].grid(alpha=0.3)
    
    # Histograma de residuales
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuales (km/h)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'Distribución de Residuales - {model_name}')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/residuals_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {PLOTS_DIR}/residuals_{model_name.replace(' ', '_')}.png")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Gráfico de importancia de features (solo para modelos basados en árboles)"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    print(f"\n📊 Generando importancia de features para {model_name}...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Top 20 Features más importantes - {model_name}')
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/feature_importance_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {PLOTS_DIR}/feature_importance_{model_name.replace(' ', '_')}.png")
    plt.close()

# ============================================================================
# GUARDAR MODELOS
# ============================================================================

def save_models(models_dict, best_model_name, scaler):
    """Guardar modelos entrenados"""
    print("\n💾 Guardando modelos...")
    
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Guardar todos los modelos
    for name, model in models_dict.items():
        filename = f"{MODELS_DIR}/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        print(f"   ✅ {filename}")
    
    # Guardar scaler
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print(f"   ✅ {MODELS_DIR}/scaler.pkl")
    
    # Marcar el mejor modelo
    best_model = models_dict[best_model_name]
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    print(f"   ✅ {MODELS_DIR}/best_model.pkl (Mejor modelo: {best_model_name})")
    
    print("\n✅ Todos los modelos guardados")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 ENTRENAMIENTO DE MODELOS - SF TRANSIT")
    print("="*70)
    
    # Crear directorios
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Cargar datos
    train_df, test_df = load_datasets()
    
    # 2. Preparar features
    X_train, y_train, X_test, y_test, feature_names = prepare_features(train_df, test_df)
    
    # 3. Normalizar
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # 4. Entrenar modelos
    print("\n" + "="*70)
    print("🎯 ENTRENANDO MODELOS")
    print("="*70)
    
    models = {
        'Linear Regression': train_linear_regression(X_train_scaled, y_train),
        'Ridge': train_ridge(X_train_scaled, y_train),
        'Lasso': train_lasso(X_train_scaled, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'Gradient Boosting': train_gradient_boosting(X_train, y_train)
    }
    
    # 5. Evaluar modelos
    print("\n" + "="*70)
    print("📊 EVALUANDO MODELOS")
    print("="*70)
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        if 'Decision Tree' in name or 'Forest' in name or 'Boosting' in name:
            metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        else:
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test, name)
        
        results.append(metrics)
        predictions[name] = y_pred
    
    results_df = pd.DataFrame(results)
    
    # 6. Comparar modelos
    best_model_name = compare_models(results_df)
    
    # 7. Visualizaciones
    print("\n" + "="*70)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*70)
    
    plot_predictions_vs_actual(y_test, predictions)
    plot_model_comparison(results_df)
    plot_residuals(y_test, predictions[best_model_name], best_model_name)
    
    # Feature importance para modelos basados en árboles
    for name, model in models.items():
        if 'Tree' in name or 'Forest' in name or 'Boosting' in name:
            plot_feature_importance(model, feature_names, name)
    
    # 8. Guardar modelos
    save_models(models, best_model_name, scaler)
    
    # 9. Guardar resultados
    results_df.to_csv(f'{MODELS_DIR}/model_results.csv', index=False)
    print(f"\n💾 Resultados guardados: {MODELS_DIR}/model_results.csv")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n🏆 Mejor modelo: {best_model_name}")
    print(f"   • R²: {results_df[results_df['Model'] == best_model_name]['R2'].values[0]:.4f}")
    print(f"   • RMSE: {results_df[results_df['Model'] == best_model_name]['RMSE'].values[0]:.2f} km/h")
    print(f"   • MAE: {results_df[results_df['Model'] == best_model_name]['MAE'].values[0]:.2f} km/h")
    
    print("\n📁 Archivos generados:")
    print(f"   • Modelos: {MODELS_DIR}/")
    print(f"   • Gráficos: {PLOTS_DIR}/")
    print(f"   • Resultados: {MODELS_DIR}/model_results.csv")
    
    print("\n🚀 Modelos listos para predicciones!")

if __name__ == "__main__":
    main()
