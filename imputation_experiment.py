# Importar librerías necesarias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from fancyimpute import SoftImpute
from clases import ImputationExperiment


# Configuración de estilo visual
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Colores del tema editorial
COLORS = {
    'primary': '#2C5530',      # Verde bosque profundo
    'secondary': '#F5F5DC',    # Beige suave
    'accent': '#8B4513',       # Marrón tierra
    'light_green': '#90EE90',  # Verde claro
    'warm_brown': '#D2B48C'    # Marrón cálido
}

print("✓ Todas las librerías importadas correctamente")

def main():
    """
    Función principal para demostrar el uso del experimento.
    """
    print("Experimento 01: Experimento de Imputación con Validación Cruzada")
    print("=" * 60)
    
    # Cargar el dataset
    df = pd.read_csv('/media/datos/WORKSPACE/DOCTORADO/phd_code/notebooks/experiments/df_encoded.csv')
    # del df['Unnamed: 0']
    print(f"Main - Columnas del dataframe: {df.columns}")
    
    X = df.values
    
    print(f"Main - Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Inicializar y ejecutar experimento
    experiment = ImputationExperiment(n_folds=5, random_state=42)
    
    try:
        # Ejecutar experimento
        experiment.run_experiment(X, None, missing_rate=0.2, complete_sample_frac=0.3)
        
        # Generar reportes
        results_df = experiment.generate_ranking_report()
        
        # Crear visualizaciones
        print("\nMain - Generando visualizaciones...")
        experiment.plot_results_comparison('plots_imputacion/imputation_comparison.png')
        
        # Mostrar tabla de resultados
        print("\nMain - Tabla completa de resultados:")
        print("=" * 100)
        print(results_df.round(4))
        
        # Guardar resultados
        os.makedirs('imputation_experiment', exist_ok=True)
        results_df.to_csv('imputation_experiment/imputation_results.csv', index=False)
        print(f"\nMain - Resultados guardados en: imputation_experiment/imputation_results.csv")
        
        return experiment, results_df
        
    except Exception as e:
        print(f"Main - Error en la ejecución: {str(e)}")
        return None, None

# Ejecutar la demo
experiment, results = main()