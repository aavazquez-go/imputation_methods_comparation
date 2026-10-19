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
    Ejemplo de uso del nuevo esquema: Hold-out 70/30 repetido + Inner 5-CV + Optuna.
    
    NO modifica el experimento original. Usa la nueva funcionalidad añadida a ImputationExperiment.
    """
    print("Experimento 02: Hold-out 70/30 repetido + Optuna (Búsqueda solo en train)")
    print("=" * 80)
    
    # Cargar el dataset
    df_path = '/media/datos/WORKSPACE/DOCTORADO/phd_code/notebooks/experiments/df_encoded.csv'
    if not os.path.exists(df_path):
        # Intentar ruta relativa si existe
        alt_path = 'df_encoded.csv'
        if os.path.exists(alt_path):
            df_path = alt_path
        else:
            print(f"AVISO: No se encontró {df_path}. Ajusta la ruta al dataset.")
    
    df = pd.read_csv(df_path)
    print(f"Columnas del dataframe: {df.columns}")
    
    X = df.values
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Inicializar experimento
    experiment = ImputationExperiment(n_folds=5, random_state=42)
    
    try:
        # Ejecutar esquema propuesto por coautora
        results_repeated = experiment.run_experiment_repeated_holdout(
            X=X,
            y=None,
            missing_rate=0.2,
            test_size=0.3,           # Hold-out 70/30
            n_repeats=5,             # Cambiar a 10/30 si tiempo lo permite
            inner_cv_splits=5,       # 5-CV interno para búsqueda
            n_trials_optuna=20,      # Reducido para pruebas rápidas; aumentar a 30
            tune_methods=['miss_forest', 'mice_hgb', 'soft_impute', 'gain'],
            metric='mse'
        )
        
        # Generar reporte
        results_df_rep = experiment.generate_repeated_holdout_report()
        
        # Guardar resultados
        os.makedirs('imputation_experiment_optuna', exist_ok=True)
        results_df_rep.to_csv('imputation_experiment_optuna/results_repeated_holdout.csv', index=False)
        
        # Guardar también por repetición
        by_rep_dfs = []
        for mname, vals in results_repeated['by_repeat'].items():
            df_m = pd.DataFrame(vals)
            df_m['method'] = mname
            df_m['repeat'] = range(1, len(df_m)+1)
            by_rep_dfs.append(df_m)
        if by_rep_dfs:
            pd.concat(by_rep_dfs, ignore_index=True).to_csv(
                'imputation_experiment_optuna/results_by_repeat.csv', 
                index=False
            )
        
        print(f"\n✓ Resultados guardados en: imputation_experiment_optuna/")
        return experiment, results_repeated, results_df_rep
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    exp, res_rep, df_rep = main()
