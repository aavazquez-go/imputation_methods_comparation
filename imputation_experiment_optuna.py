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
    df_path = '/media/datos/WORKSPACE/DOCTORADO/datasets/df_encoded.csv'
    if not os.path.exists(df_path):
        # Intentar rutas alternativas si existen
        alt_paths = [
            'df_encoded.csv',
            '/media/datos/WORKSPACE/DOCTORADO/phd_code/notebooks/experiments/df_encoded.csv'
        ]
        found = None
        for ap in alt_paths:
            if os.path.exists(ap):
                found = ap
                break
        if found is not None:
            df_path = found
        else:
            print(f"AVISO: No se encontró {df_path}. Ajusta la ruta al dataset.")
    
    df = pd.read_csv(df_path)
    print(f"Columnas del dataframe: {df.columns}")
    
    X = df.values
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Inicializar experimento
    experiment = ImputationExperiment(n_folds=5, random_state=42)
    
    # Métodos que serán tuneados con Optuna (definido aquí para reutilizarlo en gráficos)
    tune_methods_list = ['miss_forest', 'mice_hgb', 'soft_impute', 'gain']
    
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
            tune_methods=tune_methods_list,
            metric='mse'
        )
        
        # Generar reporte
        results_df_rep = experiment.generate_repeated_holdout_report()
        
        # -----------------------------------------------
        # Figuras para toma de decisiones / paper
        # -----------------------------------------------
        plots_dir = 'imputation_experiment_optuna/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # (1) FIGURA PRINCIPAL: boxplot de métricas de TEST por método (repeticiones)
        try:
            experiment.plot_repeated_holdout_boxplot(
                save_path=f'{plots_dir}/boxplot_test_metrics.png')
        except Exception as e:
            print(f"No se pudo generar boxplot: {e}")
        
        # (2) JUSTIFICACIÓN METODOLÓGICA: convergencia por nº de repeticiones
        try:
            experiment.plot_convergence_repetitions(
                metric='mse',
                save_path=f'{plots_dir}/convergence_repetitions_mse.png')
        except Exception as e:
            print(f"No se pudo generar convergencia: {e}")
        
        # (3) DIAGNÓSTICO DE BÚSQUEDA (anexo): historial de Optuna por método tunado
        #     (repetición -1 = última)
        for mname in tune_methods_list:
            try:
                experiment.plot_optuna_history(
                    mname, repeat_idx=-1,
                    save_path=f'{plots_dir}/optuna_history_{mname}.png')
            except Exception as e:
                print(f"No se pudo generar historial Optuna para {mname}: {e}")
        
        # (4) DIAGNÓSTICO INTERNO: importancia de hiperparámetros (no para el paper)
        for mname in tune_methods_list:
            try:
                experiment.plot_optuna_param_importances(
                    mname, repeat_idx=-1,
                    save_path=f'{plots_dir}/optuna_importances_{mname}.png')
            except Exception as e:
                print(f"No se pudieron generar importancias para {mname}: {e}")
        
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
