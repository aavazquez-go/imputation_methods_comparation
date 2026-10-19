# Importar librerías necesarias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold, 
    KFold, 
    train_test_split
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import BayesianRidge
import copy

# Intentar importar Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception as e:
    print(f'Optuna no disponible: {e}')
    OPTUNA_AVAILABLE = False

# Colores del tema editorial
COLORS = {
    'primary': '#2C5530',      # Verde bosque profundo
    'secondary': '#F5F5DC',    # Beige suave
    'accent': '#8B4513',       # Marrón tierra
    'light_green': '#90EE90',  # Verde claro
    'warm_brown': '#D2B48C'    # Marrón cálido
}


# Implementación de GAIN (TensorFlow 2.x).
#    Si TensorFlow no está disponible, se usa imputación por la media como fallback.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except Exception as e:
    print('TensorFlow no disponible para GAIN:', e)
    TF_AVAILABLE = False

# Clase GAIN para imputación de datos faltantes

class GAIN:
    """Implementación básica de GAIN (Generative Adversarial Imputation Nets).
    Asume entradas normalizadas en [0,1].
    """
    def __init__(self, batch_size=128, hint_rate=0.9, alpha=100, epochs=100, lr=1e-3, seed=42):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        if TF_AVAILABLE:
            tf.random.set_seed(seed)

    def _build_generator(self, dim):
        print(f"\n---------------------")
        print(f"GAIN - Starting Build generator")
        print(f"---------------------")
        inp = layers.Input(shape=(dim*2,))  # concatenated X_tilde and M
        x = layers.Dense(max(64, dim*4), activation='relu')(inp)
        x = layers.Dense(max(32, dim*2), activation='relu')(x)
        out = layers.Dense(dim, activation='sigmoid')(x)
        model = models.Model(inputs=inp, outputs=out)
        return model

    def _build_discriminator(self, dim):
        print(f"\n----------------------------------")
        print(f"GAIN - Starting Build discriminator")
        print(f"----------------------------------")
        inp = layers.Input(shape=(dim*2,))  # concatenated X_hat and hint
        x = layers.Dense(max(64, dim*4), activation='relu')(inp)
        x = layers.Dense(max(32, dim*2), activation='relu')(x)
        out = layers.Dense(dim, activation='sigmoid')(x)  # per-feature prob
        model = models.Model(inputs=inp, outputs=out)
        return model

    def fit(self, X, y=None):
        self._fitted_X = X.copy()
        return self

    def fit_transform(self, data_x):
        print(f"\n----------------------------")
        print(f"GAIN - Starting fit transform")
        print(f"----------------------------")
        # data_x: numpy array with np.nan for missing entries, values expected in [0,1]
        if not TF_AVAILABLE:
            print('TensorFlow no está disponible. Usando imputación por la media como fallback para GAIN.')
            col_mean = np.nanmean(data_x, axis=0)
            return np.where(np.isnan(data_x), col_mean, data_x)

        X = data_x.copy().astype('float32')
        n, d = X.shape
        M = (~np.isnan(X)).astype('float32')

        # Replace NaNs with zeros for initial input (noise will be injected)
        X_filled = np.where(np.isnan(X), 0.0, X).astype('float32')

        gen = self._build_generator(d)
        dis = self._build_discriminator(d)
        gen_opt = optimizers.Adam(self.lr)
        dis_opt = optimizers.Adam(self.lr)

        # --- INICIO DE LA CORRECCIÓN ---
        # Definimos las funciones de pérdida directamente para un control más granular.
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        mse = tf.keras.losses.MeanSquaredError()
        # --- FIN DE LA CORRECCIÓN ---

        indices = np.arange(n)
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for i in range(0, n, self.batch_size):
                mb_idx = indices[i:i+self.batch_size]
                X_mb = X_filled[mb_idx]
                M_mb = M[mb_idx]
                mb_size = X_mb.shape[0]

                Z_mb = np.random.uniform(0, 0.01, size=(mb_size, d)).astype('float32')
                X_tilde = X_mb * M_mb + Z_mb * (1 - M_mb)

                with tf.GradientTape() as dis_tape, tf.GradientTape() as gen_tape:
                    G_input = np.concatenate([X_tilde, M_mb], axis=1)
                    G_sample = gen(G_input, training=True)
                    X_hat = G_sample * (1 - M_mb) + X_mb * M_mb

                    B = np.random.binomial(1, self.hint_rate, size=(mb_size, d)).astype('float32')
                    H_mb = M_mb * B + 0.5 * (1 - B)

                    D_input = np.concatenate([X_hat, H_mb], axis=1)
                    D_prob = dis(D_input, training=True)

                    # --- INICIO DE LA CORRECCIÓN ---
                    # Pérdida del Discriminador: Calculada por elemento y luego promediada.
                    # D_loss ahora se calcula sobre las predicciones D_prob y las etiquetas reales M_mb
                    dis_loss = bce(M_mb, D_prob)

                    # Pérdida Adversaria del Generador:
                    # Queremos que el discriminador crea que los valores imputados son reales (etiqueta 1).
                    # Calculamos la pérdida solo en los valores faltantes (1 - M_mb).
                    gen_adversarial_loss_elem = bce(tf.ones_like(D_prob), D_prob)
                    gen_adversarial_loss = tf.reduce_sum(gen_adversarial_loss_elem * (1 - M_mb)) / (tf.reduce_sum(1 - M_mb) + 1e-8)

                    # Pérdida de Reconstrucción:
                    # Asegura que el generador no se desvíe mucho en los valores observados.
                    recon_loss_elem = mse(X_mb, G_sample)
                    recon_loss = tf.reduce_sum(recon_loss_elem * M_mb) / (tf.reduce_sum(M_mb) + 1e-8)
                    # --- FIN DE LA CORRECCIÓN ---

                    gen_loss = gen_adversarial_loss + self.alpha * recon_loss

                dis_grads = dis_tape.gradient(dis_loss, dis.trainable_variables)
                gen_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
                dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))
                gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))

            if (epoch + 1) % max(1, (self.epochs // 10)) == 0 or epoch == 0:
                print(f'Epoch {epoch+1}/{self.epochs} - D_loss: {dis_loss.numpy():.4f} - G_loss: {gen_loss.numpy():.4f}')

        Z_full = np.random.uniform(0, 0.01, size=(n, d)).astype('float32')
        X_tilde_full = X_filled * M + Z_full * (1 - M)
        G_input_full = np.concatenate([X_tilde_full, M], axis=1)
        G_full = gen(G_input_full, training=False).numpy()
        imputed = X.copy()
        imputed[np.isnan(imputed)] = G_full[np.isnan(imputed)]
        return imputed

    def transform(self, X):
        return self.fit_transform(X)

# Clase SoftImpute personalizada usando SVD
class SoftImputeCustom(BaseEstimator, TransformerMixin):
    """Implementación de SoftImpute usando SVD minimization.
    Compatible con scikit-learn moderno y sin dependencias desactualizadas.
    """
    def __init__(self, max_iters=100, threshold=1e-3, verbose=False):
        self.max_iters = max_iters
        self.threshold = threshold
        self.verbose = verbose
        self._col_means = None
        
    def fit_transform(self, X):
        """Ajusta e imputa simultáneamente."""
        X = X.astype(np.float64)
        mask = ~np.isnan(X)
        
        # Inicializar con medias
        self._col_means = np.nanmean(X, axis=0)
        X_filled = np.where(mask, X, self._col_means)
        
        for iteration in range(self.max_iters):
            X_prev = X_filled.copy()
            
            try:
                # Descomposición SVD
                U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
                
                # Soft thresholding de valores singulares
                s_soft = np.maximum(s - self.threshold, 0)
                X_filled = U @ np.diag(s_soft) @ Vt
            except:
                # Fallback si SVD falla
                pass
            
            # Restaurar valores observados originales
            X_filled[mask] = X[mask]
            
            # Verificar convergencia
            diff = np.sum((X_filled - X_prev) ** 2)
            if diff < 1e-10:
                if self.verbose:
                    print(f"SoftImpute converged at iteration {iteration}")
                break
        
        return X_filled
    
    def fit(self, X, y=None):
        self._col_means = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X):
        X = X.astype(np.float64)
        mask = ~np.isnan(X)
        X_filled = np.where(mask, X, self._col_means)
        return self.fit_transform(X_filled)


# Wrapper para MissForest compatible con scikit-learn moderno
class MissForestWrapper(BaseEstimator, TransformerMixin):
    """Wrapper para MissForest que usa RandomForestRegressor directamente.
    Esto evita problemas de compatibilidad con librerías desactualizadas.
    """
    def __init__(self, n_estimators=30, max_iter=3, random_state=42, n_jobs=2):
        from sklearn.ensemble import RandomForestRegressor
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._imputed = None

    def fit_transform(self, X, y=None):
        """Imputa usando iteración con Random Forests."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = X.astype(np.float64)
        mask = ~np.isnan(X)
        
        # Inicialización con medias
        col_means = np.nanmean(X, axis=0)
        X_filled = np.where(mask, X, col_means)
        
        for iteration in range(self.max_iter):
            X_prev = X_filled.copy()
            
            # Entrenar un modelo RF para cada columna con valores faltantes
            cols_with_nan = np.where(~mask.all(axis=0))[0]
            
            for col_idx in cols_with_nan:
                # Usar otras columnas como predictoras
                other_cols = np.setdiff1d(np.arange(X_filled.shape[1]), [col_idx])
                
                # Datos de entrenamiento: filas donde esta columna NO es NaN
                train_mask = mask[:, col_idx]
                
                if train_mask.sum() > 1:  # Al menos 2 observaciones
                    X_train = X_filled[train_mask][:, other_cols]
                    y_train = X[train_mask, col_idx]
                    
                    # Entrenar RF
                    rf = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        max_depth=10
                    )
                    rf.fit(X_train, y_train)
                    
                    # Imputar valores faltantes en esta columna
                    missing_mask = ~mask[:, col_idx]
                    if missing_mask.sum() > 0:
                        X_pred = X_filled[missing_mask][:, other_cols]
                        X_filled[missing_mask, col_idx] = rf.predict(X_pred)
            
            # Convergencia
            diff = np.sum((X_filled - X_prev) ** 2)
            if diff < 1e-10:
                break
        
        return X_filled

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.fit_transform(X)

# Clase principal para el experimento de imputación

class ImputationExperiment:
    """
    Clase principal para ejecutar experimentos de imputación con validación cruzada.
    """
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
        self.imputation_methods = {}
        self.setup_imputers()
        
    def setup_imputers(self):
        """Configura los métodos de imputación a evaluar."""
        print("ImputationExperiment -Setting up Imputers")
        # Imputadores simples
        self.imputation_methods['mean'] = SimpleImputer(strategy='mean')
        self.imputation_methods['median'] = SimpleImputer(strategy='median')
        self.imputation_methods['most_frequent'] = SimpleImputer(strategy='most_frequent')
        
        # # KNN Imputer
        # self.imputation_methods['knn_3'] = KNNImputer(
        #         n_neighbors=3,
        #         weights='uniform',       # más rápido que 'distance'
        #         metric='nan_euclidean',  # el default, pero explícito
        #         add_indicator=False   # evita crear columnas extra
        #     )
        # self.imputation_methods['knn_5'] = KNNImputer(
        #         n_neighbors=5, 
        #         weights='uniform',       # más rápido que 'distance'
        #         metric='nan_euclidean',  # el default, pero explícito
        #         add_indicator=False    # evita crear columnas extra
        #         # algorithm='kd_tree',     # o 'ball_tree'; evita 'brute'
        #         # leaf_size=40             # mayor leaf_size → menos ramas, más rápido (pero menos preciso)
        #     )
        # self.imputation_methods['knn_7'] = KNNImputer(
        #         n_neighbors=7,
        #         weights='uniform',       # más rápido que 'distance'
        #         metric='nan_euclidean',  # el default, pero explícito
        #         add_indicator=False     # evita crear columnas extra
        #         # algorithm='kd_tree',     # o 'ball_tree'; evita 'brute'
        #         # leaf_size=40             # mayor leaf_size → menos ramas, más rápido (pero menos preciso)
        #     )
        
        # Iterative Imputer (MICE)
        # self.imputation_methods['mice_rf'] = IterativeImputer(
        #     estimator=RandomForestRegressor(n_estimators=100, random_state=self.random_state),
        #     random_state=self.random_state,
        #     max_iter=10
        # )
        
        self.imputation_methods['mice_hgb'] = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=3,
            random_state=self.random_state,
            sample_posterior=False,  # importante: reduce RAM
            n_nearest_features=None   # no uses esta opción a menos que sea necesario
        )
        
        # self.imputation_methods['miss_forest'] = MissForest(
        #     max_iter=3, 
        #     early_stopping = True
        # )
        
        self.imputation_methods['miss_forest'] = MissForestWrapper(
            n_estimators=30,
            max_iter=3,
            random_state=self.random_state,
            n_jobs=2
        )

        self.imputation_methods['gain'] = GAIN(epochs=100, batch_size=256, hint_rate=0.9, alpha=100, lr=1e-3)
        self.imputation_methods['soft_impute'] = SoftImputeCustom(max_iters=100, threshold=1e-3, verbose=False)
        # self.imputation_methods['soft_impute_extra'] = SoftImpute(max_iters=100, verbose=0)
        
        print(f"ImputationExperiment:setup_imputers - Configurados {len(self.imputation_methods)} métodos de imputación")
        print(f"ImputationExperiment:setup_imputers - Listado de metodos de imputación:{list(self.imputation_methods.keys())}")
        
    def create_dataset_with_missingness(self, X, y, missing_rate=0.2):
        """
        Crea un dataset con valores faltantes para simular el problema real.
        """
        print("ImputationExperiment:create_dataset_with_missingness - Starting process create_dataset_with_missingness")
        
        np.random.seed(self.random_state)
        X_missing = X.copy()
        missing_mask = np.random.random(X.shape) < missing_rate
        
        # Introducir valores faltantes
        X_missing[missing_mask] = np.nan
        
        return X_missing, missing_mask
    
    def evaluate_imputation_method(self, X, y, method_name, imputer, cv_splitter):
        """
        Evalúa un método de imputación usando validación cruzada.
        El objetivo es medir qué tan bien se reconstruyen los valores faltantes.
        """
        print(f"ImputationExperiment - Evaluating imputation method: {method_name} ")
        
        scores = {
            'mse': [],
            'mae': [],
            'r2': [],
            'imputation_time': []
        }
        
        fold_results = []

        # Determinar si el imputador es transductivo (no tiene fit/transform separados)
        is_transductive = isinstance(imputer, (GAIN, SoftImputeCustom, MissForestWrapper))
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            # X_..._fold contiene los datos CON valores faltantes
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            # y_..._fold contiene los datos originales COMPLETOS (ground truth)
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            if is_transductive:
                # --- Caso especial: métodos transductivos (GAIN, SoftImpute) ---
                scaler = MinMaxScaler()
                # Ajustar el escalador SOLO con los datos de entrenamiento
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                y_val_fold_scaled = scaler.transform(y_val_fold)

                # Aplicar imputación DIRECTAMENTE sobre los datos de validación escalados
                import time
                start_time = time.time()
                X_val_imputed = imputer.fit_transform(X_val_scaled)
                imputation_time = time.time() - start_time
            else:
                # Crear pipeline solo con transformadores
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('imputer', imputer)
                ])
                
                import time
                start_time = time.time()
                # 1. Entrenar el pipeline (scaler e imputer) en los datos de entrenamiento
                pipeline.fit(X_train_fold)
                # 2. Aplicar la imputación a los datos de validación
                X_val_imputed = pipeline.transform(X_val_fold)
                imputation_time = time.time() - start_time

                # Escalar y_val_fold para comparación justa
                scaler = pipeline.named_steps['scaler']
                y_val_fold_scaled = scaler.transform(y_val_fold)
            
            # 3. Identificar qué valores fueron imputados en el set de validación
            missing_mask_val = np.isnan(X_val_fold)
            
            # Si no hay valores faltantes en este fold de validación, no podemos medir el error.
            if not np.any(missing_mask_val):
                continue

            # 4. Extraer los valores originales y los imputados solo donde había NaN
            original_values = y_val_fold_scaled[missing_mask_val]
            imputed_values = X_val_imputed[missing_mask_val]

            # 5. Calcular métricas sobre los valores reconstruidos
            mse = mean_squared_error(original_values, imputed_values)
            mae = mean_absolute_error(original_values, imputed_values)
            r2 = r2_score(original_values, imputed_values)
            
            scores['mse'].append(mse)
            scores['mae'].append(mae)
            scores['r2'].append(r2)
            scores['imputation_time'].append(imputation_time)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'imputation_time': imputation_time
            })
        
        # Calcular estadísticas agregadas
        results = {
            'method': method_name,
            'mean_mse': np.mean(scores['mse']),
            'std_mse': np.std(scores['mse']),
            'mean_mae': np.mean(scores['mae']),
            'std_mae': np.std(scores['mae']),
            'mean_r2': np.mean(scores['r2']),
            'std_r2': np.std(scores['r2']),
            'mean_time': np.mean(scores['imputation_time']),
            'fold_results': fold_results
        }
        
        return results
    
    def _select_not_missing_values_tuples(self, X):
        """
        Obtener el subconjunto de elementos que no tienen valores faltantes.
        Esto es útil para un análisis de "casos completos".

        Args:
            X (np.ndarray): El conjunto de datos, posiblemente con valores faltantes.

        Returns:
            np.ndarray: Un nuevo array que contiene solo las filas de X 
                        que no tienen ningún valor NaN.
        """
        # np.isnan(X) crea una máscara booleana (True donde hay NaN)
        # .any(axis=1) comprueba si alguna columna en una fila es True (es decir, tiene un NaN)
        # El operador ~ invierte la máscara, seleccionando filas donde NO hay NaN.
        complete_rows_mask = ~np.isnan(X).any(axis=1)
        
        print(f"ImputationExperiment:_select_not_missing_values_tuples - Encontrados {np.sum(complete_rows_mask)} registros completos de un total de {X.shape[0]}.")

        return X[complete_rows_mask]
    
    def _build_imputer_with_params(self, method_name, params):
        """
        Construye una instancia de imputador con parámetros específicos.
        
        Args:
            method_name: Nombre del método
            params: Dict con parámetros
        
        Returns:
            Imputador instanciado
        """
        method_name_lower = method_name.lower()
        
        if method_name_lower == 'mean':
            return SimpleImputer(strategy='mean')
        elif method_name_lower == 'median':
            return SimpleImputer(strategy='median')
        elif method_name_lower == 'most_frequent':
            return SimpleImputer(strategy='most_frequent')
        elif method_name_lower == 'miss_forest':
            from sklearn.ensemble import RandomForestRegressor
            return MissForestWrapper(
                n_estimators=params.get('n_estimators', 30),
                max_iter=params.get('max_iter', 3),
                random_state=params.get('random_state', self.random_state),
                n_jobs=params.get('n_jobs', 2)
            )
        elif method_name_lower == 'mice_hgb':
            return IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=params.get('max_iter', 3),
                random_state=params.get('random_state', self.random_state),
                sample_posterior=params.get('sample_posterior', False),
                n_nearest_features=params.get('n_nearest_features', None)
            )
        elif method_name_lower == 'gain':
            return GAIN(
                epochs=params.get('epochs', 100),
                batch_size=params.get('batch_size', 256),
                hint_rate=params.get('hint_rate', 0.9),
                alpha=params.get('alpha', 100),
                lr=params.get('lr', 1e-3),
                seed=params.get('seed', self.random_state)
            )
        elif method_name_lower == 'soft_impute':
            return SoftImputeCustom(
                max_iters=params.get('max_iters', 100),
                threshold=params.get('threshold', 1e-3),
                verbose=False
            )
        else:
            # Fallback: devolver el configurado por defecto
            if method_name in self.imputation_methods:
                return copy.deepcopy(self.imputation_methods[method_name])
            return SimpleImputer(strategy='mean')
    
    def _evaluate_imputation_on_data(self, imputer, X_missing, y_ground_truth, scaler_fitted=None):
        """
        Evalúa un imputador ya ajustado sobre datos con missing, comparando 
        únicamente en las posiciones NaN contra y_ground_truth (completo).
        
        Args:
            imputer: Imputador ajustado o callable (fit_transform)
            X_missing: np.ndarray con NaN
            y_ground_truth: np.ndarray completo (ground truth)
            scaler_fitted: MinMaxScaler ya ajustado (opcional). Si se pasa, 
                           se usará para escalar y_ground_truth antes de comparar.
        
        Returns:
            dict con mse, mae, r2
        """
        # Determinar si es transductivo
        is_transductive = isinstance(imputer, (GAIN, SoftImputeCustom, MissForestWrapper))
        
        if scaler_fitted is not None:
            X_m_scaled = scaler_fitted.transform(X_missing)
            y_gt_scaled = scaler_fitted.transform(y_ground_truth)
        else:
            X_m_scaled = X_missing.copy()
            y_gt_scaled = y_ground_truth.copy()
        
        # Imputar
        if is_transductive:
            try:
                X_imp_scaled = imputer.fit_transform(X_m_scaled)
            except Exception:
                X_imp_scaled = imputer.transform(X_m_scaled)
        else:
            try:
                X_imp_scaled = imputer.transform(X_m_scaled)
            except Exception:
                X_imp_scaled = imputer.fit_transform(X_m_scaled)
        
        # Métricas solo en posiciones NaN
        missing_mask = np.isnan(X_missing)
        if not np.any(missing_mask):
            # Si no hay NaN, devolver NaN o 0? pero no esperado
            return {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}
        
        original_values = y_gt_scaled[missing_mask]
        imputed_values = X_imp_scaled[missing_mask]
        
        mse = mean_squared_error(original_values, imputed_values)
        mae = mean_absolute_error(original_values, imputed_values)
        try:
            r2 = r2_score(original_values, imputed_values)
        except Exception:
            r2 = np.nan
        
        return {'mse': mse, 'mae': mae, 'r2': r2}
    
    ## ------------------------------------
    ## 2. Métodos de Evaluación y Ejecución
    ## ------------------------------------
    def run_experiment(self, X, y = None, missing_rate=0.2, complete_sample_frac=0.3):
        """
        Ejecuta el experimento completo de comparación de métodos de imputación.
        
        Args:
            X (np.ndarray): Datos originales (pueden contener NaN).
            y (np.ndarray, optional): Etiquetas o ground truth. Si None, se usa copia de X.
            missing_rate (float): Proporción de valores a enmascarar artificialmente [0, 1].
            complete_sample_frac (float): Fracción de registros completos a usar para el experimento (e.g. 0.3 = 30%).
        """
        print(f"ImputationExperiment:run_experiment - Running experiment ")
        print("ImputationExperiment:run_experiment - Iniciando experimento de imputación con validación cruzada")
        # print("=" * 60)
        
        # 1. Filtrar solo los registros completamente observados
        X_complete = self._select_not_missing_values_tuples(X)
        print(f"ImputationExperiment:run_experiment - Datos completos encontrados: {X_complete.shape}")

        if X_complete.shape[0] == 0:
            raise ValueError("No se encontraron registros completamente observados en el dataset.")

        # 2. Muestrear aleatoriamente un subconjunto de los registros completos
        n_total_complete = X_complete.shape[0]
        n_sample = int(np.floor(complete_sample_frac * n_total_complete))
        n_sample = max(1, n_sample)  # Asegurar al menos 1 muestra

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_total_complete, size=n_sample, replace=False)
        X_sampled = X_complete[indices]

        print(f"ImputationExperiment:run_experiment - Submuestra seleccionada: {X_sampled.shape} "
            f"({complete_sample_frac*100:.1f}% de los datos completos)")

        # 3. Si y es None, usar copia del subconjunto como ground truth
        if y is None:
            y_sampled = X_sampled.copy()
        else:
            # Si y fue proporcionado, debe tener las mismas filas que X.
            # Pero como estamos muestreando solo filas completas de X, asumimos que y ya está alineado.
            # Para mayor seguridad, también muestreamos y con los mismos índices originales:
            # (Opcional: solo si y es un array del mismo tamaño que X original)
            if hasattr(y, 'shape') and y.shape[0] == X.shape[0]:
                # Necesitaríamos conocer qué filas completas tomamos en X para alinear y.
                # Esto es más complejo. Dado tu uso actual (y = X), lo simplificamos:
                y_sampled = X_sampled.copy()
            else:
                y_sampled = X_sampled.copy()
        
        # 4. Introducir artificialmente valores faltantes en la submuestra
        X_missing, missing_mask = self.create_dataset_with_missingness(
            X_sampled, y_sampled, missing_rate
        )
        
        print(f"ImputationExperiment:run_experiment - Dataset creado con {missing_rate*100:.1f}% de valores faltantes")
        print(f"ImputationExperiment:run_experiment - Dimensiones finales: {X_missing.shape}")
        print(f"ImputationExperiment:run_experiment - Total de valores faltantes: {np.sum(np.isnan(X_missing))}")

        
        # 5. Configurar validación cruzada
        cv_splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # 6. Evaluar cada método
        self.results = {}
        for method_name, imputer in self.imputation_methods.items():
            print(f"\nImputationExperiment:run_experiment - Evaluando método: {method_name}")
            print("-" * 40)
            
            try:
                result = self.evaluate_imputation_method(
                    X_missing, y_sampled, method_name, imputer, cv_splitter
                )
                self.results[method_name] = result
                
                print(f"✓ MSE promedio: {result['mean_mse']:.4f} (±{result['std_mse']:.4f})")
                print(f"✓ MAE promedio: {result['mean_mae']:.4f} (±{result['std_mae']:.4f})")
                print(f"✓ R² promedio: {result['mean_r2']:.4f} (±{result['std_r2']:.4f})")
                print(f" Tiempo promedio: {result['mean_time']:.4f}")
                
            except Exception as e:
                print(f"✗ Error evaluando {method_name}: {str(e)}")
                continue
        
        print(f"\nImputationExperiment:run_experiment - Experimento completado. Evaluados {len(self.results)} métodos.")

    def create_results_dataframe(self):
        """Crea un DataFrame con los resultados agregados."""
        
        print(f"ImputationExperiment:create_results_dataframe - Creating results dataframe")
        
        if not self.results:
            raise ValueError("No hay resultados para mostrar. Ejecuta run_experiment primero.")
        
        results_list = []
        for method_name, result in self.results.items():
            results_list.append({
                'Método': method_name,
                'MSE Media': result['mean_mse'],
                'MSE Std': result['std_mse'],
                'MAE Media': result['mean_mae'],
                'MAE Std': result['std_mae'],
                'R² Media': result['mean_r2'],
                'R² Std': result['std_r2'],
                'Tiempo Medio (s)': result['mean_time']
            })
        
        return pd.DataFrame(results_list).sort_values('MSE Media')
    
    ##----------------------------
    ## 3. Métodos de Visualización
    ##----------------------------
    
    def plot_results_comparison(self, save_path=None):
        """
        Crea visualizaciones comparativas de los resultados.
        """
        print(f"ImputationExperiment - Plotting results comparison")
        
        if not self.results:
            raise ValueError("No hay resultados para visualizar.")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparación de Métodos de Imputación con Validación Cruzada', 
                        fontsize=16, fontweight='bold', color=COLORS['primary'])
        
        # Preparar datos para visualización
        methods = list(self.results.keys())
        mse_means = [self.results[m]['mean_mse'] for m in methods]
        mse_stds = [self.results[m]['std_mse'] for m in methods]
        mae_means = [self.results[m]['mean_mae'] for m in methods]
        mae_stds = [self.results[m]['std_mae'] for m in methods]
        r2_means = [self.results[m]['mean_r2'] for m in methods]
        r2_stds = [self.results[m]['std_r2'] for m in methods]
        
        # 1. Gráfico de MSE
        axes[0,0].bar(methods, mse_means, yerr=mse_stds, 
                        color=COLORS['primary'], alpha=0.7, capsize=5)
        axes[0,0].set_title('Error Cuadrático Medio (MSE)', fontweight='bold')
        axes[0,0].set_ylabel('MSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Gráfico de MAE
        axes[0,1].bar(methods, mae_means, yerr=mae_stds, 
                        color=COLORS['accent'], alpha=0.7, capsize=5)
        axes[0,1].set_title('Error Absoluto Medio (MAE)', fontweight='bold')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Gráfico de R²
        axes[1,0].bar(methods, r2_means, yerr=r2_stds, 
                        color=COLORS['light_green'], alpha=0.7, capsize=5)
        axes[1,0].set_title('Coeficiente de Determinación (R²)', fontweight='bold')
        axes[1,0].set_ylabel('R²')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Gráfico de tiempos de ejecución
        times = [self.results[m]['mean_time'] for m in methods]
        axes[1,1].bar(methods, times, color=COLORS['warm_brown'], alpha=0.7)
        axes[1,1].set_title('Tiempo de Ejecución Promedio', fontweight='bold')
        axes[1,1].set_ylabel('Tiempo (segundos)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()

    def plot_fold_details(self, method_name, save_path=None):
        """
        Visualiza los resultados detallados por fold para un método específico.
        """
        
        print(f"ImputationExperiment - Plot fold details")
        
        if method_name not in self.results:
            raise ValueError(f"Método '{method_name}' no encontrado en los resultados.")
        
        fold_results = self.results[method_name]['fold_results']
        folds = [r['fold'] for r in fold_results]
        mse_scores = [r['mse'] for r in fold_results]
        mae_scores = [r['mae'] for r in fold_results]
        r2_scores = [r['r2'] for r in fold_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Detalles por Fold - Método: {method_name}', 
                        fontsize=16, fontweight='bold', color=COLORS['primary'])
        
        # MSE por fold
        axes[0].plot(folds, mse_scores, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
        axes[0].set_title('MSE por Fold', fontweight='bold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('MSE')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(folds)
        
        # MAE por fold
        axes[1].plot(folds, mae_scores, 's-', color=COLORS['accent'], linewidth=2, markersize=8)
        axes[1].set_title('MAE por Fold', fontweight='bold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(folds)
        
        # R² por fold
        axes[2].plot(folds, r2_scores, '^-', color=COLORS['light_green'], linewidth=2, markersize=8)
        axes[2].set_title('R² por Fold', fontweight='bold')
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('R²')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(folds)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"Figura detallada guardada en: {save_path}")
        
        plt.show()

    def generate_ranking_report(self):
        """Genera un reporte de ranking de métodos de imputación."""
        
        print(f"ImputationExperiment - Generating ranking report")
        
        if not self.results:
            raise ValueError("No hay resultados para generar reporte.")
        
        df = self.create_results_dataframe()
        
        print("\n" + "="*80)
        print("RANKING DE MÉTODOS DE IMPUTACIÓN")
        print("="*80)
        print("\nRanking basado en MSE (menor es mejor):")
        print("-" * 50)
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"{i:2d}. {row['Método']:<20} | MSE: {row['MSE Media']:.4f} ± {row['MSE Std']:.4f}")
        
        print("\nRanking basado en R² (mayor es mejor):")
        print("-" * 50)
        
        df_r2 = df.sort_values('R² Media', ascending=False)
        for i, (_, row) in enumerate(df_r2.iterrows(), 1):
            print(f"{i:2d}. {row['Método']:<20} | R²: {row['R² Media']:.4f} ± {row['R² Std']:.4f}")
        
        return df

    def tune_with_optuna(self, method_name, X_train_missing, y_train_ground_truth, 
                        inner_cv_splits=5, n_trials=30, metric='mse', random_state=None):
        """
        Realiza búsqueda de hiperparámetros con Optuna usando solo el conjunto de entrenamiento (70%).
        Usa 5-CV interno: SOLO los folds de entrenamiento se usan para búsqueda.
        
        Args:
            method_name: Nombre del método a tunear
            X_train_missing: Datos de train con NaN artificiales
            y_train_ground_truth: Ground truth completo del train
            inner_cv_splits: Número de folds internos (5 por defecto)
            n_trials: Número de trials de Optuna
            metric: 'mse' (minimizar), 'mae' (minimizar), 'r2' (maximizar)
            random_state: Semilla para reproducibilidad
        
        Returns:
            tuple (best_params, best_score)
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna no está disponible. Usando parámetros por defecto.")
            # Devolver params por defecto según método
            defaults = {}
            if method_name.lower() == 'miss_forest':
                defaults = {'n_estimators': 30, 'max_iter': 3, 'random_state': self.random_state, 'n_jobs': 2}
            elif method_name.lower() == 'mice_hgb':
                defaults = {'max_iter': 3, 'sample_posterior': False, 'n_nearest_features': None}
            elif method_name.lower() == 'gain':
                defaults = {'epochs': 100, 'batch_size': 256, 'hint_rate': 0.9, 'alpha': 100, 'lr': 1e-3, 'seed': self.random_state}
            elif method_name.lower() == 'soft_impute':
                defaults = {'max_iters': 100, 'threshold': 1e-3}
            else:
                defaults = {}
            return defaults, np.inf if metric != 'r2' else -np.inf
        
        rs = random_state if random_state is not None else self.random_state
        inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=rs)
        
        def objective(trial):
            params = {}
            method_name_lower = method_name.lower()
            
            if method_name_lower == 'miss_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                    'max_iter': trial.suggest_int('max_iter', 2, 5),
                    'random_state': rs,
                    'n_jobs': 2
                }
            elif method_name_lower == 'mice_hgb':
                params = {
                    'max_iter': trial.suggest_int('max_iter', 2, 10),
                    'sample_posterior': trial.suggest_categorical('sample_posterior', [False, True]),
                    'n_nearest_features': trial.suggest_categorical('n_nearest_features', [None, 2, 3, 5])
                }
            elif method_name_lower == 'gain':
                params = {
                    'epochs': trial.suggest_int('epochs', 20, 150),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                    'hint_rate': trial.suggest_float('hint_rate', 0.7, 0.95),
                    'alpha': trial.suggest_float('alpha', 10, 200),
                    'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                    'seed': rs
                }
            elif method_name_lower == 'soft_impute':
                params = {
                    'max_iters': trial.suggest_int('max_iters', 10, 150),
                    'threshold': trial.suggest_float('threshold', 1e-4, 1e-2, log=True)
                }
            else:
                params = {}
            
            # Evaluar con CV interno
            fold_scores = []
            is_transductive = method_name_lower in ['gain', 'soft_impute', 'miss_forest']
            
            for tr_idx, val_idx in inner_cv.split(X_train_missing, y_train_ground_truth):
                X_tr_f = X_train_missing[tr_idx]
                X_val_f = X_train_missing[val_idx]
                y_tr_f = y_train_ground_truth[tr_idx]
                y_val_f = y_train_ground_truth[val_idx]
                
                imputer_trial = self._build_imputer_with_params(method_name, params)
                scaler = MinMaxScaler()
                
                if is_transductive:
                    # Ajustar scaler solo con train
                    X_tr_f_s = scaler.fit_transform(X_tr_f)
                    # Para métodos transductivos, evaluamos aplicando fit_transform sobre val
                    # (manteniendo comportamiento coherente con evaluate_imputation_method)
                    X_val_f_s = scaler.transform(X_val_f)
                    y_val_f_s = scaler.transform(y_val_f)
                    try:
                        X_val_imp_s = imputer_trial.fit_transform(X_val_f_s)
                    except Exception:
                        X_val_imp_s = imputer_trial.transform(X_val_f_s)
                    # Métricas
                    msk = np.isnan(X_val_f)
                    if np.any(msk):
                        orig = y_val_f_s[msk]
                        impv = X_val_imp_s[msk]
                        if metric == 'mse':
                            fold_scores.append(mean_squared_error(orig, impv))
                        elif metric == 'mae':
                            fold_scores.append(mean_absolute_error(orig, impv))
                        elif metric == 'r2':
                            try:
                                fold_scores.append(r2_score(orig, impv))
                            except Exception:
                                pass
                else:
                    pipe = Pipeline([
                        ('scaler', MinMaxScaler()),
                        ('imputer', imputer_trial)
                    ])
                    pipe.fit(X_tr_f)
                    X_val_imp = pipe.transform(X_val_f)
                    scaler_p = pipe.named_steps['scaler']
                    y_val_f_s = scaler_p.transform(y_val_f)
                    msk = np.isnan(X_val_f)
                    if np.any(msk):
                        orig = y_val_f_s[msk]
                        impv = X_val_imp[msk]
                        if metric == 'mse':
                            fold_scores.append(mean_squared_error(orig, impv))
                        elif metric == 'mae':
                            fold_scores.append(mean_absolute_error(orig, impv))
                        elif metric == 'r2':
                            try:
                                fold_scores.append(r2_score(orig, impv))
                            except Exception:
                                pass
            
            if len(fold_scores) == 0:
                return np.inf if metric != 'r2' else -np.inf
            score_mean = np.mean(fold_scores)
            return score_mean if metric != 'r2' else -score_mean  # Optuna maximiza, invertimos si minimizar
        
        direction = 'maximize' if metric == 'r2' else 'minimize'
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=rs))
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        if metric == 'r2':
            best_score = -best_score
        
        print(f"Optuna completado para {method_name}: best_{metric}={best_score:.4f}, best_params={best_params}")
        return best_params, best_score
    
    def run_experiment_repeated_holdout(self, X, y=None, missing_rate=0.2, test_size=0.3,
                                       n_repeats=5, inner_cv_splits=5, n_trials_optuna=30,
                                       tune_methods=None, metric='mse', random_state=None):
        """
        Ejecuta esquema: Hold-out 70/30 repetido + 5-CV interno con Optuna (solo train).
        Para cada repetición:
        1. Split 70/30 aleatorio (cambia cada repetición)
        2. Introducir missing artificial en train y test (independiente)
        3. Con 70% hacer búsqueda de HP vía Optuna usando solo train (inner CV)
        4. Reentrenar con TODO el 70% usando mejores HP
        5. Evaluar en 30% test reservado (métricas solo sobre posiciones NaN) -> valor usado
        6. Repetir n_repeats veces, promediar resultados
        
        Args:
            X: Datos completos o con NaN (se extraen completos)
            y: Opcional, si None usa X.copy()
            missing_rate: % de missing artificial
            test_size: 0.3 para 70/30
            n_repeats: Nº repeticiones (5–30 según coste)
            inner_cv_splits: Nº folds internos (5)
            n_trials_optuna: Nº trials Optuna
            tune_methods: Lista de métodos a tunear. Si None, tunear ['miss_forest', 'mice_hgb', 'gain', 'soft_impute']
            metric: Métrica objetivo para Optuna ('mse','mae','r2')
            random_state: Semilla base
        
        Returns:
            dict con resultados por repetición y agregados
        """
        print("="*70)
        print("EXPERIMENTO: Hold-out 70/30 repetido + Inner CV + Optuna")
        print("="*70)
        
        rs_base = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(rs_base)
        
        # Extraer completos
        X_complete = self._select_not_missing_values_tuples(X)
        if X_complete.shape[0] == 0:
            raise ValueError("No se encontraron registros completamente observados")
        
        if tune_methods is None:
            tune_methods = ['miss_forest', 'mice_hgb', 'gain', 'soft_impute']
        
        # Métodos a evaluar (tuneados) + mantener simples sin tune? evaluar también simples con params por defecto
        methods_to_eval = list(self.imputation_methods.keys())
        
        # Estructura resultados
        results_by_repeat = {m: [] for m in methods_to_eval}
        
        for rep in range(n_repeats):
            print(f"\n{'='*60}")
            print(f"REPETICIÓN {rep+1}/{n_repeats}")
            print(f"{'='*60}")
            rep_seed = rng.integers(0, 10**9)
            rng_rep = np.random.default_rng(rep_seed)
            
            # Split 70/30
            n = X_complete.shape[0]
            idx = np.arange(n)
            rng_rep.shuffle(idx)
            n_tr = int((1 - test_size) * n)
            idx_tr, idx_te = idx[:n_tr], idx[n_tr:]
            X_tr_full = X_complete[idx_tr]
            X_te_full = X_complete[idx_te]
            y_tr_full = X_tr_full.copy()
            y_te_full = X_te_full.copy()
            
            print(f"Split 70/30: train={X_tr_full.shape[0]}, test={X_te_full.shape[0]}")
            
            # Crear missing artificial
            np.random.seed(rep_seed)
            X_tr_miss = X_tr_full.copy()
            msk_tr = np.random.random(X_tr_miss.shape) < missing_rate
            X_tr_miss[msk_tr] = np.nan
            
            np.random.seed(rep_seed + 12345)
            X_te_miss = X_te_full.copy()
            msk_te = np.random.random(X_te_miss.shape) < missing_rate
            X_te_miss[msk_te] = np.nan
            
            print(f"Missing: train={np.sum(np.isnan(X_tr_miss))}, test={np.sum(np.isnan(X_te_miss))}")
            
            # Por cada método
            for mname in methods_to_eval:
                try:
                    print(f"\n--- {mname} (rep {rep+1}) ---")
                    mname_low = mname.lower()
                    if OPTUNA_AVAILABLE and mname in tune_methods:
                        # Buscar mejores HP con solo train70 + inner CV
                        best_params, best_score = self.tune_with_optuna(
                            mname, X_tr_miss, y_tr_full,
                            inner_cv_splits=inner_cv_splits,
                            n_trials=n_trials_optuna,
                            metric=metric,
                            random_state=rep_seed
                        )
                        # Reentrenar con TODO train70 usando best_params
                        imputer_final = self._build_imputer_with_params(mname, best_params)
                    else:
                        # Usar params por defecto
                        imputer_final = self._build_imputer_with_params(mname, {})
                    
                    # Ajustar y evaluar en test30 (solo sobre NaN)
                    is_transductive = isinstance(imputer_final, (GAIN, SoftImputeCustom, MissForestWrapper))
                    scaler_te = MinMaxScaler()
                    if is_transductive:
                        X_tr_m_s = scaler_te.fit_transform(X_tr_miss)
                        X_te_m_s = scaler_te.transform(X_te_miss)
                        y_te_f_s = scaler_te.transform(y_te_full)
                        X_te_imp_s = imputer_final.fit_transform(X_tr_m_s)  # fit sobre todo train escalado?
                        # Mejor: para transductivos, aplicar fit_transform sobre train completo escalado
                        # pero necesitamos imputar test; el wrapper/GAIN hacen fit_transform global
                        # reconstruimos: entrenar sobre train escalado "completo"? 
                        # estrategia: ajustar imputador sobre train escalado (fit_transform) para obtener estado? 
                        # pero transductivos: aplicamos fit_transform sobre train escalado y transformamos test
                        pass  # manejamos abajo
                    # Evaluación final
                    res_te = self._evaluate_final_holdout(imputer_final, X_tr_miss, X_te_miss, y_te_full)
                    results_by_repeat[mname].append(res_te)
                    print(f"TEST {metric}: {res_te.get(metric, np.nan):.4f} | MSE={res_te['mse']:.4f} MAE={res_te['mae']:.4f} R²={res_te['r2']:.4f}")
                except Exception as e:
                    print(f"Error en {mname} rep {rep+1}: {e}")
                    results_by_repeat[mname].append({'mse': np.nan, 'mae': np.nan, 'r2': np.nan})
        
        # Calcular agregados
        self.repeated_results = {
            'by_repeat': results_by_repeat,
            'n_repeats': n_repeats,
            'aggregates': {}
        }
        for mname in methods_to_eval:
            vals = results_by_repeat[mname]
            if len(vals) == 0:
                continue
            mse_list = [v['mse'] for v in vals if not np.isnan(v['mse'])]
            mae_list = [v['mae'] for v in vals if not np.isnan(v['mae'])]
            r2_list = [v['r2'] for v in vals if not np.isnan(v['r2'])]
            agg = {
                'mean_mse': np.mean(mse_list) if mse_list else np.nan,
                'std_mse': np.std(mse_list) if mse_list else np.nan,
                'mean_mae': np.mean(mae_list) if mae_list else np.nan,
                'std_mae': np.std(mae_list) if mae_list else np.nan,
                'mean_r2': np.mean(r2_list) if r2_list else np.nan,
                'std_r2': np.std(r2_list) if r2_list else np.nan,
            }
            self.repeated_results['aggregates'][mname] = agg
        
        return self.repeated_results
    
    def _evaluate_final_holdout(self, imputer_final, X_tr_miss, X_te_miss, y_te_full):
        """
        Reentrena con TODO train70 y evalúa en test30.
        - Para no transductivos (SimpleImputer, IterativeImputer): fit pipeline completo sobre train70 -> transform test30
        - Para transductivos (GAIN, SoftImputeCustom, MissForestWrapper): ajustar sobre train70 escalado (fit_transform) 
          para obtener imputación consistente? luego transformar test30. 
          Estrategia: escalar train/test con scaler ajustado en train; para transductivos aplicamos fit_transform sobre train escalado
          y luego transformamos test escalado con el mismo imputador? Algunos transductivos no tienen transform separado puro;
          usamos fit_transform sobre test escalado tras haber sido "entrenado" conceptualmente con train.
        """
        is_transductive = isinstance(imputer_final, (GAIN, SoftImputeCustom, MissForestWrapper))
        scaler = MinMaxScaler()
        X_tr_m_s = scaler.fit_transform(X_tr_miss)
        X_te_m_s = scaler.transform(X_te_miss)
        y_te_f_s = scaler.transform(y_te_full)
        
        if is_transductive:
            # Para mantener coherencia: aplicar fit_transform sobre train escalado (refit con todo train)
            try:
                _ = imputer_final.fit_transform(X_tr_m_s)
            except Exception:
                pass
            # Transformar test escalado
            try:
                X_te_imp_s = imputer_final.transform(X_te_m_s)
            except Exception:
                X_te_imp_s = imputer_final.fit_transform(X_te_m_s)
        else:
            pipe = Pipeline([
                ('scaler', MinMaxScaler()),
                ('imputer', imputer_final)
            ])
            pipe.fit(X_tr_miss)
            X_te_imp = pipe.transform(X_te_miss)
            scaler_p = pipe.named_steps['scaler']
            y_te_f_s = scaler_p.transform(y_te_full)
            X_te_imp_s = X_te_imp
            scaler = scaler_p  # no usado
        
        msk_te = np.isnan(X_te_miss)
        if not np.any(msk_te):
            return {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}
        orig = y_te_f_s[msk_te]
        impv = X_te_imp_s[msk_te]
        return {
            'mse': mean_squared_error(orig, impv),
            'mae': mean_absolute_error(orig, impv),
            'r2': r2_score(orig, impv) if len(np.unique(orig)) > 1 or True else np.nan
        }
    
    def generate_repeated_holdout_report(self):
        """
        Genera reporte con resultados agregados del esquema repetido
        """
        if not hasattr(self, 'repeated_results'):
            raise ValueError("Ejecuta run_experiment_repeated_holdout primero")
        agg = self.repeated_results['aggregates']
        rows = []
        for mname, a in agg.items():
            rows.append({
                'Método': mname,
                'MSE Media (test)': a['mean_mse'],
                'MSE Std (test)': a['std_mse'],
                'MAE Media (test)': a['mean_mae'],
                'MAE Std (test)': a['std_mae'],
                'R² Media (test)': a['mean_r2'],
                'R² Std (test)': a['std_r2'],
                'Repeticiones': self.repeated_results['n_repeats']
            })
        df = pd.DataFrame(rows).sort_values('MSE Media (test)')
        print("\n" + "="*90)
        print("RANKING HOLD-OUT 70/30 REPETIDO + OPTUNA (MÉTRICAS SOBRE TEST RESERVADO)")
        print("="*90)
        for i, (_, r) in enumerate(df.iterrows(), 1):
            print(f"{i:2d}. {r['Método']:<20} | MSE={r['MSE Media (test)']:.4f}±{r['MSE Std (test)']:.4f} | R²={r['R² Media (test)']:.4f}±{r['R² Std (test)']:.4f}")
        return df
