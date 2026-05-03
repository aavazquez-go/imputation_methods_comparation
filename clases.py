# Importar librerías necesarias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import BayesianRidge

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
