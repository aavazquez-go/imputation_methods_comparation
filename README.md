# imputation_methods_comparation

Proyecto de investigación para comparar métodos de imputación de datos faltantes, desarrollado en el contexto de un doctorado. Evalúa el rendimiento de diferentes técnicas de imputación mediante validación cruzada, métricas de error y tiempos de ejecución.

## Descripción

Implementa un pipeline completo para:
- Simular valores faltantes en datasets completos
- Evaluar 7 métodos de imputación con validación cruzada (5 folds)
- Calcular métricas de rendimiento: MSE, MAE, R²
- Generar visualizaciones comparativas y exportar resultados

## Métodos de Imputación

| Método | Descripción |
|--------|-------------|
| `mean` | Imputación por media de columna |
| `median` | Imputación por mediana de columna |
| `most_frequent` | Valor más frecuente |
| `mice_hgb` | MICE con BayesianRidge |
| `miss_forest` | Basado en Random Forest (MissForest) |
| `gain` | Generative Adversarial Imputation Nets (requiere TensorFlow) |
| `soft_impute` | SoftImpute basado en SVD (implementación personalizada) |

## Estructura del Proyecto

```
├── clases.py                  # Clases core: imputadores y lógica de experimento
├── imputation_experiment.py   # Script principal de ejecución
├── 01_imputation_method_selection_improved.ipynb  # Notebook exploratorio
├── plots_imputacion/          # Gráficos generados
├── imputation_experiment/     # Resultados exportados (CSV)
└── output.txt                 # Ejemplo de salida de ejecución
```

## Dependencias

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow fancyimpute
```

TensorFlow es opcional: GAIN usa imputación por media como fallback si no está disponible.

## Uso

1. Ajustar la ruta del dataset en `imputation_experiment.py` (línea 49):
   ```python
   df = pd.read_csv('ruta/a/tu/dataset.csv')
   ```

2. Ejecutar:
   ```bash
   python imputation_experiment.py
   ```

3. Resultados:
   - Ranking impreso en consola
   - Gráficos en `plots_imputacion/`
   - CSV con métricas en `imputation_experiment/imputation_results.csv`

## Resultados de Referencia

Ejecución con dataset de 1.59M muestras, 57 features, 20% de valores faltantes:

| Ranking (MSE ↓) | Método | MSE Media | R² Media | Tiempo (s) |
|-----------------|--------|-----------|----------|------------|
| 1 | miss_forest | 0.0264 | 0.8266 | 1222.85 |
| 2 | mice_hgb | 0.0358 | 0.7714 | 465.14 |
| 3 | soft_impute | 0.0557 | 0.6333 | 55.38 |
| 4 | mean | 0.0558 | 0.6329 | 0.32 |

**Hallazgos**: `miss_forest` es el más preciso pero lento; `mice_hgb` equilibra precisión y tiempo; imputación simple es rápida pero menos precisa.
