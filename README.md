# Clasificación automática de quejas financieras de consumidores

**Curso:** Aprendizaje Supervisado — Proyecto final
**Autor:** Jorge Cunalata
**Notebook:** [`Proyecto_Final_Quejas_CFPB.ipynb`](Proyecto_Final_Quejas_CFPB.ipynb)

**Pipeline:** Texto → limpieza + lematización + stop words → Bag of Words TF‑IDF → Regresión Logística / Naive Bayes Multinomial
**Dataset:** CFPB Consumer Complaint Database (narrativas de consumidores, 2019‑2024)

---

## 1. Resumen

Las entidades financieras y los reguladores reciben miles de quejas escritas en lenguaje natural que deben **enrutarse al área de producto correcta**. Hacerlo manualmente aumenta los tiempos de respuesta, consume horas de analistas y genera errores de enrutamiento.

Este proyecto construye un clasificador multiclase que, a partir del texto libre de una queja, predice el **producto financiero** al que se refiere (5 clases: *Credit reporting*, *Debt collection*, *Mortgage*, *Credit card*, *Bank account*). Se comparan dos técnicas con el mismo pipeline de `scikit-learn`:

* **Regresión Logística multinomial** (modelo discriminativo, regularización L2 con λ = 1/C).
* **Naive Bayes Multinomial** (modelo generativo, suavizado α).

**Resultado principal:** la Regresión Logística es **significativamente mejor** que Naive Bayes (macro‑F1 0.826 vs 0.806 en 10×10 *repeated k‑fold*; Wilcoxon p = 1.96·10⁻¹⁶, t‑test corregido p = 1.73·10⁻⁴) y lo confirma en el conjunto de test (macro‑F1 0.823 vs 0.802; log‑loss 0.273 vs 0.358).

---

## 2. Resultados de la ejecución entregada

Ejecución completa en Google Colab con los parámetros por defecto (`N_SAMPLE = 15000`, comparación 10×10, 12 *trials* de Optuna).

### 2.1 Datos

| Concepto | Valor |
|---|---|
| Quejas filtradas en la población 2019‑2024 (5 productos, con narrativa) | 2 011 276 |
| Muestra para modelar (tras limpieza) | 14 996 quejas |
| Train / test (estratificado 80/20) | 11 996 / 3 000 |
| Clase mayoritaria / minoritaria | *Credit reporting* 73.3 % / *Mortgage* 3.8 % |
| Ratio de desbalance | 19.1 : 1 |

### 2.2 Respuestas del EDA

| Pregunta del enunciado | Resultado | Decisión de diseño |
|---|---|---|
| ¿La data está balanceada? | **No** (ratio 19.1:1) | `StratifiedKFold` repetido, macro‑F1 como métrica principal, *split* estratificado |
| ¿Se necesitan transformaciones no lineales? | **Sí**: skew de la longitud = 7.24 → −0.08 con `log1p` | `sublinear_tf=True` (1 + log tf) e IDF logarítmico |
| ¿Está normalizada? | Tras TF‑IDF con `norm='l2'` **sí** (norma media = 1.000, std = 0.000; BoW: 27.8 ± 32.9) | No se usa `StandardScaler` |
| ¿Hay estacionalidad? | Estacionalidad **débil** (F_s = 0.39) y tendencia **fuerte** (F_t = 0.97) | *Split* aleatorio estratificado; en producción re‑entrenar periódicamente (*prior shift*) |

### 2.3 Hiperparámetros

| Modelo | Búsqueda | Mejor valor (usado en los modelos finales) | CV log‑loss |
|---|---|---|---|
| Regresión Logística | Grid λ ∈ [10⁻³, 10²] (11 valores) | λ = 0.1 (C = 10) | 0.2964 |
| Regresión Logística | **Optuna** (λ + `min_df`, n‑gramas, `sublinear_tf`) | **λ = 0.1459 (C = 6.855)**, `min_df=6`, `ngram_range=(1, 2)`, `sublinear_tf=True` | **0.2956** |
| Naive Bayes | Grid α ∈ [10⁻⁴, 10¹] sobre el TF‑IDF final | **α = 0.3162** | 0.3967 |

Optuna mejoró ligeramente al grid, por lo que la sección 6.3 adoptó sus hiperparámetros. Naive Bayes usa **el mismo TF‑IDF** para que la comparación sea justa.

### 2.4 Curva de aprendizaje

| Modelo | Training loss | CV loss | Gap | Diagnóstico |
|---|---|---|---|---|
| Regresión Logística | 0.092 | 0.296 | 0.204 | *Good fit* (alcanza el objetivo log‑loss ≤ 0.30; la CV loss aún desciende con más datos) |
| Naive Bayes | 0.295 | 0.397 | 0.102 | *Good fit* (no alcanza el objetivo) |

### 2.5 Comparación estadística (10 folds × 10 repeticiones = 100 pares)

| Métrica (media ± std) | Regresión Logística | Naive Bayes |
|---|---|---|
| macro‑F1 | **0.8257 ± 0.0172** | 0.8057 ± 0.0144 |
| Exactitud | **0.9007 ± 0.0080** | 0.8854 ± 0.0079 |
| Log‑loss | **0.2923 ± 0.0179** | 0.3978 ± 0.0312 |

| Prueba (H₀: diferencia de macro‑F1 = 0, bilateral, α = 0.05) | p‑valor | Decisión |
|---|---|---|
| Wilcoxon signed‑rank | 1.96·10⁻¹⁶ | Se rechaza H₀ |
| t‑test pareado corregido (Nadeau & Bengio) | 1.73·10⁻⁴ | Se rechaza H₀ |
| t‑test pareado clásico (solo referencia, optimista) | 2.37·10⁻²⁴ | — |

LR gana en **92 de 100 folds** (diferencia media +0.0201; tamaño de efecto *rank‑biserial* r = 0.95).

### 2.6 Evaluación final en test (3 000 quejas, nunca usadas antes)

| Modelo | macro‑F1 | Exactitud | Log‑loss |
|---|---|---|---|
| **Regresión Logística** | **0.8233** | **0.9073** | **0.2730** |
| Naive Bayes | 0.8023 | 0.8893 | 0.3584 |

* Mayor confusión (LR): **Debt collection → Credit reporting** (30.1 % de las quejas de cobranza), porque las deudas en cobranza aparecen en el reporte de crédito y comparten vocabulario.
* t‑SNE sobre el test (SVD‑50 → 2D): silhouette por label = 0.057; las clases forman regiones propias pero se solapan en las fronteras, donde se concentran los errores.

### 2.7 Conclusión y recomendación

Se recomienda la **Regresión Logística con TF‑IDF (1‑2 gramas)**: es significativamente mejor que Naive Bayes en validación cruzada y en test, entrega probabilidades mejor calibradas (menor log‑loss), entrena en segundos y es interpretable (términos con mayor peso por clase). Las quejas en la frontera *Debt collection / Credit reporting* con probabilidad máxima baja deberían derivarse a revisión humana. Las conclusiones completas se generan automáticamente en la sección 11 del notebook a partir de los resultados.

---

## 3. Cumplimiento del enunciado

| Requisito | Dónde está en el notebook |
|---|---|
| Introducción: motivación del problema e importancia | Sección 1 |
| EDA: ¿balanceada? ¿normalizada? ¿estacionalidad? ¿skewness? | Secciones 3.1 a 3.4 |
| Texto: lematización y *stop words* + word cloud y palabras frecuentes | Secciones 4 y 4.1 |
| Sin selección de características (texto con TF‑IDF) | Nota de diseño en la sección 1 |
| Pipelines de sklearn: `Texto → TF‑IDF → modelo` | Sección 5 |
| Optimización de hiperparámetros con *repeated k‑fold CV* + gráfica de λ (training vs validación) | Secciones 6 y 6.1 |
| Opcional: Optuna | Sección 6.2 |
| Uso del mejor hiperparámetro en los modelos finales | Sección 6.3 |
| Curva de aprendizaje (CV loss vs training loss y gap) | Sección 7 |
| Comparación de dos técnicas con test de hipótesis (Wilcoxon / t‑test) en 10×10 *repeated k‑fold* | Sección 8 |
| Visualización con t‑SNE (color = label) | Sección 10 |
| Referencias en formato IEEE | Sección 12 |
| Prompts usados (uso de IA generativa) | Anexo |
| Conclusiones a partir de los resultados (no teoría) | Sección 11 |

---

## 4. Estructura del proyecto

```
aprendiza_supervisado_proyecto_final/
├── Proyecto_Final_Quejas_CFPB.ipynb   # notebook entregable (con las salidas de la ejecución)
├── README.md                          # este archivo
└── dataset/                           # se crea en la primera corrida (no se versiona)
    ├── complaints.csv.zip             # archivo de la CFPB (~1.17 GB)
    ├── cfpb_sample_n15000_k0.05.parquet
    ├── cfpb_monthly.parquet
    └── cfpb_lemmas_n15000_k0.05.parquet
```

---

## 5. Dataset

* **Fuente:** Consumer Financial Protection Bureau (CFPB), *Consumer Complaint Database*.
* **Archivo utilizado:** la copia del archivo oficial `complaints.csv.zip` del **31 de mayo de 2025**, preservada en Internet Archive: <https://web.archive.org/web/20250531184504/https://files.consumerfinance.gov/ccdb/complaints.csv.zip> (1.17 GB comprimido, 5.9 GB descomprimido).
* **¿Por qué no la descarga oficial actual?** Desde 2026 la CFPB ya no publica la columna *Consumer complaint narrative* (ni en el archivo masivo ni en su API), y sin narrativas no existe el problema de texto. La copia archivada conserva las narrativas, cubre completo el periodo 2019‑2024 y, al ser un archivo **fijo**, garantiza que cualquier ejecución obtenga la misma muestra.
* **Filtros aplicados:** quejas con narrativa, recibidas entre 2019‑01‑01 y 2024‑12‑31, de 5 productos (se agrupan los nombres que la CFPB cambió en el tiempo) y sin tarjetas *prepaid*.
* **Muestra:** se guarda aleatoriamente el 5 % de las narrativas (`KEEP_FRAC`) y de ahí se toma una muestra aleatoria simple de 15 000 quejas (`N_SAMPLE`), que conserva el desbalance real. La serie mensual de **todas** las quejas filtradas se usa para el análisis de estacionalidad.

---

## 6. Cómo ejecutar

### Opción A — Google Colab en el navegador (recomendada)

1. Abrir el notebook en Colab directamente desde este repositorio: <https://colab.research.google.com/github/jorgecunalata/aprendiza_supervisado_proyecto_final/blob/main/Proyecto_Final_Quejas_CFPB.ipynb> (o descargar `Proyecto_Final_Quejas_CFPB.ipynb` y subirlo con *File → Upload notebook*).
2. Ejecutar *Runtime → Run all*.
3. En la celda de parámetros, Colab pide **autorizar el acceso a Google Drive**: la carpeta `dataset` se crea en `MyDrive/proyecto_cfpb/dataset` y se conserva entre sesiones. Si no se autoriza, el notebook continúa con una carpeta local `/content/dataset` (se borra al cerrar el entorno).

No requiere instalar nada: la primera celda instala lo que Colab no trae (`wordcloud`, `optuna`, `spacy` + `en_core_web_sm`, `pyarrow`).

### Opción B — VS Code con el kernel de Colab

1. Instalar la extensión oficial **Google Colab** para VS Code (versión ≥ 0.2.1, que permite montar Drive).
2. Abrir el notebook, *Select Kernel → Colab → Python 3* e iniciar sesión con la cuenta de Google.
3. Ejecutar todas las celdas. El código corre en la máquina de Colab, no en el equipo local.

Para montar Drive, la cuenta de Google del Drive debe ser **la misma** con la que se inició sesión en la extensión; si no, el montaje falla y el notebook usa `/content/dataset`.

### Opción C — Local (VS Code / Jupyter)

```bash
git clone https://github.com/jorgecunalata/aprendiza_supervisado_proyecto_final.git
cd aprendiza_supervisado_proyecto_final
python3.12 -m venv .venv            # o: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install scikit-learn pandas numpy scipy matplotlib seaborn statsmodels \
            spacy wordcloud optuna pyarrow joblib ipykernel
python -m spacy download en_core_web_sm
```

En VS Code, seleccionar el kernel `.venv` de la carpeta del proyecto y ejecutar todas las celdas. La carpeta `dataset/` se crea automáticamente junto al notebook. Se necesitan ~1.5 GB libres en disco.

### Tiempos aproximados (ejecución completa)

| Etapa | Primera corrida | Siguientes corridas |
|---|---|---|
| Descarga del ZIP (1.17 GB desde Internet Archive) | ~10‑30 min (según la red) | 0 (se reutiliza) |
| Lectura del ZIP por bloques | ~2 min | 0 (se cargan los `parquet`) |
| Lematización con spaCy | varios minutos | 0 (se carga el `parquet` de lemas) |
| Grid de λ (5×2 CV) | ~9 min | ~9 min |
| Comparación estadística 10×10 | ~9 min | ~9 min |
| Optuna, curvas, test, t‑SNE | ~10‑15 min | ~10‑15 min |

---

## 7. Reproducibilidad

| Aspecto | Cómo se garantiza |
|---|---|
| Datos | Snapshot **fijo** del archivo oficial (Internet Archive, 31‑may‑2025); no cambia día a día |
| Descarga única | El ZIP se guarda en `dataset/` y se reutiliza; la descarga es reanudable si se corta |
| Caché de datos procesados | Los `parquet` de muestra, serie y lemas se guardan en `dataset/`; sus nombres incluyen `N_SAMPLE` y `KEEP_FRAC`, así un cambio de parámetros nunca reutiliza una muestra equivocada |
| Semillas | `SEED = 42` en *split*, validación cruzada, SVD, t‑SNE, Optuna (TPE) y word clouds |
| Generadores aleatorios | La carga de datos usa su propio generador; el generador global se usa solo en t‑SNE. Los resultados son **idénticos** con o sin caché (verificado) |
| Sin fuga de información | El TF‑IDF y los modelos van dentro de `sklearn.pipeline.Pipeline` y se ajustan solo con el *fold* de entrenamiento; el test se usa una sola vez (sección 9); t‑SNE se aplica después del modelo, solo para visualizar |
| Hiperparámetros | Elegidos solo con validación cruzada sobre train; si el óptimo cae en el borde de la grilla, la grilla se amplía automáticamente |

**Versiones probadas en local** (Python 3.12.14):

| Paquete | Versión | Paquete | Versión |
|---|---|---|---|
| scikit-learn | 1.9.1 | spacy / en_core_web_sm | 3.8.18 / 3.8.0 |
| pandas | 3.1.1 | optuna | 5.1.0 |
| numpy | 2.5.5 | statsmodels | 0.15.0 |
| scipy | 1.18.1 | wordcloud | 1.9.6 |
| matplotlib | 3.11.2 | pyarrow | 26.0.1 |
| seaborn | 0.13.2 | joblib | 1.6.0 |

En Colab se usan las versiones preinstaladas del entorno. Diferencias de versión (sobre todo en scikit-learn, spaCy y Optuna) pueden cambiar levemente el t‑SNE, los *trials* de Optuna y los decimales de las métricas, pero no las conclusiones.

### Parámetros configurables (variables de entorno)

| Variable | Por defecto | Uso |
|---|---|---|
| `DATASET_DIR` | `dataset` (en Colab: Drive) | Carpeta del ZIP y de los `parquet` |
| `CFPB_ZIP` | `<DATASET_DIR>/complaints.csv.zip` | Ruta de un ZIP ya descargado |
| `CFPB_URL` | Snapshot de Internet Archive | Origen de la descarga |
| `N_SAMPLE` | 15000 | Nº de quejas para modelar |
| `CFPB_KEEP_FRAC` | 0.05 | Fracción de narrativas que se guardan al leer el CSV |
| `FAST_MODE` | 0 | 1 → comparación 5×2 en lugar de 10×10 (solo pruebas) |
| `OPTUNA_TRIALS` | 12 | Nº de *trials* de Optuna |

Para la entrega se usan los valores por defecto.

---

## 8. Problemas conocidos

| Síntoma | Causa | Solución |
|---|---|---|
| `Usecols do not match columns ... 'Consumer complaint narrative'` | Se usó la descarga oficial actual de la CFPB, que ya no trae narrativas | El notebook usa el snapshot de Internet Archive y detecta/reemplaza un ZIP sin narrativas automáticamente |
| `No se pudo montar Google Drive ... Credentials propagation unsuccessful` | La cuenta del Drive no es la misma con la que se inició sesión en Colab / extensión de VS Code | Usar la misma cuenta; el notebook continúa con `/content/dataset` |
| La descarga se interrumpe | Corte de red | Volver a ejecutar la celda de la sección 2: la descarga continúa desde donde quedó |
| Gráficos con fondo gris después de Optuna | Optuna activa el estilo `ggplot` | Ya corregido: se restaura el estilo por defecto tras Optuna |

---

## 9. Uso de IA generativa

Siguiendo la indicación del curso, el código se desarrolló con apoyo de un asistente de IA generativa. El **Anexo** al final del notebook documenta el proceso de desarrollo: las decisiones y pruebas realizadas, los *prompts* utilizados por sección y las iteraciones de corrección (problema encontrado y arreglo aplicado).

## 10. Referencias

Las 25 referencias en formato IEEE están en la sección 12 del notebook. Las principales:

* Consumer Financial Protection Bureau, "Consumer Complaint Database." <https://www.consumerfinance.gov/data-research/consumer-complaints/>
* F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011.
* A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," AAAI‑98 Workshop, 1998.
* C. Nadeau and Y. Bengio, "Inference for the generalization error," *Mach. Learn.*, vol. 52, 2003.
* J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *JMLR*, vol. 7, 2006.
* L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *JMLR*, vol. 9, 2008.
* T. Akiba *et al.*, "Optuna: A next-generation hyperparameter optimization framework," KDD 2019.
* R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., 2021.
