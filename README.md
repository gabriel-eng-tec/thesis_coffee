# Coffee Berry Counting Project (Tesis)

## Descripción
Proyecto de maestría en computación para conteo automático de bayas de café usando modelos de visión por computadora.

## Estructura
```
models/          → Arquitecturas de modelos
pipeline/        → Scripts de entrenamiento e inferencia
utils/           → Funciones de utilidad (pérdidas, preprocessing)
data/            → Datasets (generado dinámicamente)
results/         → Checkpoints, evaluaciones y visualizaciones
config.py        → Configuración centralizada (RUTAS, HIPERPARÁMETROS)
experiments_archive/  → Código experimental archivado
```

## Configuración

Todos los parámetros están centralizados en [config.py](config.py):

- **Rutas**: DATASET_COFFEE, CDMENET_CHECKPOINTS, YOLO_CHECKPOINTS, etc.
- **Hiperparámetros**: CDMENET_LEARNING_RATE, YOLO_BATCH_SIZE, etc.
- **Thresholds y weights**: DENSITY_THRESHOLDS, LOSS_WEIGHTS

### Editar configuración
Abre `config.py` y modifica los parámetros según necesites.

## Uso

### 1. Generar mapas de densidad (preprocessing)
```bash
python -m utils.density
```
Genera archivos `.h5` y JSON para entrenamiento.

### 2. Entrenar modelos
```bash
python pipeline/train_yolo.py
python pipeline/train_cdmenet.py
```

### 3. Evaluar modelos
```bash
python inference/evaluator.py  # (próximamente)
```

## Dependencias
Ver `requirements.txt` (crear si falta)

## Notas
- Los imports usan rutas relativas portátiles desde `config.py`
- Los modelos se guardan en `results/checkpoints/`
- Cada split (train/val/test) tiene su propio JSON de dataset
