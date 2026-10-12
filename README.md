# Clasificador como servicio web con FastAPI

Modelo de clasificación (Random Forest sobre el dataset Iris) desplegado como API REST con FastAPI,
con validación de datos, registro de eventos (logging), métricas básicas y pruebas automáticas.

## Estructura
```
app/main.py        -> API: endpoints, logging y métricas
app/schemas.py     -> Validación de entrada/salida (Pydantic)
train_model.py     -> Entrena y guarda el modelo
tests/test_api.py  -> Pruebas automáticas (pytest)
notebook.ipynb     -> Documentación paso a paso (Google Colab)
capturas/          -> Capturas de pantalla de las pruebas
```

## Instalación y ejecución
```bash
pip install -r requirements.txt
python train_model.py                 # genera models/iris_model.joblib
uvicorn app.main:app --reload         # inicia el servidor en http://127.0.0.1:8000
```
Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Estado del servicio y del modelo |
| GET | `/model-info` | Tipo de modelo, clases y accuracy |
| GET | `/metrics` | Contadores: peticiones, predicciones, errores, latencia media |
| POST | `/predict` | Predicción para una muestra |
| POST | `/predict/batch` | Predicción para varias muestras |

## Ejemplo con curl
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Respuesta:
```json
{"prediction":"setosa","confidence":1.0,"probabilities":{"setosa":1.0,"versicolor":0.0,"virginica":0.0}}
```

## Pruebas
```bash
pytest -v
```

## Logs
Se guardan en `logs/api.log` (rotativo). Cada petición registra: id, método, ruta, código HTTP y tiempo.
Los códigos 4xx se marcan como `WARNING` y los fallos internos como `ERROR` con traza completa.
Este proyecto se desarrolló con apoyo de una herramienta de IA para la estructura del código y la documentación; las pruebas y capturas si son propias
