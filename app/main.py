from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from inference.predictor import RegimePredictor
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.middleware.wsgi import WSGIMiddleware
from dash_app.app import app as dash_app
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regime-api")

# Create FastAPI app
app = FastAPI(
    title="Market Regime Forecasting API",
    description="Predicts next-day market regime using pre-trained model",
    version="1.0.0"
)

# Mount the Dash app at /dash
app.mount("/dash", WSGIMiddleware(dash_app.server))

# # Attach Prometheus instrumentation
# Instrumentator().instrument(app).expose(app) # collects basic http metrics (request count, request duration, response codes, etc.)
# Attach Prometheus instrumentation with explicit configuration
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

# Instrument and expose
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")


# let's add some other example custom metrics we want to track
prediction_requests = Counter("prediction_requests_total", "Total prediction requests")
inference_latency = Histogram("inference_latency_seconds", "Time spent on predictions")

# load the predictor
try:
    predictor = RegimePredictor()
    logger.info("RegimePredictor loaded successfully.")
except Exception as e:
    logger.error("Failed to initialize RegimePredictor", exc_info=True)
    raise e


# root
@app.get("/", tags=["info"])
# def root():
#     return {"message": "Welcome to the Market Regime Forecasting API. Visit /docs for Swagger UI."}
def read_root():
    return {
        "message": "Welcome to the Regime Forecasting API. Visit /docs for Swagger UI.",
        "dash_app": "/dash",
        "predict": "/predict",
        "metrics": "/metrics"
    }


# Explicit metrics endpoint (backup)
@app.get("/metrics", tags=["metrics"])
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Healthcheck endpoint
@app.get("/healthcheck", tags=["health"])
def healthcheck():
    return {"status": "ok"}

# Prediction endpoint
@app.get("/predict", tags=["inference"])
def predict():
    try:
        prediction_requests.inc()
        with inference_latency.time():
            prediction = predictor.predict()
        return {"prediction": int(prediction)}
    except Exception as e:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))