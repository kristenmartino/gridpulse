"""
Airflow DAG — WattCast Model Training Pipeline (Daily).

Runs daily at 02:00 UTC:
    1. Ingest latest data
    2. Train all 3 models (XGBoost, Prophet, ARIMA) for all 8 regions
    3. Persist models to disk via joblib
    4. Score with freshly trained models and cache results

CADENCE: Daily at 02:00 UTC. Model weights don't change meaningfully
hour-to-hour when training on a 365-day window. Adding 24 new hours
to 8,760 is noise. Daily retraining is standard practice.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── DAG Configuration ───────────────────────────
default_args = {
    "owner": "wattcast",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    dag_id="wattcast_training_pipeline",
    default_args=default_args,
    description="Daily model training: train all models, persist to disk, score, cache",
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["wattcast", "ml", "training"],
)


# ── Task Functions ──────────────────────────────
def ingest_weather():
    """Pull weather data from APIs -> Kafka topic."""
    from src.ingestion.weather_producer import run
    run()


def ingest_grid_demand():
    """Pull grid demand data from EIA -> Kafka topic."""
    from src.ingestion.grid_producer import run
    run()


def consume_to_feature_store():
    """Kafka topics -> Postgres feature store."""
    from src.processing.kafka_consumer import run
    run()


def train_score_and_cache():
    """Train all models, persist to disk, score, write to Redis + Postgres."""
    from src.processing.batch_scorer import run
    run(mode="train")


def log_pipeline_run():
    """Record training pipeline run metadata."""
    from src.observability import PipelineLogger
    pl = PipelineLogger("wattcast_training_completion")
    pl.step("dag_completed", status="success", mode="train")
    try:
        import psycopg2
        from src.config import DatabaseConfig
        conn = psycopg2.connect(DatabaseConfig().url)
        pl.persist(conn)
        conn.close()
    except Exception:
        pass


# ── Tasks ───────────────────────────────────────
t_weather = PythonOperator(
    task_id="ingest_weather",
    python_callable=ingest_weather,
    dag=dag,
)

t_grid = PythonOperator(
    task_id="ingest_grid_demand",
    python_callable=ingest_grid_demand,
    dag=dag,
)

t_consume = PythonOperator(
    task_id="consume_to_feature_store",
    python_callable=consume_to_feature_store,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_score_and_cache",
    python_callable=train_score_and_cache,
    dag=dag,
    execution_timeout=timedelta(hours=2),  # Training can take a while
)

t_log = PythonOperator(
    task_id="log_pipeline_run",
    python_callable=log_pipeline_run,
    dag=dag,
    trigger_rule="all_done",
)


# ── Dependencies ────────────────────────────────
#
#   ingest_weather --+
#                     +--> consume --> train_score_and_cache --> log
#   ingest_grid    --+
#
[t_weather, t_grid] >> t_consume >> t_train >> t_log
