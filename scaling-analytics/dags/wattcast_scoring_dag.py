"""
Airflow DAG — WattCast Scoring Pipeline (Hourly).

Runs inference every hour at :05 past using persisted models:
    1. Check data freshness (skip if no new data)
    2. Ingest weather data (Kafka producer)
    3. Ingest grid demand data (Kafka producer)
    4. Consume from Kafka -> write to Postgres feature store
    5. Load persisted models -> score -> write predictions to Redis
    6. Log pipeline run

CADENCE: Hourly at :05 — matches EIA + Open-Meteo hourly publication.
Running at :05 ensures the :00 data has landed.

This DAG does NOT train models. Training is done daily by wattcast_training_dag.
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="wattcast_scoring_pipeline",
    default_args=default_args,
    description="Hourly inference: load models, score regions, cache forecasts",
    schedule_interval="5 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["wattcast", "ml", "forecasting", "scoring"],
)


# ── Task Functions ──────────────────────────────
def check_freshness(**context):
    """Check if new data exists since last scoring run."""
    try:
        import psycopg2
        from src.config import DatabaseConfig
        from src.processing.freshness_checker import FreshnessChecker

        conn = psycopg2.connect(DatabaseConfig().url)
        checker = FreshnessChecker(conn)
        should_score = checker.should_score()
        conn.close()

        if not should_score:
            # Push skip signal to XCom
            context["ti"].xcom_push(key="skip_scoring", value=True)
    except Exception:
        # On error, proceed with scoring (conservative)
        pass


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


def score_and_cache(**context):
    """Load persisted models -> Score -> Write to Redis + Postgres."""
    # Check if freshness checker said to skip
    skip = context.get("ti", None)
    if skip:
        skip_val = skip.xcom_pull(key="skip_scoring", default=False)
        if skip_val:
            import logging
            logging.getLogger(__name__).info("Skipping scoring — no new data")
            return

    from src.processing.batch_scorer import run
    run(mode="inference")


def record_freshness():
    """Record that we've processed data up to the current timestamp."""
    try:
        import psycopg2
        from src.config import DatabaseConfig
        from src.processing.freshness_checker import FreshnessChecker

        conn = psycopg2.connect(DatabaseConfig().url)
        checker = FreshnessChecker(conn)
        checker.record_check("demand")
        checker.record_check("weather")
        conn.close()
    except Exception:
        pass


def log_pipeline_run():
    """Record pipeline run metadata for observability."""
    from src.observability import PipelineLogger
    pl = PipelineLogger("wattcast_scoring_completion")
    pl.step("dag_completed", status="success", mode="inference")
    try:
        import psycopg2
        from src.config import DatabaseConfig
        conn = psycopg2.connect(DatabaseConfig().url)
        pl.persist(conn)
        conn.close()
    except Exception:
        pass


# ── Tasks ───────────────────────────────────────
t_freshness = PythonOperator(
    task_id="check_freshness",
    python_callable=check_freshness,
    provide_context=True,
    dag=dag,
)

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

t_score = PythonOperator(
    task_id="score_and_cache",
    python_callable=score_and_cache,
    provide_context=True,
    dag=dag,
)

t_record = PythonOperator(
    task_id="record_freshness",
    python_callable=record_freshness,
    dag=dag,
)

t_log = PythonOperator(
    task_id="log_pipeline_run",
    python_callable=log_pipeline_run,
    dag=dag,
    trigger_rule="all_done",
)


# ── Dependencies ────────────────────────────────
#
#   check_freshness --+
#                      |
#   ingest_weather  --+--> consume --> score --> record_freshness --> log
#   ingest_grid     --+
#
t_freshness >> [t_weather, t_grid]
[t_weather, t_grid] >> t_consume >> t_score >> t_record >> t_log
