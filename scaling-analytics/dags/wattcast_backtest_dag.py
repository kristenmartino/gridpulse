"""
Airflow DAG — WattCast Backtest Pipeline (Daily).

Runs daily at 03:00 UTC (after training completes at ~02:00):
    1. Load freshly trained models from disk
    2. Run backtests at 3 horizons (24h, 168h, 720h) for all 8 regions
    3. Cache backtest results to Redis

CADENCE: Daily at 03:00 UTC. Backtests only need to update when models
change. Running them hourly is wasteful — the holdout metrics don't
change if the model hasn't been retrained.
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
    dag_id="wattcast_backtest_pipeline",
    default_args=default_args,
    description="Daily backtests: load trained models, run holdout validation, cache results",
    schedule_interval="0 3 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["wattcast", "ml", "backtest", "validation"],
)


# ── Task Functions ──────────────────────────────
def run_backtests():
    """Load persisted models and run backtests for all regions."""
    from src.processing.batch_scorer import run

    run(mode="backtest")


def log_pipeline_run():
    """Record backtest pipeline run metadata."""
    from src.observability import PipelineLogger

    pl = PipelineLogger("wattcast_backtest_completion")
    pl.step("dag_completed", status="success", mode="backtest")
    try:
        import psycopg2
        from src.config import DatabaseConfig

        conn = psycopg2.connect(DatabaseConfig().url)
        pl.persist(conn)
        conn.close()
    except Exception:
        pass


# ── Tasks ───────────────────────────────────────
t_backtest = PythonOperator(
    task_id="run_backtests",
    python_callable=run_backtests,
    dag=dag,
    execution_timeout=timedelta(hours=2),
)

t_log = PythonOperator(
    task_id="log_pipeline_run",
    python_callable=log_pipeline_run,
    dag=dag,
    trigger_rule="all_done",
)


# ── Dependencies ────────────────────────────────
t_backtest >> t_log
