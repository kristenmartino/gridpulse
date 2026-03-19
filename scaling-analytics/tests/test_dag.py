"""Tests for the Airflow DAG structures."""
import ast
from pathlib import Path

import pytest


DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"


class TestScoringDag:

    def test_dag_loads_without_error(self):
        """Scoring DAG Python file parses without import/syntax errors."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)
        assert tree is not None

    def test_dag_has_task_functions(self):
        """Scoring DAG defines expected task functions."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)

        function_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "check_freshness" in function_names
        assert "ingest_weather" in function_names
        assert "ingest_grid_demand" in function_names
        assert "consume_to_feature_store" in function_names
        assert "score_and_cache" in function_names
        assert "record_freshness" in function_names
        assert "log_pipeline_run" in function_names

    def test_dag_id_is_correct(self):
        """Scoring DAG file contains the expected dag_id."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'dag_id="wattcast_scoring_pipeline"' in source

    def test_dag_schedule_is_hourly(self):
        """Scoring DAG runs hourly at :05 past."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert "5 * * * *" in source

    def test_dag_max_active_runs_is_one(self):
        """Only one scoring run at a time to prevent overlap."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert "max_active_runs=1" in source

    def test_scoring_dag_uses_inference_mode(self):
        """Scoring DAG calls run with mode='inference'."""
        dag_path = DAGS_DIR / "wattcast_scoring_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'mode="inference"' in source


class TestTrainingDag:

    def test_dag_loads_without_error(self):
        """Training DAG Python file parses without errors."""
        dag_path = DAGS_DIR / "wattcast_training_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)
        assert tree is not None

    def test_dag_has_task_functions(self):
        """Training DAG defines expected task functions."""
        dag_path = DAGS_DIR / "wattcast_training_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)

        function_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "ingest_weather" in function_names
        assert "ingest_grid_demand" in function_names
        assert "consume_to_feature_store" in function_names
        assert "train_score_and_cache" in function_names
        assert "log_pipeline_run" in function_names

    def test_dag_id_is_correct(self):
        """Training DAG has correct dag_id."""
        dag_path = DAGS_DIR / "wattcast_training_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'dag_id="wattcast_training_pipeline"' in source

    def test_dag_schedule_is_daily(self):
        """Training DAG runs daily at 02:00 UTC."""
        dag_path = DAGS_DIR / "wattcast_training_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert "0 2 * * *" in source

    def test_training_dag_uses_train_mode(self):
        """Training DAG calls run with mode='train'."""
        dag_path = DAGS_DIR / "wattcast_training_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'mode="train"' in source


class TestBacktestDag:

    def test_dag_loads_without_error(self):
        """Backtest DAG Python file parses without errors."""
        dag_path = DAGS_DIR / "wattcast_backtest_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)
        assert tree is not None

    def test_dag_has_task_functions(self):
        """Backtest DAG defines expected task functions."""
        dag_path = DAGS_DIR / "wattcast_backtest_dag.py"
        with open(dag_path) as f:
            source = f.read()
        tree = ast.parse(source)

        function_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "run_backtests" in function_names
        assert "log_pipeline_run" in function_names

    def test_dag_id_is_correct(self):
        """Backtest DAG has correct dag_id."""
        dag_path = DAGS_DIR / "wattcast_backtest_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'dag_id="wattcast_backtest_pipeline"' in source

    def test_dag_schedule_is_daily_after_training(self):
        """Backtest DAG runs daily at 03:00 UTC (after training)."""
        dag_path = DAGS_DIR / "wattcast_backtest_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert "0 3 * * *" in source

    def test_backtest_dag_uses_backtest_mode(self):
        """Backtest DAG calls run with mode='backtest'."""
        dag_path = DAGS_DIR / "wattcast_backtest_dag.py"
        with open(dag_path) as f:
            source = f.read()
        assert 'mode="backtest"' in source
