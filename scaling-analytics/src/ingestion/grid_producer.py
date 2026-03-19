"""
Kafka producer wrapping the v1 eia_client.py.

Calls fetch_demand() for each region and publishes to the
'grid-demand-raw' Kafka topic as JSON messages.
"""
from __future__ import annotations

import json
import logging

from confluent_kafka import Producer

from src.config import KafkaConfig, GRID_REGIONS, REGION_COORDINATES

logger = logging.getLogger(__name__)


def _delivery_callback(err, msg):
    if err is not None:
        logger.error("Grid message delivery failed: %s", err)


def produce_grid_demand(config: KafkaConfig | None = None):
    """Fetch demand for all regions and publish to Kafka."""
    # v1 import — resolved via sys.path set in src/__init__.py
    from data.eia_client import fetch_demand

    config = config or KafkaConfig()
    producer = Producer({
        "bootstrap.servers": config.bootstrap_servers,
        "client.id": "wattcast-grid-producer",
    })

    for region in GRID_REGIONS:
        if region not in REGION_COORDINATES:
            logger.warning("Skipping %s: not in REGION_COORDINATES", region)
            continue
        try:
            df = fetch_demand(region, use_cache=False)
            if df.empty:
                logger.warning("No demand data for %s", region)
                continue

            for _, row in df.iterrows():
                demand_val = row.get("demand_mw")
                forecast_val = row.get("forecast_mw")
                message = {
                    "region": region,
                    "timestamp": str(row["timestamp"]),
                    "demand_mw": float(demand_val) if demand_val is not None else None,
                    "forecast_mw": float(forecast_val) if forecast_val is not None else None,
                }
                producer.produce(
                    topic=config.grid_topic,
                    key=region.encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                    callback=_delivery_callback,
                )

            producer.flush()
            logger.info("Published %d demand records for %s", len(df), region)

        except Exception:
            logger.exception("Grid producer failed for %s", region)

    producer.flush()


def run():
    """Entry point called by Airflow task."""
    produce_grid_demand()
