"""
Kafka producer wrapping the v1 weather_client.py.

Calls fetch_weather() for each region and publishes the result
to the 'weather-raw' Kafka topic as JSON messages.
"""

from __future__ import annotations

import json
import logging

from confluent_kafka import Producer

from src.config import GRID_REGIONS, REGION_COORDINATES, KafkaConfig

logger = logging.getLogger(__name__)


def _delivery_callback(err, msg):
    if err is not None:
        logger.error("Weather message delivery failed: %s", err)


def produce_weather(config: KafkaConfig | None = None):
    """Fetch weather for all regions and publish to Kafka."""
    # v1 import — resolved via sys.path set in src/__init__.py
    from data.weather_client import fetch_weather

    config = config or KafkaConfig()
    producer = Producer(
        {
            "bootstrap.servers": config.bootstrap_servers,
            "client.id": "wattcast-weather-producer",
        }
    )

    for region in GRID_REGIONS:
        if region not in REGION_COORDINATES:
            logger.warning("Skipping %s: not in REGION_COORDINATES", region)
            continue
        try:
            df = fetch_weather(region, use_cache=False)
            if df.empty:
                logger.warning("No weather data for %s", region)
                continue

            for _, row in df.iterrows():
                message = {
                    "region": region,
                    "timestamp": str(row["timestamp"]),
                }
                for col in df.columns:
                    if col != "timestamp":
                        val = row[col]
                        # Convert numpy types to native Python for JSON
                        message[col] = float(val) if hasattr(val, "item") else val

                producer.produce(
                    topic=config.weather_topic,
                    key=region.encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                    callback=_delivery_callback,
                )

            producer.flush()
            logger.info("Published %d weather records for %s", len(df), region)

        except Exception:
            logger.exception("Weather producer failed for %s", region)

    producer.flush()


def run():
    """Entry point called by Airflow task."""
    produce_weather()
