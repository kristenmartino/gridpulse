"""
Kafka consumer: reads from weather-raw and grid-demand-raw topics,
writes normalized data to Postgres (the feature store).
"""
from __future__ import annotations

import json
import logging

import psycopg2
from psycopg2.extras import execute_values
from confluent_kafka import Consumer, KafkaError

from src.config import KafkaConfig, DatabaseConfig

logger = logging.getLogger(__name__)

# Max messages to consume per Airflow run (every 30 min)
MAX_MESSAGES = 50_000
CONSUME_TIMEOUT_SECONDS = 30

# All 17 weather columns matching the raw_weather table schema
WEATHER_COLUMNS = [
    "region", "timestamp", "temperature_2m", "apparent_temperature",
    "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
    "wind_speed_80m", "wind_speed_120m", "wind_direction_10m",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "cloud_cover", "precipitation", "snowfall", "surface_pressure",
    "soil_temperature_0cm", "weather_code",
]


def consume_to_postgres(
    kafka_config: KafkaConfig | None = None,
    db_config: DatabaseConfig | None = None,
):
    """
    Consume messages from Kafka topics and write to Postgres.

    Two topics:
      weather-raw     -> raw_weather table
      grid-demand-raw -> raw_demand table
    """
    kafka_config = kafka_config or KafkaConfig()
    db_config = db_config or DatabaseConfig()

    consumer = Consumer({
        "bootstrap.servers": kafka_config.bootstrap_servers,
        "group.id": kafka_config.consumer_group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([kafka_config.weather_topic, kafka_config.grid_topic])

    conn = psycopg2.connect(db_config.url)
    weather_batch: list[dict] = []
    demand_batch: list[dict] = []
    msg_count = 0

    try:
        while msg_count < MAX_MESSAGES:
            msg = consumer.poll(timeout=CONSUME_TIMEOUT_SECONDS)
            if msg is None:
                break
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    break
                logger.error("Kafka error: %s", msg.error())
                continue

            topic = msg.topic()
            data = json.loads(msg.value().decode("utf-8"))
            msg_count += 1

            if topic == kafka_config.weather_topic:
                weather_batch.append(data)
            elif topic == kafka_config.grid_topic:
                demand_batch.append(data)

            # Flush in batches of 1000
            if len(weather_batch) >= 1000:
                _insert_weather(conn, weather_batch)
                weather_batch = []
            if len(demand_batch) >= 1000:
                _insert_demand(conn, demand_batch)
                demand_batch = []

        # Final flush
        if weather_batch:
            _insert_weather(conn, weather_batch)
        if demand_batch:
            _insert_demand(conn, demand_batch)

        logger.info("Consumed %d messages from Kafka", msg_count)

    finally:
        consumer.close()
        conn.close()


def _insert_weather(conn, batch: list[dict]):
    """Bulk insert weather records into Postgres."""
    rows = [
        tuple(record.get(c) for c in WEATHER_COLUMNS)
        for record in batch
    ]
    update_cols = [c for c in WEATHER_COLUMNS if c not in ("region", "timestamp")]
    sql = f"""
        INSERT INTO raw_weather ({', '.join(WEATHER_COLUMNS)})
        VALUES %s
        ON CONFLICT (region, timestamp)
        DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in update_cols)}
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    logger.info("Inserted %d weather records", len(rows))


def _insert_demand(conn, batch: list[dict]):
    """Bulk insert demand records into Postgres."""
    rows = [
        (r["region"], r["timestamp"], r.get("demand_mw"), r.get("forecast_mw"))
        for r in batch
    ]
    sql = """
        INSERT INTO raw_demand (region, timestamp, demand_mw, forecast_mw)
        VALUES %s
        ON CONFLICT (region, timestamp)
        DO UPDATE SET
            demand_mw = EXCLUDED.demand_mw,
            forecast_mw = EXCLUDED.forecast_mw
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    logger.info("Inserted %d demand records", len(rows))


def run():
    """Entry point called by Airflow task."""
    consume_to_postgres()
