from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import psycopg2
from psycopg2.extras import Json


@dataclass(frozen=True)
class DBConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    dbname: str = os.getenv("PGDATABASE", "vehicle_triage")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "")


class PostgresDB:
    def __init__(self):
        self.cfg = DBConfig()
        self.conn = psycopg2.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            dbname=self.cfg.dbname,
            user=self.cfg.user,
            password=self.cfg.password,
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def create_table(self) -> None:
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_results (
                id SERIAL PRIMARY KEY,
                image_path TEXT NOT NULL,
                split TEXT,
                variant TEXT,
                det_conf REAL,
                plate_bbox JSONB,
                p_blur REAL,
                blur_threshold REAL,
                blur_decision BOOLEAN,
                ocr_text TEXT,
                ocr_conf REAL,
                final_outcome TEXT NOT NULL,
                manual_reason TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

    def insert_result(self, row: Mapping[str, Any]) -> None:
        bbox = row.get("plate_bbox")
        payload = dict(row)
        payload["plate_bbox"] = Json(bbox) if bbox is not None else None

        self.cur.execute(
            """
            INSERT INTO image_results (
                image_path, split, variant, det_conf, plate_bbox,
                p_blur, blur_threshold, blur_decision,
                ocr_text, ocr_conf, final_outcome, manual_reason
            )
            VALUES (
                %(image_path)s, %(split)s, %(variant)s, %(det_conf)s, %(plate_bbox)s,
                %(p_blur)s, %(blur_threshold)s, %(blur_decision)s,
                %(ocr_text)s, %(ocr_conf)s, %(final_outcome)s, %(manual_reason)s
            );
            """,
            payload,
        )

    def close(self) -> None:
        try:
            self.cur.close()
        finally:
            self.conn.close()
