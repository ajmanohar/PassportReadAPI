"""
SQLite database repository for asynchronous Form-C batch extraction jobs.
Manages disk persistence of job statuses, input document paths, and fused extraction results.
Provides automatic crash recovery to reset interrupted jobs on system reboots.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Central path configuration
DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(DATABASE_DIR, "jobs.db")


def get_db_connection() -> sqlite3.Connection:
    """
    Creates and returns a new SQLite connection with row factories enabled
    for dictionary-style key-value access.
    """
    connection = sqlite3.connect(DATABASE_PATH, timeout=15.0)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """
    Initializes the SQLite database file and creates the jobs table schema
    along with indexing on status and creation timestamps.
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                doc_paths_json TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
        conn.commit()


def create_job(job_id: str, doc_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Registers a new asynchronous job with status QUEUED and records local document paths.

    :param job_id: Unique UUID string identifier.
    :param doc_paths: Dictionary mapping document roles to local storage paths.
    :return: Record metadata dictionary containing job_id, status, and creation timestamp.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    doc_paths_encoded = json.dumps(doc_paths)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message)
            VALUES (?, 'QUEUED', ?, ?, ?, NULL, NULL)
            """,
            (job_id, now_iso, now_iso, doc_paths_encoded)
        )
        conn.commit()

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": now_iso
    }


def update_job_status(job_id: str, status: str, error_message: Optional[str] = None) -> None:
    """
    Updates the lifecycle status of a job (e.g., transitioning to PROCESSING or FAILED).

    :param job_id: Target UUID string.
    :param status: New status state (QUEUED, PROCESSING, COMPLETED, FAILED).
    :param error_message: Optional error description string if job failed.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, error_message = ?
            WHERE job_id = ?
            """,
            (status, now_iso, error_message, job_id)
        )
        conn.commit()


def save_job_result(job_id: str, result_data: Dict[str, Any]) -> None:
    """
    Saves the final fused Form-C extraction data and marks the job as COMPLETED.

    :param job_id: Target UUID string.
    :param result_data: Fused dictionary matching the GuestFormCRecord schema.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    result_encoded = json.dumps(result_data)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE jobs
            SET status = 'COMPLETED', updated_at = ?, result_json = ?, error_message = NULL
            WHERE job_id = ?
            """,
            (now_iso, result_encoded, job_id)
        )
        conn.commit()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single job record by UUID identifier and decodes stored JSON attributes.

    :param job_id: Target UUID string.
    :return: Job dictionary representation or None if not found.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "doc_paths": json.loads(row["doc_paths_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_message": row["error_message"]
        }


def list_recent_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent jobs ordered chronologically descending for synchronization.

    :param limit: Maximum number of rows to retrieve.
    :return: List of job summary dictionaries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id, status, created_at, updated_at, result_json, error_message
            FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        jobs_list = []
        for row in rows:
            jobs_list.append({
                "job_id": row["job_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "error_message": row["error_message"]
            })
        return jobs_list


def reset_interrupted_jobs() -> List[str]:
    """
    Crash recovery utility invoked on server startup.
    Finds any jobs stranded in PROCESSING state during unexpected power outages
    and resets them to QUEUED so the background task picks them up again.

    :return: List of recovered job UUIDs.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id FROM jobs WHERE status = 'PROCESSING'
            """
        )
        rows = cursor.fetchall()
        recovered_ids = [row["job_id"] for row in rows]

        if recovered_ids:
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'QUEUED', updated_at = ?
                WHERE status = 'PROCESSING'
                """,
                (now_iso,)
            )
            conn.commit()

        return recovered_ids


# Ensure database tables exist upon module import
initialize_database()