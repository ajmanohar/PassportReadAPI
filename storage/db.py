"""
==============================================================================
storage/db.py - PassportReadAPI
==============================================================================
SQLite database storage layer for PassportReadAPI.
Manages schema initialization, job lifecycle state transitions, document path manifests,
progressive intermediate extraction persistence for asynchronous batch jobs,
startup recovery, and cascading job deletion.
==============================================================================
"""

import json
import logging
import os
import sqlite3
from typing import Dict, Any, Optional, List

# Central configuration settings
from config import settings

# Configure module-level logger
logger = logging.getLogger("storage.db")
logger.setLevel(logging.INFO)

# Absolute path for the SQLite database file on disk
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
DB_PATH = os.path.join(STORAGE_DIR, "jobs.db")


# ==============================================================================
# Database Connection Helper
# ==============================================================================

def get_db_connection() -> sqlite3.Connection:
    """
    Creates and returns a SQLite database connection configured with
    row factory for dictionary-like column access and immediate timeout.
    """
    os.makedirs(STORAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


# ==============================================================================
# Database Schema Initialization
# ==============================================================================

def init_db() -> None:
    """
    Initializes the SQLite schema for tracking batch extraction jobs.
    Creates the 'jobs' table and indexes if they do not already exist.
    """
    logger.info(f"[*] Initializing SQLite database schema at: {DB_PATH}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                doc_paths_json TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created 
            ON jobs(status, created_at DESC);
            """
        )
        conn.commit()
    logger.info("[*] SQLite database schema initialized successfully.")


# Convenience alias for application startup hooks
initialize_database = init_db


# ==============================================================================
# Job Lifecycle Management Functions
# ==============================================================================

def create_job(job_id: str, doc_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Inserts a newly ingested job with initial status 'QUEUED' and serializes
    the initial document paths manifest.

    :param job_id: Unique UUID identifier string for the job.
    :param doc_paths: Dictionary mapping document keys ('passport_bio', etc.) to disk paths.
    :return: Dictionary representation of the created job record.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        paths_json = json.dumps(doc_paths)
        cursor.execute(
            """
            INSERT INTO jobs (job_id, status, doc_paths_json, created_at, updated_at)
            VALUES (?, 'QUEUED', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """,
            (job_id, paths_json)
        )
        conn.commit()
        logger.info(f"[*] Created QUEUED database record for job: {job_id}")

    return get_job(job_id)


def update_job_status(job_id: str, new_status: str, error_message: Optional[str] = None) -> None:
    """
    Updates the execution state of an existing job (e.g., 'PROCESSING', 'FAILED').

    :param job_id: Unique job identifier string.
    :param new_status: Target status string ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED').
    :param error_message: Optional error diagnostics if transitioning to 'FAILED'.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE jobs
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP,
                error_message = ?
            WHERE job_id = ?;
            """,
            (new_status, error_message, job_id)
        )
        conn.commit()
        logger.info(f"[*] Job {job_id} transitioned status -> {new_status}")


def save_intermediate_job_result(job_id: str, result_dict: Dict[str, Any]) -> None:
    """
    Progressively persists partial extraction results into SQLite while keeping
    status as 'PROCESSING'. This enables real-time polling by mobile clients so
    individual stage badges can turn green before all stages finish.

    :param job_id: Unique job identifier string.
    :param result_dict: Serialized dictionary of the current GuestFormCRecord model.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        serialized_result = json.dumps(result_dict)
        cursor.execute(
            """
            UPDATE jobs
            SET result_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?;
            """,
            (serialized_result, job_id)
        )
        conn.commit()
        logger.info(f"[*] Saved progressive intermediate results to SQLite for job: {job_id}")


def save_job_result(job_id: str, result_dict: Dict[str, Any]) -> None:
    """
    Persists the final completed Form-C fused record and transitions status to 'COMPLETED'.

    :param job_id: Unique job identifier string.
    :param result_dict: Serialized dictionary of the full GuestFormCRecord model.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        serialized_result = json.dumps(result_dict)
        cursor.execute(
            """
            UPDATE jobs
            SET status = 'COMPLETED',
                result_json = ?,
                updated_at = CURRENT_TIMESTAMP,
                error_message = NULL
            WHERE job_id = ?;
            """,
            (serialized_result, job_id)
        )
        conn.commit()
        logger.info(f"[+] Persisted final extraction result for job: {job_id}")


def delete_job_record(job_id: str) -> bool:
    """
    Permanently deletes a job row from the SQLite database.
    Used during mobile-triggered submission deletion (Task 44).

    :param job_id: UUID of the job to remove.
    :return: True if a row was found and removed, False otherwise.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE job_id = ?;", (job_id,))
        conn.commit()
        deleted_count = cursor.rowcount
        logger.info(f"[*] Deleted job {job_id} from SQLite (rows affected: {deleted_count})")
        return deleted_count > 0


# ==============================================================================
# Job Querying & Retrieval Functions
# ==============================================================================

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single job row by ID and deserializes JSON manifests into Python structures.

    :param job_id: Target UUID string.
    :return: Dictionary representation of the job, or None if not found.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
            FROM jobs
            WHERE job_id = ?;
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
        "doc_paths": json.loads(row["doc_paths_json"]) if row["doc_paths_json"] else {},
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error_message": row["error_message"]
    }


def list_recent_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent jobs ordered by creation timestamp descending.

    :param limit: Maximum number of rows to return (default 50).
    :return: List of job record dictionaries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
            FROM jobs
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,)
        )
        rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "doc_paths": json.loads(row["doc_paths_json"]) if row["doc_paths_json"] else {},
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_message": row["error_message"]
        })
    return results


def get_all_pending_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves jobs currently in 'QUEUED' or 'PROCESSING' state for server restart recovery.

    :param limit: Maximum count of pending jobs to recover.
    :return: List of pending job dictionaries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
            FROM jobs
            WHERE status IN ('QUEUED', 'PROCESSING')
            ORDER BY created_at ASC
            LIMIT ?;
            """,
            (limit,)
        )
        rows = cursor.fetchall()

    pending = []
    for row in rows:
        pending.append({
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "doc_paths": json.loads(row["doc_paths_json"]) if row["doc_paths_json"] else {},
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_message": row["error_message"]
        })
    return pending