"""
SQLite persistence repository for asynchronous Form-C processing tasks.
Provides atomic job creation, status updates, crash recovery reset queries,
and serialized JSON storage for nested FieldResult extraction structures.
"""

from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
from typing import Dict, Any, List, Optional

# Set up module-level logger
logger = logging.getLogger("storage.db")
logger.setLevel(logging.INFO)

# Define filesystem path for SQLite database file
STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE_PATH = os.path.join(STORAGE_DIR, "jobs.db")


# ==============================================================================
# Database Connection Helper
# ==============================================================================

def get_db_connection() -> sqlite3.Connection:
    """
    Creates and returns an active SQLite database connection configured with
    a row factory for column-name dictionary mapping.

    :return: sqlite3.Connection instance.
    """
    # Establish connection with timeout for multi-thread safety
    conn = sqlite3.connect(DB_FILE_PATH, timeout=15.0)

    # Enable column access by dictionary key
    conn.row_factory = sqlite3.Row

    # Enforce Write-Ahead Logging (WAL) mode for concurrent read-write performance
    conn.execute("PRAGMA journal_mode=WAL;")

    return conn


# ==============================================================================
# Schema Initialization
# ==============================================================================

def initialize_database() -> None:
    """
    Creates the 'jobs' table and indexes if they do not already exist.
    Stores metadata, filesystem paths, lifecycle status, and the JSON extraction result.
    """
    logger.info(f"[*] Initializing SQLite database schema at: {DB_FILE_PATH}")

    create_table_query = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        doc_paths_json TEXT NOT NULL,
        result_json TEXT,
        error_message TEXT
    );
    """

    create_index_query = """
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        cursor.execute(create_index_query)
        conn.commit()

    logger.info("[*] SQLite database schema initialized successfully.")


# ==============================================================================
# Job Management Operations
# ==============================================================================

def create_job(job_id: str, doc_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Inserts a newly ingested document bundle record into the database with initial state 'QUEUED'.

    :param job_id: Unique UUID string identifier.
    :param doc_paths: Dictionary mapping document types ('passport_bio', etc.) to disk paths.
    :return: Dictionary containing initialized job metadata.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    paths_serialized = json.dumps(doc_paths)

    insert_query = """
    INSERT INTO jobs (job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message)
    VALUES (?, 'QUEUED', ?, ?, ?, NULL, NULL);
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(insert_query, (job_id, now_iso, now_iso, paths_serialized))
        conn.commit()

    logger.info(f"[*] Created QUEUED database record for job: {job_id}")

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": now_iso,
        "updated_at": now_iso,
        "doc_paths": doc_paths
    }


def update_job_status(job_id: str, new_status: str, error_message: Optional[str] = None) -> None:
    """
    Updates the lifecycle status of a specific job (e.g., 'PROCESSING', 'FAILED').

    :param job_id: Target job UUID.
    :param new_status: New status string matching JobStatus enum.
    :param error_message: Optional error details if status transition is FAILED.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    update_query = """
    UPDATE jobs
    SET status = ?, updated_at = ?, error_message = ?
    WHERE job_id = ?;
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(update_query, (new_status, now_iso, error_message, job_id))
        conn.commit()

    logger.info(f"[*] Job {job_id} transitioned status -> {new_status}")


def save_job_result(job_id: str, result_dict: Dict[str, Any]) -> None:
    """
    Persists the final completed extraction dictionary into 'result_json' and
    marks the job status as 'COMPLETED'.

    :param job_id: Target job UUID.
    :param result_dict: Dictionary representation of the fused GuestFormCRecord.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    serialized_result = json.dumps(result_dict)

    complete_query = """
    UPDATE jobs
    SET status = 'COMPLETED', updated_at = ?, result_json = ?, error_message = NULL
    WHERE job_id = ?;
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(complete_query, (now_iso, serialized_result, job_id))
        conn.commit()

    logger.info(f"[+] Persisted final extraction result for job: {job_id}")


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves full record for a specific job, deserializing JSON fields into Python dictionaries.

    :param job_id: Unique UUID string identifier.
    :return: Job record dictionary or None if not found.
    """
    select_query = """
    SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
    FROM jobs
    WHERE job_id = ?;
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(select_query, (job_id,))
        row = cursor.fetchone()

    if not row:
        return None

    # Safely deserialize stored doc paths
    try:
        doc_paths = json.loads(row["doc_paths_json"]) if row["doc_paths_json"] else {}
    except (ValueError, TypeError):
        doc_paths = {}

    # Safely deserialize fused result JSON
    try:
        result = json.loads(row["result_json"]) if row["result_json"] else None
    except (ValueError, TypeError):
        result = None

    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "doc_paths": doc_paths,
        "result": result,
        "error_message": row["error_message"]
    }


def list_recent_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves recent jobs ordered by creation timestamp descending for Android Room sync.

    :param limit: Maximum number of rows to retrieve.
    :return: List of job record dictionaries.
    """
    select_recent_query = """
    SELECT job_id, status, created_at, updated_at, doc_paths_json, result_json, error_message
    FROM jobs
    ORDER BY created_at DESC
    LIMIT ?;
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(select_recent_query, (limit,))
        rows = cursor.fetchall()

    results = []
    for r in rows:
        try:
            paths = json.loads(r["doc_paths_json"]) if r["doc_paths_json"] else {}
        except Exception:
            paths = {}

        try:
            res_dict = json.loads(r["result_json"]) if r["result_json"] else None
        except Exception:
            res_dict = None

        results.append({
            "job_id": r["job_id"],
            "status": r["status"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "doc_paths": paths,
            "result": res_dict,
            "error_message": r["error_message"]
        })

    return results


def reset_interrupted_jobs() -> List[str]:
    """
    Sweeps the database at application startup to identify jobs left stranded in
    the 'PROCESSING' state due to an unexpected power outage or server reboot.
    Resets their state back to 'QUEUED' so they can be re-enqueued into the worker.

    :return: List of job IDs that were recovered and reset to 'QUEUED'.
    """
    select_query = "SELECT job_id FROM jobs WHERE status = 'PROCESSING';"
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(select_query)
        interrupted_rows = cursor.fetchall()
        recovered_ids = [r["job_id"] for r in interrupted_rows]

        if recovered_ids:
            reset_query = """
            UPDATE jobs
            SET status = 'QUEUED', updated_at = ?
            WHERE status = 'PROCESSING';
            """
            cursor.execute(reset_query, (now_iso,))
            conn.commit()

    return recovered_ids