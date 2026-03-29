"""
utils/privacy_audit_log.py

Append-only, hash-chained audit log that records every privacy-relevant
event in the RAG pipeline: session starts, queries processed, document
accesses, and network measurements.

Each entry is a JSON object on one line (JSONL). Entries are linked via
SHA-256 hash chaining — each entry stores the hash of the previous entry
so the log cannot be silently altered after the fact.

Integrity can be verified at any time with verify_chain_integrity().
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil

from utils.logger import setup_logger

logger = setup_logger()

# 8KB, same threshold as NetworkMonitor
_EXTERNAL_DATA_THRESHOLD = 8192

_LOOPBACK_INTERFACES = frozenset(("lo", "lo0"))


def _sha256(text: str) -> str:
    """
    Return the SHA-256 hex digest of a UTF-8 string.

    Args:
        text (str): Input string to hash

    Returns:
        str: 64-character lowercase hex digest
    """
    return hashlib.sha256(text.encode("utf-8")).digest().hex()


def _canonical(entry: dict) -> str:
    """
    Serialise a dict to a canonical JSON string with sorted keys.

    Sorted keys ensure the same dict always produces the same string
    regardless of insertion order, which is required for hash chaining.

    Args:
        entry (dict): The audit entry dict

    Returns:
        str: JSON string with sorted keys, no extra whitespace
    """
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def _external_interface_names() -> list[str]:
    """
    Return the names of all non loopback network interfaces that are currently up.

    Uses psutil.net_if_stats() to check which interfaces have is_up=True,
    then excludes loopback names. These are the interfaces that can carry
    traffic to the external network.

    Returns:
        list[str]: Sorted list of active external interface names
    """
    try:
        stats = psutil.net_if_stats()
        return sorted(
            name for name, stat in stats.items()
            if stat.isup and name not in _LOOPBACK_INTERFACES
        )
    except Exception:
        return []


class PrivacyAuditLog:
    """
    Append-only, hash-chained JSONL audit log for privacy verification.

    Every write appends a single JSON line to the log file. Each entry
    carries the SHA-256 hash of the previous entry (or "GENESIS" for the
    first), and its own hash computed from its full content. This forms a
    chain that makes silent post-hoc editing detectable.

    Entry types:
        SESSION_START        — recorded once per app startup
        QUERY_PROCESSED      — recorded after each query completes
        DOCUMENT_ACCESS      — recorded when a document is indexed or removed
        NETWORK_VERIFICATION — recorded with raw network monitor measurements

    The log path defaults to ./logs/privacy_audit.jsonl and can be
    overridden via the privacy_audit_path key in config.yaml.
    """

    def __init__(self, log_path: str = "./logs/privacy_audit.jsonl"):
        """
        Initialise the audit log.

        Creates the parent directory if it does not exist. Reads the last
        line of an existing file to seed the hash chain correctly.

        Args:
            log_path (str): Path to the JSONL audit file.
                            Default: ./logs/privacy_audit.jsonl
        """
        self._path = Path(log_path)
        os.makedirs(self._path.parent, exist_ok=True)
        self._last_hash = self._read_last_hash()
        logger.info(f"PrivacyAuditLog initialised | path={self._path}")

    def _read_last_hash(self) -> str:
        """
        Return the own_hash of the most recent entry in the log file.

        If the file does not exist or is empty, returns "GENESIS" so the
        first entry correctly records that it has no predecessor.

        Returns:
            str: SHA-256 hex digest of the last entry, or "GENESIS"
        """
        if not self._path.exists():
            return "GENESIS"

        last_line = ""
        try:
            with open(self._path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
        except OSError:
            return "GENESIS"

        if not last_line:
            return "GENESIS"

        try:
            entry = json.loads(last_line)
            return entry.get("own_hash", "GENESIS")
        except json.JSONDecodeError:
            return "GENESIS"

    def _write_entry(self, event_type: str, payload: dict) -> None:
        """
        Build, hash, and append a single audit entry to the log file.

        Constructs the full entry dict with metadata and hash fields,
        then appends it as one JSON line. Updates self._last_hash so the
        next entry can reference it.

        The own_hash is computed over the canonical JSON of the complete
        entry (including previous_hash) before writing.

        Args:
            event_type (str): One of SESSION_START, QUERY_PROCESSED,
                              DOCUMENT_ACCESS, NETWORK_VERIFICATION
            payload (dict): Event-specific fields to include in the entry
        """
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": event_type,
            "previous_hash": self._last_hash,
            **payload,
        }
        entry["own_hash"] = _sha256(_canonical(entry))

        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._last_hash = entry["own_hash"]
        logger.info(f"PrivacyAuditLog: wrote {event_type} | hash={self._last_hash[:12]}…")

    def log_session_start(
        self,
        provider: str,
        model_name: str,
        docs_count: int,
        ollama_host: Optional[str],
    ) -> None:
        """
        Record a SESSION_START entry at app startup.

        Captures the active provider, model, document count, Ollama host
        (if applicable), active external network interfaces, and an
        external_api_calls counter initialised to 0.

        Args:
            provider (str): Provider name from config, e.g. "ollama" or "llamacpp"
            model_name (str): LLM model name in use
            docs_count (int): Number of documents currently indexed
            ollama_host (str | None): Ollama host URL, or None for non-Ollama providers
        """
        self._write_entry("SESSION_START", {
            "provider": provider,
            "model_name": model_name,
            "docs_count": docs_count,
            "ollama_host_or_NA": ollama_host if ollama_host else "NA",
            "active_external_interfaces": _external_interface_names(),
            "external_api_calls": 0,
        })

    def log_query(
        self,
        query: str,
        sources: list[dict],
        response_length: int,
        query_time_ms: float,
        network_bytes: int,
        verified_private: bool = True,
    ) -> None:
        """
        Record a QUERY_PROCESSED entry after a query completes.

        The raw query string is never stored, only its SHA-256 hash.
        Source filenames are extracted without path components or content.

        verified_private mirrors NetworkMonitor.end_query() — it is True
        when no single external interface exceeded the 8KB noise threshold,
        which is the correct per-interface check. network_bytes_external is
        the raw total across all interfaces and is recorded for the audit
        record but is not used to derive data_left_device.

        data_left_device is the logical inverse of verified_private so both
        fields are present in the entry and mean the same thing.

        Args:
            query (str): The raw query string (hashed before storage)
            sources (list[dict]): Source dicts from engine.last_sources;
                                  each must have a "file" key
            response_length (int): Character count of the generated answer
            query_time_ms (float): Query duration in milliseconds
            network_bytes (int): Total external bytes sent across all interfaces
            verified_private (bool): From NetworkMonitor — True when no single
                                     interface exceeded the noise threshold
        """
        filenames = [
            os.path.basename(s.get("file", ""))
            for s in sources
            if s.get("file")
        ]
        self._write_entry("QUERY_PROCESSED", {
            "query_hash": _sha256(query),
            "source_filenames": filenames,
            "response_length_chars": response_length,
            "query_time_ms": query_time_ms,
            "network_bytes_external": network_bytes,
            "verified_private": verified_private,
            "data_left_device": not verified_private,
        })

    def log_document_access(
        self,
        filename: str,
        action: str,
        chunk_count: int,
    ) -> None:
        """
        Record a DOCUMENT_ACCESS entry when a document is indexed or removed.

        transmitted_externally is always False, documents never leave the
        machine. It is recorded explicitly so the claim is auditable.

        Args:
            filename (str): Document filename (basename only, no path)
            action (str): "INDEXED" when a document is added,
                          "REMOVED" when a document is deleted from the index
            chunk_count (int): Number of chunks written to or removed from ChromaDB
        """
        self._write_entry("DOCUMENT_ACCESS", {
            "filename": os.path.basename(filename),
            "action": action,
            "chunk_count": chunk_count,
            "transmitted_externally": False,
        })

    def log_network_verification(self, measurement: dict) -> None:
        """
        Record a NETWORK_VERIFICATION entry with a raw network monitor measurement.

        Stores the measurement dict as-is so the audit log contains the
        same data that the privacy panel displays to the user.

        Args:
            measurement (dict): Raw measurement dict from NetworkMonitor,
                                e.g. the result of get_session_summary()
        """
        self._write_entry("NETWORK_VERIFICATION", {
            "measurement": measurement,
        })

    def verify_chain_integrity(self) -> tuple[bool, list[str]]:
        """
        Read the log file and verify the hash chain from start to finish.

        For each entry, recomputes the expected own_hash over the entry's
        canonical JSON (excluding the own_hash field itself, since it is
        added after). Also checks that each entry's previous_hash matches
        the own_hash of the preceding entry.

        An empty or missing file is considered valid (no entries to check).

        Returns:
            tuple[bool, list[str]]: (is_valid, error_messages)
                is_valid is True only when every entry passes both checks.
                error_messages is an empty list when is_valid is True.
        """
        if not self._path.exists():
            return True, []

        errors: list[str] = []
        prev_hash = "GENESIS"
        line_number = 0

        try:
            with open(self._path, "r") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    line_number += 1

                    try:
                        entry = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        errors.append(f"Line {line_number}: invalid JSON — {exc}")
                        continue

                    stored_own_hash = entry.get("own_hash", "")
                    stored_prev_hash = entry.get("previous_hash", "")

                    # Check backward link
                    if stored_prev_hash != prev_hash:
                        errors.append(
                            f"Line {line_number}: previous_hash mismatch — "
                            f"expected {prev_hash[:12]}… got {stored_prev_hash[:12]}…"
                        )

                    # Recompute own_hash over the entry without the own_hash field
                    entry_without_own = {k: v for k, v in entry.items() if k != "own_hash"}
                    expected_own_hash = _sha256(_canonical(entry_without_own))

                    if stored_own_hash != expected_own_hash:
                        errors.append(
                            f"Line {line_number}: own_hash mismatch — "
                            f"entry may have been altered"
                        )

                    prev_hash = stored_own_hash

        except OSError as exc:
            errors.append(f"Could not read audit log: {exc}")

        is_valid = len(errors) == 0
        logger.info(
            f"PrivacyAuditLog: chain verification complete | "
            f"lines={line_number} | valid={is_valid} | errors={len(errors)}"
        )
        return is_valid, errors