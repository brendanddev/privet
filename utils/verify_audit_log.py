
"""
utils/verify_audit_log.py

Standalone CLI tool for verifying and summarising a privacy audit log
produced by PrivacyAuditLog.

Usage:
    python3 -m utils.verify_audit_log
    python3 -m utils.verify_audit_log path/to/privacy_audit.jsonl

Exits with code 0 if the chain is valid, 1 if any integrity errors are found.
The report is printed to stdout and written to logs/privacy_report_<date>.txt.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from utils.privacy_audit_log import PrivacyAuditLog

_DEFAULT_LOG_PATH = "./logs/privacy_audit.jsonl"
_REPORT_DIR = "./logs"

_BORDER = "╔══════════════════════════════════════╗"
_TITLE1 = "║   LOCAL RAG ASSISTANT — PRIVACY      ║"
_TITLE2 = "║   AUDIT VERIFICATION REPORT          ║"
_BORDER_CLOSE = "╚══════════════════════════════════════╝"


def _read_entries(log_path: str) -> list[dict]:
    """
    Read and parse all non-empty JSONL lines from the audit log.

    Silently skips lines that are not valid JSON, integrity errors
    are reported separately by verify_chain_integrity().

    Args:
        log_path (str): Path to the JSONL audit log file

    Returns:
        list[dict]: Parsed entry dicts in file order
    """
    entries = []
    path = Path(log_path)
    if not path.exists():
        return entries
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return entries


def _fmt_ts(iso: str) -> str:
    """
    Truncate an ISO-8601 timestamp to minute precision for display.

    Args:
        iso (str): Full ISO timestamp string

    Returns:
        str: e.g. "2026-03-23T16:42Z"
    """
    # Keep up to the minute, replace +00:00 / seconds with Z
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%dT%H:%MZ")
    except ValueError:
        return iso[:16] + "Z"


def _build_report(log_path: str) -> tuple[str, bool]:
    """
    Generate the full verification report as a string.

    Reads the log, runs chain verification, then aggregates per-event-type
    statistics. Returns the rendered report and a validity flag.

    Args:
        log_path (str): Path to the JSONL audit log file

    Returns:
        tuple[str, bool]: (report_text, is_valid)
    """
    audit = PrivacyAuditLog(log_path)
    is_valid, errors = audit.verify_chain_integrity()
    entries = _read_entries(log_path)

    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

    # Header
    out(_BORDER)
    out(_TITLE1)
    out(_TITLE2)
    out(_BORDER_CLOSE)
    out()

    # Chain integrity summary
    if is_valid:
        out(f"  Chain integrity:  VALID")
    else:
        out(f"  Chain integrity:  INVALID — {len(errors)} error(s) found")
    out()

    # Period and total count
    if entries:
        first_ts = _fmt_ts(entries[0].get("timestamp", ""))
        last_ts = _fmt_ts(entries[-1].get("timestamp", ""))
        out(f"  Period:           {first_ts}  to  {last_ts}")
    else:
        out(f"  Period:           (no entries)")
    out(f"  Total entries:    {len(entries)}")
    out()

    # Sessions
    sessions = [e for e in entries if e.get("event_type") == "SESSION_START"]
    providers: Counter = Counter(e.get("provider", "unknown") for e in sessions)
    provider_str = ", ".join(f"{p} ({c})" for p, c in providers.most_common())

    out("  SESSIONS")
    out("  --------")
    out(f"  Sessions logged:        {len(sessions)}")
    out(f"  Providers used:         {provider_str if provider_str else 'N/A'}")
    out()

    # Queries
    queries = [e for e in entries if e.get("event_type") == "QUERY_PROCESSED"]
    total_ext_bytes = sum(e.get("network_bytes_external", 0) for e in queries)
    queries_with_traffic = sum(
        1 for e in queries if e.get("network_bytes_external", 0) > 0
    )
    def _query_is_private(e: dict) -> bool:
        if "verified_private" in e:
            return e["verified_private"]
        return not e.get("data_left_device", False)

    any_data_left = any(not _query_is_private(e) for e in queries)

    out("  QUERIES")
    out("  -------")
    out(f"  Total queries:          {len(queries)}")
    out(f"  External bytes total:   {total_ext_bytes}")
    out(f"  Queries with any external traffic:  {queries_with_traffic}")
    out(f"  Data left device:       {'YES ⚠' if any_data_left else 'NO'}")
    out()

    # Documents
    doc_events = [e for e in entries if e.get("event_type") == "DOCUMENT_ACCESS"]
    unique_files = {e.get("filename", "") for e in doc_events}
    action_counts: Counter = Counter(e.get("action", "UNKNOWN") for e in doc_events)
    action_str = ", ".join(f"{a} {c}" for a, c in sorted(action_counts.items()))

    out("  DOCUMENTS")
    out("  ---------")
    out(f"  Unique files accessed:  {len(unique_files)}")
    out(f"  Index operations:       {action_str if action_str else 'none'}")
    out()

    # Integrity
    last_hash = entries[-1].get("own_hash", "")[:16] if entries else "N/A"

    out("  INTEGRITY")
    out("  ---------")
    out(f"  Last entry hash:        {last_hash}...")
    if is_valid:
        out(f"  Verification result:    ALL ENTRIES VALID")
    else:
        out(f"  Verification result:    INVALID — see errors below")
        out()
        for err in errors:
            out(f"    ✗ {err}")
    out()

    return "\n".join(lines), is_valid


def _write_report(report: str) -> str:
    """
    Write the report string to a dated file in the logs directory.

    Args:
        report (str): The full report text

    Returns:
        str: Path to the written report file
    """
    os.makedirs(_REPORT_DIR, exist_ok=True)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(_REPORT_DIR, f"privacy_report_{date_str}.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    return report_path


def main(log_path: str) -> int:
    """
    Run the verification report and return an exit code.

    Prints the report to stdout, writes it to a dated file in logs/,
    and returns 0 (valid) or 1 (invalid) for use in shell scripts.

    Args:
        log_path (str): Path to the JSONL audit log file

    Returns:
        int: 0 if chain is valid, 1 if any integrity errors were found
    """
    report, is_valid = _build_report(log_path)
    print(report)

    report_path = _write_report(report)
    print(f"  Report saved to: {report_path}")

    return 0 if is_valid else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_LOG_PATH
    sys.exit(main(path))

