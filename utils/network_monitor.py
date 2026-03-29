"""
utils/network_monitor.py

Monitors external network I/O during RAG queries to verify that no data
is transmitted to external servers. Uses psutil.net_io_counters(pernic=True)
to get per-interface stats, excludes loopback interfaces ('lo', 'lo0'), and
applies a noise threshold of 8KB per interface to filter out background OS
broadcasts (mDNS, Bonjour, network discovery) that fire independently of
any query. A query is flagged as outbound only if a single interface sent
more than 8KB — a real API call would be at minimum 10-50KB.
"""

import time
from dataclasses import dataclass
from typing import Optional
import psutil
from utils.logger import setup_logger

logger = setup_logger()


@dataclass
class QueryNetworkResult:
    """Network I/O recorded during a single query window."""
    bytes_sent: int
    bytes_recv: int
    duration: float
    verified_private: bool  # True if no single interface exceeded NOISE_THRESHOLD


class NetworkMonitor:
    """
    Monitors external network I/O during RAG queries.

    Snapshots per-interface bytes_sent at query start and computes per-interface
    deltas at query end. Loopback interfaces ('lo', 'lo0') are excluded. A query
    is only flagged as outbound if any single interface sent more than NOISE_THRESHOLD
    bytes — small packets under that limit are background OS broadcasts (mDNS,
    Bonjour, network discovery) and are not counted as query data.

    Maintains a session_log of QueryNetworkResult entries for the duration
    of the Streamlit session.
    """

    _LOOPBACK_INTERFACES = frozenset(('lo', 'lo0'))
    NOISE_THRESHOLD = 8192  # 8KB — real API calls are minimum 10-50KB

    def __init__(self):
        self.session_log: list[QueryNetworkResult] = []
        self._snapshot_iface_sent: Optional[dict[str, int]] = None
        self._snapshot_recv: Optional[int] = None
        self._start_time: Optional[float] = None
        logger.info("NetworkMonitor initialized")

    def _external_stats(self) -> dict:
        """
        Return per-interface psutil stats for non-loopback interfaces only.

        Returns:
            dict: Interface name -> snetio named tuple (bytes_sent, bytes_recv, ...)
        """
        return {
            k: v for k, v in psutil.net_io_counters(pernic=True).items()
            if k not in self._LOOPBACK_INTERFACES
        }

    def start_query(self) -> None:
        """
        Snapshot per-interface bytes_sent immediately before a query begins.

        Stores a baseline per interface so end_query() can compute per-interface
        deltas and apply the noise threshold correctly.

        Must be called before stream_query(). Pair with end_query().
        """
        stats = self._external_stats()
        self._snapshot_iface_sent = {k: v.bytes_sent for k, v in stats.items()}
        self._snapshot_recv = sum(v.bytes_recv for v in stats.values())
        self._start_time = time.monotonic()
        logger.info("NetworkMonitor: query window started")

    def end_query(self) -> QueryNetworkResult:
        """
        Compute per-interface deltas after the query completes.

        A query is marked verified_private if no single external interface sent
        more than NOISE_THRESHOLD bytes. Packets below that limit are treated as
        background OS traffic (mDNS, Bonjour) rather than query data.

        Appends the result to session_log. Resets internal state so
        start_query() can be called again for the next query.

        Returns:
            QueryNetworkResult: External bytes sent/received during the query
                window, duration in seconds, and whether the query was private.

        Raises:
            RuntimeError: If called without a preceding start_query().
        """
        if self._snapshot_iface_sent is None or self._start_time is None:
            raise RuntimeError("end_query() called without a preceding start_query()")

        stats = self._external_stats()
        duration = round(time.monotonic() - self._start_time, 3)

        # Per-interface delta — new interfaces that appeared mid-query count fully
        total_sent = 0
        max_iface_sent = 0
        for iface, counters in stats.items():
            delta = counters.bytes_sent - self._snapshot_iface_sent.get(iface, 0)
            total_sent += delta
            max_iface_sent = max(max_iface_sent, delta)

        recv = sum(v.bytes_recv for v in stats.values()) - self._snapshot_recv
        verified = max_iface_sent <= self.NOISE_THRESHOLD

        result = QueryNetworkResult(
            bytes_sent=total_sent,
            bytes_recv=recv,
            duration=duration,
            verified_private=verified,
        )

        self.session_log.append(result)

        self._snapshot_iface_sent = None
        self._snapshot_recv = None
        self._start_time = None

        logger.info(
            f"NetworkMonitor: query window ended | bytes_sent={total_sent} | "
            f"max_iface_sent={max_iface_sent} | bytes_recv={recv} | "
            f"duration={duration}s | private={verified}"
        )
        return result

    def get_session_summary(self) -> dict:
        """
        Return totals across all queries this session.

        all_private is True only if every query in the session was verified private
        (i.e. no single interface exceeded NOISE_THRESHOLD in any query).

        Returns:
            dict: total_bytes_sent, total_bytes_recv, query_count,
                  verified_private_count, all_private (bool)
        """
        total_sent = sum(r.bytes_sent for r in self.session_log)
        total_recv = sum(r.bytes_recv for r in self.session_log)
        count = len(self.session_log)
        private_count = sum(1 for r in self.session_log if r.verified_private)

        return {
            "total_bytes_sent": total_sent,
            "total_bytes_recv": total_recv,
            "query_count": count,
            "verified_private_count": private_count,
            "all_private": private_count == count,
        }
