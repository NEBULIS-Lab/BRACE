import dataclasses
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict

EVENT_SCHEMA_VERSION = 1
EPISODE_SCHEMA_VERSION = 1
# Human-readable schema freeze tag for integration (stored in run.json only).
SCHEMA_TAG = "2026-01-26"


def _infer_event_type(event: Dict[str, Any]) -> str:
    if event.get("event_type") in ("replan", "phase"):
        return str(event["event_type"])
    if "phase" in event and "t" not in event and "step" not in event and "replan_cycle" not in event:
        return "phase"
    return "replan"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _safe_run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None


def collect_env_metadata() -> Dict[str, Any]:
    return {
        "time_utc": _utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
            "EGL_VISIBLE_DEVICES": os.environ.get("EGL_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "git": {
            "commit": _safe_run(["git", "rev-parse", "HEAD"]),
            "status": _safe_run(["git", "status", "--porcelain"]),
        },
    }


@dataclass
class RunContext:
    run_dir: str
    run_id: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, runs_root: str, run_name: str, config: Dict[str, Any]) -> "RunContext":
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}__{run_name}"
        run_dir = os.path.join(runs_root, run_id)
        os.makedirs(run_dir, exist_ok=False)
        metadata = collect_env_metadata()
        ctx = cls(run_dir=run_dir, run_id=run_id, config=config, metadata=metadata)
        ctx.write_run_json()
        return ctx

    def write_run_json(self) -> None:
        path = os.path.join(self.run_dir, "run.json")
        payload = {
            "run_id": self.run_id,
            "schema": {
                "events_jsonl": EVENT_SCHEMA_VERSION,
                "episode_metrics_jsonl": EPISODE_SCHEMA_VERSION,
            },
            "schema_tag": SCHEMA_TAG,
            "config": self.config,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event = dict(event)
        event.setdefault("schema_version", EVENT_SCHEMA_VERSION)
        event.setdefault("run_id", self.run_id)
        event.setdefault("domain", self.config.get("domain"))
        event.setdefault("task", self.config.get("task"))
        event.setdefault("event_type", _infer_event_type(event))
        event_type = str(event.get("event_type"))

        # Unify commonly-used aliases (keep originals too).
        if "t" not in event and "step" in event:
            event["t"] = event.get("step")
        if "clarification_budget_turns" not in event and "clarification_turns" in event:
            event["clarification_budget_turns"] = event.get("clarification_turns")

        # Phase accounting (optional; used by VLM/VLA tracks and overhead breakdown).
        if event_type == "phase":
            event.setdefault("phase", None)
            event.setdefault("phase_idx", None)
            event.setdefault("phase_parent", None)
            event.setdefault("vlm_model", None)
            event.setdefault("vlm_tokens_in", None)
            event.setdefault("vlm_tokens_out", None)
            event.setdefault("lat_vlm_ms", None)
            # Optional demo artifact metadata (videos/screenshots as experiment artifacts).
            event.setdefault("path", None)
            event.setdefault("fps", None)
            event.setdefault("frame_wh", None)
            event.setdefault("probe", None)

        # Budget defaults (may be overridden by per-event values).
        event.setdefault("slo_ms", self.config.get("slo_ms"))
        event.setdefault("token_budget", self.config.get("token_budget"))
        event.setdefault(
            "clarification_budget_turns",
            self.config.get("clarification_budget_turns", self.config.get("clarification_turns")),
        )

        # Feature toggles: keep as null if unknown.
        for k in ("brace_enabled", "pruning_enabled", "rag_enabled", "summary_compress_enabled"):
            event.setdefault(k, None)

        # BRACE mode (for mode-aware aggregation).
        event.setdefault("mode", None)

        # Standard latency keys (ms).
        for k in ("lat_total_ms", "lat_prune_ms", "lat_retrieval_ms", "lat_prefill_ms", "lat_decode_ms"):
            event.setdefault(k, None)

        # Standard token keys (counts).
        for k in (
            "tokens_in",
            "tokens_after_prune",
            "tokens_task",
            "tokens_state",
            "tokens_safety",
            "tokens_coord",
            "tokens_history",
        ):
            event.setdefault(k, None)

        # Tail/SLO accounting.
        event.setdefault("slo_violation", None)
        event.setdefault("slo_over_ms", None)
        if (
            event.get("slo_violation") is None
            and event.get("slo_over_ms") is None
            and event.get("lat_total_ms") is not None
            and event.get("slo_ms") is not None
        ):
            try:
                lat_ms = float(event["lat_total_ms"])
                slo_ms = float(event["slo_ms"])
                event["slo_over_ms"] = max(0.0, lat_ms - slo_ms)
                event["slo_violation"] = bool(lat_ms > slo_ms)
            except Exception:
                # Leave as null if types are not numeric.
                event["slo_over_ms"] = None
                event["slo_violation"] = None

        # Stability proxies (may be filled by controller/planner integrations).
        event.setdefault("plan_hash", None)
        event.setdefault("plan_churn_score", None)

        # Multi-agent coordination (optional; keep null when not applicable).
        event.setdefault("deadlock_flag", None)
        event.setdefault("wait_time_ms", None)
        event.setdefault("locks_held", None)
        event.setdefault("locks_waiting", None)

        return event

    def _normalize_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        episode = dict(episode)
        episode.setdefault("schema_version", EPISODE_SCHEMA_VERSION)
        episode.setdefault("run_id", self.run_id)
        episode.setdefault("domain", self.config.get("domain"))
        episode.setdefault("task", self.config.get("task"))
        if "replan_cycles" not in episode and "replans" in episode:
            episode["replan_cycles"] = episode.get("replans")

        # Standard derived metrics (may be computed by the runner or by analysis).
        for k in (
            "lat_p50_ms",
            "lat_p95_ms",
            "lat_p99_ms",
            "slo_violation_rate",
            "tokens_in_mean",
            "tokens_in_p95",
            "tokens_in_p99",
            "tokens_after_prune_mean",
            "tokens_after_prune_p95",
            "tokens_after_prune_p99",
            "deadlock_flag",
            "wait_time_ms",
        ):
            episode.setdefault(k, None)
        return episode

    def append_event(self, event: Dict[str, Any]) -> None:
        event = self._normalize_event(event)
        event.setdefault("time_utc", _utc_now_iso())
        path = os.path.join(self.run_dir, "events.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def log_phase(self, *, phase: str, **fields: Any) -> None:
        """Append a lightweight phase event into the same `events.jsonl`.

        Intended usage: VLM/VLA tracks and overhead breakdown where one replanning call is split into
        phases like `vlm_summarize/context_compress/planner_call`.
        """

        event: Dict[str, Any] = {"event_type": "phase", "phase": str(phase)}
        event.update(fields)
        self.append_event(event)

    def append_episode(self, episode: Dict[str, Any]) -> None:
        episode = self._normalize_episode(episode)
        episode.setdefault("time_utc", _utc_now_iso())
        path = os.path.join(self.run_dir, "episode_metrics.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, sort_keys=True) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary = dict(summary)
        summary.setdefault("time_utc", _utc_now_iso())
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
