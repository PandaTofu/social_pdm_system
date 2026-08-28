"""Generate replayable, correlated social-media backend telemetry.

The generated label is calculated from a future incident schedule, never copied
from same-row metrics.  This prevents target leakage in the offline experiment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import numpy as np


@dataclass(frozen=True)
class Settings:
    seed: int
    start_time: datetime
    logical_servers: int
    days: int
    interval: int
    data_centers: int
    schema_v2_start_day: int
    drift_start_day: int
    horizon: int
    quality_error_rate: float
    late_event_rate: float
    duplicate_rate: float
    quality_defect_rates: dict[str, float]
    burst_windows: list[dict[str, Any]]
    scenario: str
    precursor_window: int
    incident_windows: list[dict[str, int]]
    concept_drift_strength: float
    drift_transition_days: int


def settings_from(path: Path) -> Settings:
    # JSON is a YAML subset. Keeping this configuration JSON-formatted YAML
    # lets the lightweight generator run without a parser dependency.
    raw = json.loads(path.read_text(encoding="utf-8"))
    legacy_quality_rate = float(raw["quality_error_rate"])
    explicit_quality_rates = raw.get("quality_defect_rates")
    quality_defect_rates = (
        {str(name): float(rate) for name, rate in explicit_quality_rates.items()}
        if explicit_quality_rates is not None
        else {
            "range": legacy_quality_rate / 2,
            "completeness": legacy_quality_rate / 2,
            "temporal": float(raw["late_event_rate"]),
        }
    )
    settings = Settings(
        seed=int(raw["seed"]), start_time=datetime.fromisoformat(raw["start_time"].replace("Z", "+00:00")),
        logical_servers=int(raw["logical_servers"]), days=int(raw["days"]),
        interval=int(raw["sample_interval_minutes"]), data_centers=int(raw["data_centers"]),
        schema_v2_start_day=int(raw["schema_v2_start_day"]), drift_start_day=int(raw["drift_start_day"]),
        horizon=int(raw["failure_horizon_minutes"]), quality_error_rate=legacy_quality_rate,
        late_event_rate=float(raw["late_event_rate"]), duplicate_rate=float(raw["duplicate_rate"]),
        quality_defect_rates=quality_defect_rates,
        burst_windows=list(raw.get("burst_windows", [])),
        scenario=str(raw.get("scenario", "covariate_shift_v1")),
        precursor_window=int(raw.get("precursor_window_minutes", 90)),
        incident_windows=list(raw.get("incident_windows", [])),
        concept_drift_strength=float(raw.get("concept_drift_strength", 1.0)),
        drift_transition_days=int(raw.get("drift_transition_days", 0)),
    )
    if settings.scenario not in {
        "covariate_shift_v1", "concept_drift_v2", "concept_drift_30d_v3", "concept_drift_30d_v4"
    }:
        raise ValueError(f"Unsupported scenario: {settings.scenario}")
    if settings.precursor_window <= 0 or settings.horizon <= 0:
        raise ValueError("Failure horizon and precursor window must be positive")
    if settings.interval <= 0 or settings.days <= 0 or settings.logical_servers <= 0:
        raise ValueError("Sampling interval, days and logical server count must be positive")
    if not 1 <= settings.schema_v2_start_day <= settings.days:
        raise ValueError("schema_v2_start_day must fall inside the experiment")
    if not 1 <= settings.drift_start_day <= settings.days:
        raise ValueError("drift_start_day must fall inside the experiment")
    if settings.drift_transition_days < 0:
        raise ValueError("drift_transition_days cannot be negative")
    unknown_categories = set(settings.quality_defect_rates) - {
        "schema", "range", "temporal", "completeness", "cross_field"
    }
    if unknown_categories:
        raise ValueError(f"Unsupported quality categories: {sorted(unknown_categories)}")
    if any(rate < 0 for rate in settings.quality_defect_rates.values()):
        raise ValueError("Quality defect rates cannot be negative")
    if sum(settings.quality_defect_rates.values()) >= 1:
        raise ValueError("Combined quality defect rate must be below one")
    return settings


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def burst_multiplier(cfg: Settings, minute: int) -> float:
    for window in cfg.burst_windows:
        start = (int(window["start_day"]) - 1) * 1440 + int(window["start_hour"]) * 60
        if start <= minute < start + int(window["duration_minutes"]):
            return float(window["multiplier"])
    return 1.0


def future_failure(incidents: dict[int, list[int]], server: int, minute: int, horizon: int) -> int:
    return int(any(minute < incident <= minute + horizon for incident in incidents[server]))


def incident_schedule(cfg: Settings, rng: np.random.Generator, total_minutes: int) -> dict[int, list[int]]:
    """Build the historical v1 schedule or a phase-stratified v2 schedule."""
    if cfg.scenario == "covariate_shift_v1":
        return {server: sorted(rng.choice(
            np.arange(180, total_minutes - 60), size=max(1, cfg.days // 4), replace=False
        ).tolist()) for server in range(cfg.logical_servers)}
    if not cfg.incident_windows:
        raise ValueError("concept_drift_v2 requires incident_windows")
    incidents: dict[int, list[int]] = {}
    for server in range(cfg.logical_servers):
        scheduled: list[int] = []
        for window in cfg.incident_windows:
            start = (int(window["start_day"]) - 1) * 1440 + cfg.precursor_window + 1
            end = int(window["end_day"]) * 1440 - 60
            if start >= end or end > total_minutes:
                raise ValueError(f"Invalid incident window: {window}")
            scheduled.append(int(rng.integers(start, end)))
        incidents[server] = sorted(scheduled)
    return incidents


def minutes_to_next_incident(incidents: list[int], minute: int, window: int) -> int | None:
    candidates = [incident - minute for incident in incidents if 0 < incident - minute <= window]
    return min(candidates) if candidates else None


def precursor_severity(minutes_remaining: int | None, window: int) -> float:
    """Return a gradual 0..1 degradation signal inside the label horizon."""
    if minutes_remaining is None:
        return 0.0
    return float(np.clip(1.0 - minutes_remaining / window, 0.0, 1.0))


def drift_progress(cfg: Settings, minute: int) -> float:
    """Return a reproducible 0..1 transition from the legacy to new failure regime."""
    drift_start = (cfg.drift_start_day - 1) * 1440
    if minute < drift_start:
        return 0.0
    if cfg.drift_transition_days == 0:
        return 1.0
    transition_minutes = cfg.drift_transition_days * 1440
    return float(np.clip((minute - drift_start) / transition_minutes, 0.0, 1.0))


def deterministic_event_id(cfg: Settings, minute: int, server: int) -> str:
    """Return a stable event identifier for a logical observation.

    The feedback/calibration split hashes ``event_id``.  Random UUID4 values
    therefore made nominally seeded scenario regenerations produce slightly
    different model inputs.  UUID5 keeps IDs unique while making the complete
    experiment reproducible from its scenario configuration.
    """
    key = f"social-pdm:{cfg.scenario}:{cfg.seed}:{minute}:{server}"
    return str(uuid5(NAMESPACE_URL, key))


def choose_quality_defect(cfg: Settings, rng: np.random.Generator) -> str | None:
    """Choose at most one labelled defect category for a base observation."""
    draw = float(rng.random())
    cumulative = 0.0
    for category in ("schema", "range", "temporal", "completeness", "cross_field"):
        cumulative += cfg.quality_defect_rates.get(category, 0.0)
        if draw < cumulative:
            return category
    return None


def task_profile(
    rng: np.random.Generator,
    failure_warning: int,
    traffic_campaign: bool,
) -> tuple[str, str, int, str, float]:
    """Generate an independently labelled task class for the E2 router."""
    if failure_warning:
        return "incident_response", "high", 250, "stream", 2.5
    draw = float(rng.random())
    incident_cutoff = 0.35 if traffic_campaign else 0.12
    if draw < incident_cutoff:
        return "incident_enrichment", "high", 500, "stream", 1.8
    if draw < 0.70:
        return "health_monitor", "medium", 1000, "stream", 1.0
    if draw < 0.92:
        return "capacity_analytics", "low", 300_000, "batch", 1.4
    return "model_feedback", "low", 900_000, "batch", 2.0


def records(cfg: Settings) -> Iterable[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    total_minutes = cfg.days * 1440
    # Per-server baselines make entities non-identical.  A shared data-centre
    # shock creates realistic correlation among co-located servers.
    server_cpu = rng.uniform(35, 58, cfg.logical_servers)
    server_mem = rng.uniform(40, 66, cfg.logical_servers)
    incidents = incident_schedule(cfg, rng, total_minutes)
    for minute in range(0, total_minutes, cfg.interval):
        ts = cfg.start_time + timedelta(minutes=minute)
        day = minute // 1440 + 1
        hour = (minute % 1440) / 60
        daily = 0.65 + 0.45 * max(0.0, np.sin((hour - 8) * np.pi / 12))
        burst = burst_multiplier(cfg, minute)
        dc_shock = rng.normal(0, 2.5, cfg.data_centers)
        for server in range(cfg.logical_servers):
            dc = server % cfg.data_centers
            lead = minutes_to_next_incident(incidents[server], minute, cfg.precursor_window)
            near_incident = lead is not None
            severity = precursor_severity(lead, cfg.precursor_window)
            drift_fraction = drift_progress(cfg, minute)
            drift = drift_fraction > 0
            request_rate = max(1.0, rng.negative_binomial(24, 24 / (24 + 160 * daily * burst)))
            cpu = np.clip(server_cpu[server] + 0.075 * request_rate + dc_shock[dc] + rng.normal(0, 5), 0, 100)
            memory = np.clip(server_mem[server] + 0.025 * request_rate + rng.normal(0, 4), 0, 100)
            queue = max(0.0, rng.gamma(2.0, 4.0) + max(0, request_rate - 180) * .2)
            disk_incident_shift = 0.0
            if near_incident and cfg.scenario == "covariate_shift_v1":
                cpu = min(100, cpu + rng.uniform(15, 30)); queue += rng.uniform(45, 100)
                disk_incident_shift = 12.0
            elif near_incident:
                # Gradually transfer failure evidence from resource saturation
                # to application latency/errors, creating a controlled P(y|X)
                # change without a discontinuity at the drift boundary.
                strength = cfg.concept_drift_strength
                cpu_shift = 28 * (1 - drift_fraction) + 4 * drift_fraction
                queue_shift = 80 * (1 - drift_fraction) + 12 * drift_fraction
                disk_shift = 15 * (1 - drift_fraction) + 3 * drift_fraction
                cpu = min(100, cpu + cpu_shift * severity * strength)
                queue += queue_shift * severity * strength
                disk_incident_shift = disk_shift * severity * strength
            response_p50 = float(rng.lognormal(4.15, .32) * (1 + queue / 120))
            response_p95 = max(response_p50, response_p50 * float(rng.lognormal(.7, .22)))
            # Deployment changes the relationship between app metrics and risk.
            if cfg.scenario == "covariate_shift_v1":
                error_alpha = 1.2 + (3 if near_incident else 0)
                error_rate = float(np.clip(rng.beta(error_alpha, 145), 0, 1))
                if drift:
                    response_p95 *= 1.17; error_rate = min(1.0, error_rate * 1.7)
            else:
                post_signal = ((0.5 * (1 - drift_fraction) + 16 * drift_fraction)
                               * severity * cfg.concept_drift_strength)
                error_rate = float(np.clip(rng.beta(1.2 + post_signal, 145), 0, 1))
                if drift:
                    # Retain a detectable deployment-wide covariate shift while
                    # the conditional incident signal above creates concept drift.
                    response_p95 *= ((1 + .17 * drift_fraction)
                                     * (1 + 2.2 * severity * cfg.concept_drift_strength * drift_fraction))
                    error_rate = min(1.0, error_rate * (1 + .7 * drift_fraction))
            failure_warning = future_failure(incidents, server, minute, cfg.horizon)
            traffic_campaign = burst > 1
            task_type, criticality, latency_sla_ms, expected_route, task_cost = task_profile(
                rng, failure_warning, traffic_campaign
            )
            event: dict[str, Any] = {
                "event_id": deterministic_event_id(cfg, minute, server),
                "event_time": iso(ts), "ingest_time": iso(ts + timedelta(seconds=int(rng.integers(0, 8)))),
                "schema_version": 2 if day >= cfg.schema_v2_start_day else 1,
                "data_center_id": f"dc-{dc+1:02d}", "rack_id": f"rack-{server // 10 + 1:02d}",
                "server_id": f"dc-{dc+1:02d}-srv-{server+1:03d}", "service_name": ("feed" if server % 3 else "messaging"),
                "cpu_util_pct": round(float(cpu), 3), "memory_util_pct": round(float(memory), 3),
                "disk_util_pct": round(float(np.clip(45 + rng.normal(0, 8) + disk_incident_shift, 0, 100)), 3),
                "disk_iops": round(float(rng.lognormal(6.0, .4)), 3), "disk_queue_depth": round(float(queue), 3),
                "network_in_mbps": round(float(max(0, request_rate * rng.uniform(.02, .06))), 3),
                "network_out_mbps": round(float(max(0, request_rate * rng.uniform(.03, .08))), 3),
                "request_rate_rps": round(float(request_rate), 3), "response_p50_ms": round(response_p50, 3),
                "response_p95_ms": round(response_p95, 3), "error_rate": round(error_rate, 6),
                "timeout_rate": round(float(min(1, error_rate * rng.uniform(.1, .5))), 6), "queue_depth": round(float(queue), 3),
                "active_connections": int(max(1, request_rate * rng.uniform(1.5, 5))),
                "deployment_id": f"deploy-{day:03d}" if drift else None, "software_version": "2.0.0" if drift else "1.0.0",
                "config_version": "cfg-b" if drift else "cfg-a", "maintenance_flag": False,
                "traffic_campaign_flag": traffic_campaign, "failure_within_30min": failure_warning,
                "is_injected_defect": False, "injected_defect_category": None,
                "injected_defect_type": None, "expected_quality_action": "accept",
                "duplicate_ordinal": 0, "task_type": task_type, "criticality": criticality,
                "latency_sla_ms": latency_sla_ms, "expected_route": expected_route,
                "workload_units": round(float(request_rate * task_cost), 3),
                "source_sequence": int((minute // cfg.interval) * cfg.logical_servers + server) * 2,
            }
            if event["schema_version"] == 2:
                event.update({"cache_hit_ratio": round(float(rng.beta(9, 2)), 5), "gc_pause_ms": round(float(rng.gamma(2, 6)), 3)})
            # One labelled representative defect per affected base row keeps
            # category-level Precision/Recall unambiguous and reproducible.
            defect_category = choose_quality_defect(cfg, rng)
            defect_type = None
            if defect_category == "schema":
                event["schema_version"] = 99
                defect_type = "unsupported_schema"
            elif defect_category == "range":
                event["cpu_util_pct"] = 125.0
                defect_type = "cpu_out_of_range"
            elif defect_category == "temporal":
                event["ingest_time"] = iso(ts + timedelta(minutes=20))
                defect_type = "event_too_late"
            elif defect_category == "completeness":
                event.pop("memory_util_pct")
                defect_type = "missing_required_field"
            elif defect_category == "cross_field":
                event["response_p95_ms"] = round(max(0.001, event["response_p50_ms"] * 0.5), 3)
                defect_type = "p95_below_p50"
            if defect_category:
                event.update({
                    "is_injected_defect": True,
                    "injected_defect_category": defect_category,
                    "injected_defect_type": defect_type,
                    "expected_quality_action": "quarantine",
                })
            yield event
            # Duplicate only clean base rows, then mark the later occurrence so
            # the detector can retain the original and score the copy as truth.
            if not defect_category and rng.random() < cfg.duplicate_rate:
                duplicate = dict(event)
                duplicate.update({
                    "is_injected_defect": True,
                    "injected_defect_category": "completeness",
                    "injected_defect_type": "duplicate_event",
                    "expected_quality_action": "quarantine",
                    "duplicate_ordinal": 1,
                    "source_sequence": event["source_sequence"] + 1,
                })
                yield duplicate


def publish(rows: Iterable[dict[str, Any]], output: Path | None,
            output_dir: Path | None, bootstrap: str | None) -> int:
    producer = None
    handle = None
    current_partition = None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True); handle = output.open("w", encoding="utf-8")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if bootstrap:
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda x: json.dumps(x).encode("utf-8"))
    count = 0
    try:
        for row in rows:
            if output_dir:
                partition = row["event_time"][:10]
                if partition != current_partition:
                    if handle:
                        handle.close()
                    partition_dir = output_dir / f"event_date={partition}"
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    handle = (partition_dir / "part-00000.ndjson").open("w", encoding="utf-8")
                    current_partition = partition
            if handle: handle.write(json.dumps(row) + "\n")
            if producer: producer.send("telemetry-raw", key=row["server_id"].encode("utf-8"), value=row)
            count += 1
        if producer: producer.flush()
    finally:
        if handle: handle.close()
        if producer: producer.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="legacy single NDJSON output")
    output.add_argument("--output-dir", type=Path, help="daily partitioned NDJSON root")
    parser.add_argument("--kafka-bootstrap")
    args = parser.parse_args()
    if not args.output and not args.output_dir and not args.kafka_bootstrap:
        parser.error("choose --output, --output-dir and/or --kafka-bootstrap")
    count = publish(records(settings_from(args.config)), args.output, args.output_dir, args.kafka_bootstrap)
    print(json.dumps({
        "events_written": count,
        "output": str(args.output) if args.output else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
