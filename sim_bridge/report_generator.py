"""Mission report generator for ResQ-AI.

Aggregates data from all subsystems (fire, civilians, YOLO, Cosmos,
drone telemetry) into structured JSON reports.

Isaac Sim 5.1 compatibility: NO f-strings.
"""

import json
import os
import time
import uuid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPORT_INTERVAL_S = 5.0
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator(object):
    """Collects data from all subsystems and writes mission reports.

    Parameters
    ----------
    reports_dir : str or None
        Directory to write JSON report files.  Defaults to
        ``sim_bridge/reports/``.
    """

    def __init__(self, reports_dir=None):
        self._reports_dir = reports_dir or REPORTS_DIR
        if not os.path.isdir(self._reports_dir):
            os.makedirs(self._reports_dir, exist_ok=True)

        self._mission_id = str(uuid.uuid4())[:8]
        self._start_time = time.time()
        self._reports = []
        self._last_report_time = 0.0
        self._ai_decisions = []

        print("[ReportGen] Mission " + self._mission_id +
              " — reports dir: " + self._reports_dir)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def should_generate(self):
        """Return True if enough time has elapsed since last report."""
        return (time.time() - self._last_report_time) >= REPORT_INTERVAL_S

    def generate(self, fire_report=None, civilian_report=None,
                 detections=None, cosmos_decisions=None,
                 drone_position=None, drone_battery=None,
                 drone_status="patrolling"):
        """Create and store a mission report.

        Parameters
        ----------
        fire_report : dict or None
            From FireManager.get_fire_report().
        civilian_report : dict or None
            From CivilianTracker.get_civilian_report().
        detections : list[dict] or None
            From DualYOLODetector.detect().
        cosmos_decisions : list[dict] or None
            Recent decisions from CosmosNavigator.
        drone_position : list or None
            [x, y, z] current drone position.
        drone_battery : float or None
            Battery percentage.
        drone_status : str
            One of: patrolling, investigating, hovering, returning.

        Returns
        -------
        dict
            The generated report.
        """
        now = time.time()
        elapsed = now - self._start_time
        self._last_report_time = now

        # Count detections in this interval
        people_spotted = 0
        fire_confirmed = 0
        if detections is not None:
            di = 0
            while di < len(detections):
                d = detections[di]
                if d["class"] == "person":
                    people_spotted = people_spotted + 1
                elif d["class"] == "fire" and d.get("confirmed", False):
                    fire_confirmed = fire_confirmed + 1
                di = di + 1

        # Fire data
        fire_data = {
            "active_count": 0,
            "total_area_m2": 0.0,
            "spread_rate": 0.0,
            "zones": [],
        }
        if fire_report:
            fire_data["active_count"] = len(
                fire_report.get("active_fires", []))
            fire_data["total_area_m2"] = fire_report.get(
                "total_area_burning_m2", 0.0)
            fire_data["spread_rate"] = fire_report.get(
                "spread_rate_m_per_min", 0.0)
            fire_data["zones"] = fire_report.get("active_fires", [])

        # Civilian data
        civ_data = {
            "total": 0,
            "safe": 0,
            "in_danger": 0,
            "injured": 0,
            "incapacitated": 0,
            "rescued": 0,
        }
        if civilian_report:
            civ_data["total"] = civilian_report.get("total", 0)
            bs = civilian_report.get("by_state", {})
            civ_data["safe"] = bs.get("idle", 0) + bs.get("alert", 0)
            civ_data["in_danger"] = civilian_report.get("critical_danger", 0)
            civ_data["injured"] = bs.get("injured", 0)
            civ_data["incapacitated"] = bs.get("incapacitated", 0)
            civ_data["rescued"] = bs.get("rescued", 0)

        # Determine urgency level
        urgency = "low"
        if civ_data["in_danger"] > 0 or civ_data["injured"] > 0:
            urgency = "high"
        if civ_data["incapacitated"] > 0:
            urgency = "critical"
        elif fire_data["active_count"] > 2:
            urgency = "medium"

        # AI decisions
        ai_decs = []
        if cosmos_decisions:
            ai_decs = cosmos_decisions

        # Recommended actions
        actions = []
        if fire_data["active_count"] > 0:
            actions.append("Deploy fire suppression units")
        if civ_data["in_danger"] > 0:
            actions.append("Evacuate civilians in danger zone")
        if civ_data["injured"] > 0:
            actions.append("Dispatch medical teams for " +
                           str(civ_data["injured"]) + " injured")
        if urgency == "critical":
            actions.append("CRITICAL: Immediate emergency response required")

        report = {
            "mission_id": self._mission_id,
            "timestamp": now,
            "elapsed_seconds": round(elapsed, 1),
            "drone": {
                "position": drone_position or [0, 0, 50],
                "battery_pct": drone_battery or 100.0,
                "status": drone_status,
            },
            "fires": fire_data,
            "civilians": civ_data,
            "detections_this_interval": {
                "people_spotted": people_spotted,
                "fire_confirmed": fire_confirmed,
            },
            "ai_decisions": ai_decs,
            "urgency_level": urgency,
            "recommended_actions": actions,
        }

        self._reports.append(report)

        # Write to disk
        self._write_report(report)

        return report

    def _write_report(self, report):
        """Write a single report as JSON file."""
        ts = str(int(report["timestamp"]))
        filename = "mission_" + ts + ".json"
        filepath = os.path.join(self._reports_dir, filename)

        try:
            with open(filepath, "w") as fh:
                json.dump(report, fh, indent=2, default=str)
        except Exception as e:
            print("[ReportGen] Write error: " + str(e))

        # Also write latest.json for easy access
        latest_path = os.path.join(self._reports_dir, "latest.json")
        try:
            with open(latest_path, "w") as fh:
                json.dump(report, fh, indent=2, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_latest_report(self):
        """Return the most recent report or empty dict."""
        if self._reports:
            return self._reports[-1]
        return {}

    def get_all_reports(self):
        """Return all reports generated this mission."""
        return list(self._reports)

    def get_mission_id(self):
        """Return the mission ID string."""
        return self._mission_id
