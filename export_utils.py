"""
Proctoring Audit Report Exporters.
Generates CSV incident logs and JSON telemetry packages for LMS / Canvas / Moodle integration.
"""

import json
import csv
import io
from typing import Dict, Any, List
from datetime import datetime


def generate_csv_audit_report(
    candidate_id: str,
    exam_title: str,
    infractions: List[Dict[str, Any]]
) -> str:
    """
    Serializes timestamped infraction incidents to an institutional CSV format.
    """
    output = io.StringIO()
    fieldnames = [
        "incident_id",
        "timestamp",
        "elapsed_sec",
        "candidate_id",
        "exam_title",
        "violation_type",
        "severity",
        "suspicion_index",
        "description"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for idx, inf in enumerate(infractions, start=1):
        writer.writerow({
            "incident_id": f"INC-{idx:04d}",
            "timestamp": inf.get("timestamp", datetime.now().strftime("%H:%M:%S")),
            "elapsed_sec": inf.get("elapsed_seconds", 0),
            "candidate_id": candidate_id,
            "exam_title": exam_title,
            "violation_type": inf.get("type", "UNKNOWN"),
            "severity": inf.get("severity", "WARNING"),
            "suspicion_index": inf.get("suspicion_index", 0.0),
            "description": inf.get("description", "")
        })

    return output.getvalue()


def generate_json_audit_package(
    candidate_id: str,
    exam_title: str,
    candidate_integrity_score: float,
    infractions: List[Dict[str, Any]]
) -> str:
    """
    Creates full JSON telemetry manifest suitable for LMS gradebook verification.
    """
    payload = {
        "institution_proctor_engine": "ProctorAI-v2.4",
        "report_generated_at": datetime.now().isoformat(),
        "candidate_id": candidate_id,
        "exam_title": exam_title,
        "integrity_score_pct": round(candidate_integrity_score, 1),
        "status": "APPROVED" if candidate_integrity_score >= 60.0 else "FLAGGED_FOR_MANUAL_REVIEW",
        "total_infractions": len(infractions),
        "infraction_audit_trail": infractions
    }
    return json.dumps(payload, indent=2)
