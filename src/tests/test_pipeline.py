"""
Test Story 4.4: Pipeline Orchestration.

Validates:
  1. Full pipeline completes end-to-end
  2. Phase failure stops pipeline
  3. Tune partial failure -> warning
  4. Pipeline status saved to JSON
  5. Phase timing recorded
  6. Single phase execution
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.pipeline import (
    run_pipeline,
    PipelineResult,
    PhaseResult,
    load_pipeline_status,
    PIPELINE_STATUS_PATH,
)


class TestPipelineOrchestration(unittest.TestCase):
    """Tests for pipeline orchestration."""

    def test_full_pipeline_success(self):
        """All phases succeed -> overall success."""
        def ok_phase():
            return {"assets_processed": 10, "errors": [], "total_assets": 10}
        
        result = run_pipeline(
            phases=["prices", "tune", "signals"],
            phase_functions={
                "prices": ok_phase,
                "tune": ok_phase,
                "signals": ok_phase,
            },
        )
        self.assertEqual(result.overall_status, "success")
        self.assertIsNone(result.failed_phase)
        self.assertEqual(len(result.phases), 3)

    def test_phase_failure_stops_pipeline(self):
        """Price phase failure stops everything."""
        def fail_phase():
            raise RuntimeError("Network error")
        
        def ok_phase():
            return {"assets_processed": 10, "errors": []}
        
        result = run_pipeline(
            phases=["prices", "tune", "signals"],
            phase_functions={
                "prices": fail_phase,
                "tune": ok_phase,
                "signals": ok_phase,
            },
        )
        self.assertEqual(result.overall_status, "error")
        self.assertEqual(result.failed_phase, "prices")
        # tune and signals should not have run
        self.assertNotIn("tune", result.phases)
        self.assertNotIn("signals", result.phases)

    def test_tune_partial_fail_warning(self):
        """Tune with <50% failures -> warning, continues."""
        def ok_phase():
            return {"assets_processed": 10, "errors": []}
        
        def tune_partial():
            return {
                "assets_processed": 10,
                "errors": [f"fail_{i}" for i in range(10)],
                "total_assets": 100,
            }
        
        result = run_pipeline(
            phases=["prices", "tune", "signals"],
            phase_functions={
                "prices": ok_phase,
                "tune": tune_partial,
                "signals": ok_phase,
            },
        )
        self.assertEqual(result.overall_status, "warning")
        self.assertIsNone(result.failed_phase)

    def test_tune_massive_fail_stops(self):
        """Tune with >50% failures -> error, stops."""
        def ok_phase():
            return {"assets_processed": 10, "errors": []}
        
        def tune_massive_fail():
            return {
                "assets_processed": 10,
                "errors": [f"fail_{i}" for i in range(60)],
                "total_assets": 100,
            }
        
        result = run_pipeline(
            phases=["prices", "tune", "signals"],
            phase_functions={
                "prices": ok_phase,
                "tune": tune_massive_fail,
                "signals": ok_phase,
            },
        )
        self.assertEqual(result.overall_status, "error")
        self.assertEqual(result.failed_phase, "tune")

    def test_timing_recorded(self):
        """Phase duration is recorded."""
        def ok_phase():
            return {"assets_processed": 5, "errors": []}
        
        result = run_pipeline(
            phases=["prices"],
            phase_functions={"prices": ok_phase},
        )
        pr = result.phases["prices"]
        self.assertGreater(pr.duration_seconds, 0)
        self.assertNotEqual(pr.started_at, "")
        self.assertNotEqual(pr.completed_at, "")

    def test_missing_phase_function_skipped(self):
        """Phase with no function -> skipped."""
        result = run_pipeline(
            phases=["prices", "tune"],
            phase_functions={},
        )
        self.assertEqual(result.phases["prices"].status, "skipped")

    def test_pipeline_result_serializable(self):
        """PipelineResult.to_dict() produces valid JSON."""
        def ok_phase():
            return {"assets_processed": 1, "errors": []}
        
        result = run_pipeline(
            phases=["prices"],
            phase_functions={"prices": ok_phase},
        )
        d = result.to_dict()
        json_str = json.dumps(d, default=str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["overall_status"], "success")

    def test_single_phase(self):
        """Run only one phase."""
        call_count = {"n": 0}
        def counting_phase():
            call_count["n"] += 1
            return {"assets_processed": 1, "errors": []}
        
        result = run_pipeline(
            phases=["tune"],
            phase_functions={"tune": counting_phase},
        )
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(result.overall_status, "success")


if __name__ == "__main__":
    unittest.main(verbosity=2)
