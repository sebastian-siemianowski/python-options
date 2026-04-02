"""
Test Story 8.2: FOMC/Macro Event Calendar.

Validates:
  1. Event calendar creation and lookup
  2. Pre-event detection
  3. Post-event detection
  4. Normal (no event) detection
  5. Uncertainty amplification for currencies near FOMC
  6. Asset class sensitivity differences
  7. JSON serialization roundtrip
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.macro_events import (
    MacroEventCalendar,
    MacroEvent,
    detect_event_proximity,
    adjust_for_macro_event,
    EventProximity,
    UNCERTAINTY_AMPLIFICATION,
    EVENT_SENSITIVITY,
)


class TestMacroEvents(unittest.TestCase):
    """Tests for macro event calendar integration."""

    def _make_calendar(self):
        return MacroEventCalendar([
            MacroEvent("2026-03-15", "FOMC", "Rate decision", 3),
            MacroEvent("2026-03-20", "CPI", "CPI release", 2),
        ])

    def test_event_lookup_by_date(self):
        """Find events on a specific date."""
        cal = self._make_calendar()
        events = cal.get_events_on_date("2026-03-15")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "FOMC")

    def test_pre_event_detection(self):
        """T-1 before event -> PRE_EVENT."""
        prox = detect_event_proximity(current_date_idx=99, event_date_idx=100)
        self.assertTrue(prox.in_pre_event)
        self.assertEqual(prox.context_label, "PRE_EVENT")

    def test_post_event_detection(self):
        """T+0 (event day) -> POST_EVENT."""
        prox = detect_event_proximity(current_date_idx=100, event_date_idx=100)
        self.assertTrue(prox.in_post_event)
        self.assertEqual(prox.context_label, "POST_EVENT")

    def test_normal_detection(self):
        """Far from event -> NORMAL."""
        prox = detect_event_proximity(current_date_idx=50, event_date_idx=100)
        self.assertEqual(prox.context_label, "NORMAL")

    def test_no_event(self):
        """No event -> NORMAL."""
        prox = detect_event_proximity(current_date_idx=50, event_date_idx=None)
        self.assertEqual(prox.context_label, "NORMAL")

    def test_fomc_currency_high_sensitivity(self):
        """FOMC amplifies currency sigma more than equity."""
        event = MacroEvent("2026-03-15", "FOMC", "Rate decision", 3)
        prox = EventProximity(
            in_pre_event=True, in_post_event=False,
            days_to_event=1, event=event, context_label="PRE_EVENT",
        )
        
        currency = adjust_for_macro_event(1.0, 0.8, 0.01, prox, "currency")
        equity = adjust_for_macro_event(1.0, 0.8, 0.01, prox, "equity")
        
        # Currency should have wider sigma (sensitivity 1.5 vs 1.0)
        self.assertGreater(currency["sigma"], equity["sigma"])

    def test_bond_sensitivity_to_fomc(self):
        """Bonds most sensitive to FOMC."""
        fomc_sens = EVENT_SENSITIVITY["FOMC"]
        self.assertGreater(fomc_sens["bond"], fomc_sens["equity"])
        self.assertGreater(fomc_sens["bond"], fomc_sens["currency"])

    def test_json_roundtrip(self):
        """Calendar survives JSON serialization."""
        cal = self._make_calendar()
        json_str = cal.to_json()
        cal2 = MacroEventCalendar.from_json(json_str)
        
        self.assertEqual(len(cal2.events), 2)
        self.assertEqual(cal2.events[0].event_type, "FOMC")

    def test_find_nearest_event(self):
        """Finds nearest macro event in date window."""
        cal = self._make_calendar()
        dates = [f"2026-03-{d:02d}" for d in range(1, 31)]
        
        # Date index 14 = 2026-03-15 (FOMC day)
        event = cal.find_nearest_event(12, dates, look_ahead=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, "FOMC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
