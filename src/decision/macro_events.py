"""
Story 8.2: FOMC/Macro Event Calendar Integration.

Detects proximity to macro events (FOMC, CPI, NFP, GDP) and adjusts signals:
  - Pre-event: Amplify uncertainty
  - Post-event: Accelerated regime reassessment
  - Event-type conditioning: which asset classes are most affected

Usage:
    from decision.macro_events import (
        MacroEventCalendar,
        detect_event_proximity,
        adjust_for_macro_event,
    )
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Configuration
PRE_EVENT_WINDOW = 2           # days before event
POST_EVENT_WINDOW = 1          # days after event
UNCERTAINTY_AMPLIFICATION = 1.5  # pre-event sigma multiplier
POST_EVENT_REGIME_BOOST = 2.0   # faster regime reassessment


@dataclass
class MacroEvent:
    """A scheduled macro event."""
    date: str            # "YYYY-MM-DD"
    event_type: str      # "FOMC", "CPI", "NFP", "GDP", "ECB", "BOJ"
    description: str
    importance: int      # 1=low, 2=medium, 3=high


@dataclass
class EventProximity:
    """Proximity context for a macro event."""
    in_pre_event: bool
    in_post_event: bool
    days_to_event: Optional[int]
    event: Optional[MacroEvent]
    context_label: str   # "PRE_EVENT", "POST_EVENT", "NORMAL"


# Asset class sensitivity to event types
EVENT_SENSITIVITY: Dict[str, Dict[str, float]] = {
    "FOMC": {
        "equity": 1.0,
        "currency": 1.5,
        "bond": 2.0,
        "commodity": 0.8,
        "metal": 0.7,
    },
    "CPI": {
        "equity": 0.8,
        "currency": 1.2,
        "bond": 1.5,
        "commodity": 1.0,
        "metal": 1.3,
    },
    "NFP": {
        "equity": 1.2,
        "currency": 1.3,
        "bond": 1.0,
        "commodity": 0.5,
        "metal": 0.5,
    },
    "GDP": {
        "equity": 1.0,
        "currency": 1.0,
        "bond": 1.0,
        "commodity": 0.7,
        "metal": 0.5,
    },
    "ECB": {
        "equity": 0.6,
        "currency": 1.8,
        "bond": 1.2,
        "commodity": 0.5,
        "metal": 0.5,
    },
    "BOJ": {
        "equity": 0.5,
        "currency": 2.0,
        "bond": 0.8,
        "commodity": 0.4,
        "metal": 0.4,
    },
}

# Default sensitivity for unknown event types
DEFAULT_SENSITIVITY = 1.0


class MacroEventCalendar:
    """Calendar of macro events with lookup by date index."""
    
    def __init__(self, events: Optional[List[MacroEvent]] = None):
        self.events: List[MacroEvent] = events or []
        self._date_index: Dict[str, List[MacroEvent]] = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Build date -> events lookup."""
        self._date_index.clear()
        for e in self.events:
            self._date_index.setdefault(e.date, []).append(e)
    
    def add_event(self, event: MacroEvent):
        """Add an event to the calendar."""
        self.events.append(event)
        self._date_index.setdefault(event.date, []).append(event)
    
    def get_events_on_date(self, date: str) -> List[MacroEvent]:
        """Get all events on a specific date."""
        return self._date_index.get(date, [])
    
    def find_nearest_event(
        self,
        current_date_idx: int,
        date_list: List[str],
        look_ahead: int = 10,
        look_back: int = 5,
    ) -> Optional[MacroEvent]:
        """Find nearest event within a date window."""
        if not date_list:
            return None
        
        best_event = None
        best_dist = float("inf")
        
        for delta in range(-look_back, look_ahead + 1):
            idx = current_date_idx + delta
            if 0 <= idx < len(date_list):
                date = date_list[idx]
                events = self.get_events_on_date(date)
                if events:
                    dist = abs(delta)
                    if dist < best_dist:
                        best_dist = dist
                        # Highest importance event on that day
                        best_event = max(events, key=lambda e: e.importance)
        
        return best_event
    
    def to_json(self) -> str:
        """Serialize calendar to JSON."""
        return json.dumps(
            [{"date": e.date, "event_type": e.event_type,
              "description": e.description, "importance": e.importance}
             for e in self.events],
            indent=2,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "MacroEventCalendar":
        """Deserialize calendar from JSON."""
        data = json.loads(json_str)
        events = [MacroEvent(**d) for d in data]
        return cls(events)


def detect_event_proximity(
    current_date_idx: int,
    event_date_idx: Optional[int],
    event: Optional[MacroEvent] = None,
) -> EventProximity:
    """
    Detect if current date is in pre/post event window.
    
    Args:
        current_date_idx: Current position in time series.
        event_date_idx: Position of nearest event.
        event: The macro event (if known).
    
    Returns:
        EventProximity context.
    """
    if event_date_idx is None:
        return EventProximity(
            in_pre_event=False,
            in_post_event=False,
            days_to_event=None,
            event=None,
            context_label="NORMAL",
        )
    
    days_to = event_date_idx - current_date_idx
    
    in_pre = 0 < days_to <= PRE_EVENT_WINDOW
    in_post = -POST_EVENT_WINDOW <= days_to <= 0
    
    if in_pre:
        label = "PRE_EVENT"
    elif in_post:
        label = "POST_EVENT"
    else:
        label = "NORMAL"
    
    return EventProximity(
        in_pre_event=in_pre,
        in_post_event=in_post,
        days_to_event=days_to,
        event=event,
        context_label=label,
    )


def adjust_for_macro_event(
    forecast_pct: float,
    confidence: float,
    sigma: float,
    proximity: EventProximity,
    asset_class: str = "equity",
) -> Dict[str, float]:
    """
    Adjust signal parameters for macro event proximity.
    
    Pre-event: Amplify uncertainty proportional to event sensitivity.
    Post-event: Boost regime reassessment speed (via sigma scaling).
    
    Args:
        forecast_pct: Forecast return.
        confidence: Forecast confidence.
        sigma: Forecast standard deviation.
        proximity: Event proximity context.
        asset_class: Asset class for sensitivity lookup.
    
    Returns:
        Dict with adjusted parameters.
    """
    adj_forecast = forecast_pct
    adj_confidence = confidence
    adj_sigma = sigma
    
    # Get sensitivity for this event type and asset class
    sensitivity = DEFAULT_SENSITIVITY
    if proximity.event:
        event_sens = EVENT_SENSITIVITY.get(proximity.event.event_type, {})
        sensitivity = event_sens.get(asset_class, DEFAULT_SENSITIVITY)
    
    if proximity.in_pre_event:
        # Amplify uncertainty
        adj_sigma = sigma * UNCERTAINTY_AMPLIFICATION * sensitivity
        adj_confidence = confidence * (1.0 / (1.0 + 0.3 * sensitivity))
    
    elif proximity.in_post_event:
        # Post-event: faster regime reassessment (wider sigma temporarily)
        adj_sigma = sigma * (1.0 + 0.3 * sensitivity)
    
    return {
        "forecast_pct": adj_forecast,
        "confidence": max(0.0, min(1.0, adj_confidence)),
        "sigma": adj_sigma,
    }
