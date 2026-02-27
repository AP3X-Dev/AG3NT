"""
PredictiveIntentLoop -- Action Sequence Mining and Prediction.

Monitors tool completion events on the EventBus and mines frequent action
sequences to predict the user's next likely action.  Uses sliding-window
n-gram extraction at window sizes 2-4.

Architecture
------------
EventBus (tool.completed) --> PredictiveIntentLoop --> _action_history
                                                  --> _patterns (n-gram counts)
                                                  --> emits intent.prediction
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ActionSequence:
    """A frequently occurring sequence of actions."""

    actions: tuple[str, ...]
    count: int = 0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# PredictiveIntentLoop
# ---------------------------------------------------------------------------

_MAX_HISTORY = 100
_MINE_INTERVAL = 10  # mine patterns every N actions
_MIN_WINDOW = 2
_MAX_WINDOW = 4


class PredictiveIntentLoop:
    """Reactive loop that mines action sequences and predicts next actions.

    Subscribes to ``tool.completed`` events on the :class:`EventBus`.
    Every 10 actions, mines frequent n-gram patterns from the action
    history and attempts to predict the next action with configurable
    minimum confidence.

    Emits ``intent.prediction`` when a high-confidence prediction is made.
    """

    def __init__(
        self,
        bus: EventBus,
        min_confidence: float = 0.75,
    ) -> None:
        self._bus = bus
        self._min_confidence = min_confidence

        self._action_history: list[str] = []
        self._patterns: dict[tuple[str, ...], int] = {}
        self._subscription_ids: list[str] = []
        self._running = False

        # Accuracy tracking
        self._accuracy_tracker: dict[str, Any] = {
            "predictions_made": 0,
            "predictions_correct": 0,
            "predictions_wrong": 0,
            "last_prediction": None,
            "last_prediction_time": None,
        }

        # Stats
        self._events_processed = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to tool.completed events on the bus."""
        if self._running:
            return

        self._running = True

        sub_id = self._bus.subscribe(
            handler=self._on_tool_completed,
            event_types={"tool.completed"},
        )
        self._subscription_ids.append(sub_id)

        logger.info(
            "PredictiveIntentLoop started (min_confidence=%.2f)",
            self._min_confidence,
        )

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("PredictiveIntentLoop stopped")

    # -- event handler -------------------------------------------------------

    async def _on_tool_completed(self, event: Event) -> None:
        """Handle a tool.completed event."""
        try:
            payload = event.payload or {}
            action = payload.get("tool") or payload.get("name") or event.source
            if not action:
                return

            # Check if previous prediction was correct
            if self._accuracy_tracker["last_prediction"] is not None:
                if action == self._accuracy_tracker["last_prediction"]:
                    self._accuracy_tracker["predictions_correct"] += 1
                else:
                    self._accuracy_tracker["predictions_wrong"] += 1
                self._accuracy_tracker["last_prediction"] = None

            # Record the action
            self._action_history.append(action)

            # Trim history to max size
            if len(self._action_history) > _MAX_HISTORY:
                self._action_history = self._action_history[-_MAX_HISTORY:]

            self._events_processed += 1

            # Mine patterns every N actions
            if self._events_processed % _MINE_INTERVAL == 0:
                self._mine_patterns()

                # Try to predict next action
                prediction = self._predict_next()
                if prediction is not None:
                    predicted_action, confidence = prediction
                    self._accuracy_tracker["predictions_made"] += 1
                    self._accuracy_tracker["last_prediction"] = predicted_action
                    self._accuracy_tracker["last_prediction_time"] = time.time()

                    # Emit prediction event
                    await self._bus.publish(Event(
                        event_type="intent.prediction",
                        source="predictive_intent",
                        payload={
                            "predicted_action": predicted_action,
                            "confidence": confidence,
                            "based_on": list(self._action_history[-_MAX_WINDOW:]),
                        },
                    ))

                    logger.debug(
                        "Predicted next action: %s (confidence=%.2f)",
                        predicted_action,
                        confidence,
                    )

        except Exception:
            logger.exception("Error processing tool.completed event")

    # -- pattern mining ------------------------------------------------------

    def _mine_patterns(self) -> None:
        """Extract frequent n-gram sequences from the action history.

        Scans the history with sliding windows of sizes 2 through 4,
        counting occurrences of each sequence.
        """
        self._patterns.clear()
        history = self._action_history

        for window_size in range(_MIN_WINDOW, _MAX_WINDOW + 1):
            if len(history) < window_size:
                continue

            for i in range(len(history) - window_size + 1):
                seq = tuple(history[i : i + window_size])
                self._patterns[seq] = self._patterns.get(seq, 0) + 1

    def _predict_next(self) -> tuple[str, float] | None:
        """Predict the next action based on mined patterns.

        Looks at the most recent actions and finds the most frequent
        continuation.  Returns ``(action, confidence)`` if the confidence
        is at least ``_min_confidence``, otherwise ``None``.
        """
        if not self._action_history or not self._patterns:
            return None

        best_action: str | None = None
        best_confidence: float = 0.0

        # Try each context length from longest to shortest
        for ctx_len in range(_MAX_WINDOW - 1, _MIN_WINDOW - 2, -1):
            if ctx_len < 1:
                continue
            if len(self._action_history) < ctx_len:
                continue

            context = tuple(self._action_history[-ctx_len:])

            # Count how many times this context appears as a prefix
            continuations: dict[str, int] = defaultdict(int)
            total = 0

            for seq, count in self._patterns.items():
                if len(seq) == ctx_len + 1 and seq[:ctx_len] == context:
                    next_action = seq[ctx_len]
                    continuations[next_action] += count
                    total += count

            if total == 0:
                continue

            # Find the most frequent continuation
            for action, count in continuations.items():
                confidence = count / total
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_action = action

            # If we found a confident prediction, return it
            if best_confidence >= self._min_confidence and best_action is not None:
                return best_action, best_confidence

        # Return best prediction if it meets minimum confidence
        if best_action is not None and best_confidence >= self._min_confidence:
            return best_action, best_confidence

        return None

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current loop status and statistics."""
        accuracy = 0.0
        total_checked = (
            self._accuracy_tracker["predictions_correct"]
            + self._accuracy_tracker["predictions_wrong"]
        )
        if total_checked > 0:
            accuracy = self._accuracy_tracker["predictions_correct"] / total_checked

        return {
            "running": self._running,
            "subscriptions": len(self._subscription_ids),
            "events_processed": self._events_processed,
            "action_history_size": len(self._action_history),
            "patterns_count": len(self._patterns),
            "min_confidence": self._min_confidence,
            "accuracy": round(accuracy, 4),
            "predictions_made": self._accuracy_tracker["predictions_made"],
            "predictions_correct": self._accuracy_tracker["predictions_correct"],
            "predictions_wrong": self._accuracy_tracker["predictions_wrong"],
            "last_prediction": self._accuracy_tracker["last_prediction"],
        }
