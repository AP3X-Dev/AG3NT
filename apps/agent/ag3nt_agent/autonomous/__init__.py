"""
AG3NT Autonomous System

Event-driven autonomous agent capabilities with:
- Event Bus for routing events from sources to handlers
- Goal Manager for YAML-based goal configuration
- Decision Engine for risk/confidence evaluation
- Learning Engine backed by Context-Engine semantic memory
- Event Sources for monitoring HTTP, files, and logs

Usage:
    from ag3nt_agent.autonomous import (
        EventBus,
        Event,
        GoalManager,
        DecisionEngine,
        LearningEngine,
        AutonomousMiddleware
    )

    # Initialize the autonomous system
    event_bus = EventBus()
    goal_manager = GoalManager()
    learning_engine = LearningEngine()
    decision_engine = DecisionEngine(learning_engine)

    # Start event processing
    await event_bus.start()
"""

from .event_bus import EventBus, Event, EventPriority, get_event_bus
from .learning_engine import LearningEngine
from .goal_manager import GoalManager, Goal
from .decision_engine import DecisionEngine, Decision, DecisionType

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "EventPriority",
    "get_event_bus",
    # Learning Engine
    "LearningEngine",
    # Goal Manager
    "GoalManager",
    "Goal",
    # Decision Engine
    "DecisionEngine",
    "Decision",
    "DecisionType",
]
