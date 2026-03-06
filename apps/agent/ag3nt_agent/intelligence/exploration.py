"""
ParallelExplorationLoop -- Multi-Strategy Problem Exploration.

Generates multiple candidate strategies for a given problem, evaluates
them in parallel, and selects the best one.  Exploration history is
persisted to a WAL-mode SQLite database at ``~/.ag3nt/exploration.db``.

Architecture
------------
EventBus --> ParallelExplorationLoop --> ExplorationDB (SQLite WAL)
                                    --> emits exploration.completed
                                    --> emits exploration.strategy_selected
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExplorationStrategy:
    """A single candidate strategy for solving a problem."""

    id: str = ""
    description: str = ""
    approach: str = ""
    confidence: float = 0.0
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "approach": self.approach,
            "confidence": self.confidence,
            "pros": list(self.pros),
            "cons": list(self.cons),
        }


@dataclass
class ExplorationResult:
    """Judgment result for one strategy after evaluation."""

    strategy_id: str = ""
    score: float = 0.0
    reasoning: str = ""
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "score": self.score,
            "reasoning": self.reasoning,
            "selected": self.selected,
        }


# ---------------------------------------------------------------------------
# ExplorationDB -- WAL-mode SQLite
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".ag3nt", "exploration.db"
)


class ExplorationDB:
    """Persistent store for exploration history and strategy judgments.

    Uses SQLite in WAL mode at ``~/.ag3nt/exploration.db``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    # -- lifecycle -----------------------------------------------------------

    def init_db(self) -> None:
        """Create tables and enable WAL mode."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS explorations (
                id         TEXT PRIMARY KEY,
                query      TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS strategies (
                id              TEXT PRIMARY KEY,
                exploration_id  TEXT NOT NULL,
                description     TEXT NOT NULL,
                approach        TEXT NOT NULL DEFAULT '',
                confidence      REAL NOT NULL DEFAULT 0.0,
                score           REAL,
                FOREIGN KEY (exploration_id) REFERENCES explorations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_strategies_exploration
                ON strategies(exploration_id);

            CREATE TABLE IF NOT EXISTS judgments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                exploration_id  TEXT NOT NULL,
                strategy_id     TEXT NOT NULL,
                score           REAL NOT NULL DEFAULT 0.0,
                reasoning       TEXT NOT NULL DEFAULT '',
                selected        INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (exploration_id) REFERENCES explorations(id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            );

            CREATE INDEX IF NOT EXISTS idx_judgments_exploration
                ON judgments(exploration_id);
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- explorations --------------------------------------------------------

    def create_exploration(self, query: str) -> str:
        """Create a new exploration record and return its id."""
        assert self._conn is not None
        exploration_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO explorations (id, query, timestamp, status) VALUES (?, ?, ?, ?)",
            (exploration_id, query, now, "pending"),
        )
        self._conn.commit()
        return exploration_id

    def update_exploration_status(self, exploration_id: str, status: str) -> None:
        """Update the status of an exploration."""
        assert self._conn is not None
        self._conn.execute(
            "UPDATE explorations SET status = ? WHERE id = ?",
            (status, exploration_id),
        )
        self._conn.commit()

    # -- strategies ----------------------------------------------------------

    def add_strategy(self, exploration_id: str, strategy: ExplorationStrategy) -> None:
        """Insert a strategy record linked to an exploration."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO strategies (id, exploration_id, description, approach, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                strategy.id,
                exploration_id,
                strategy.description,
                strategy.approach,
                strategy.confidence,
            ),
        )
        self._conn.commit()

    def update_strategy_score(self, strategy_id: str, score: float) -> None:
        """Set the final score for a strategy."""
        assert self._conn is not None
        self._conn.execute(
            "UPDATE strategies SET score = ? WHERE id = ?",
            (score, strategy_id),
        )
        self._conn.commit()

    # -- judgments ------------------------------------------------------------

    def record_judgment(self, exploration_id: str, result: ExplorationResult) -> None:
        """Record a judgment for a strategy within an exploration."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO judgments (exploration_id, strategy_id, score, reasoning, selected)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                exploration_id,
                result.strategy_id,
                result.score,
                result.reasoning,
                1 if result.selected else 0,
            ),
        )
        # Also update the strategy's score
        self.update_strategy_score(result.strategy_id, result.score)
        self._conn.commit()

    # -- queries -------------------------------------------------------------

    def get_exploration(self, exploration_id: str) -> dict[str, Any] | None:
        """Return full exploration details including strategies and judgments."""
        assert self._conn is not None

        row = self._conn.execute(
            "SELECT * FROM explorations WHERE id = ?", (exploration_id,)
        ).fetchone()
        if row is None:
            return None

        strategies = self._conn.execute(
            "SELECT * FROM strategies WHERE exploration_id = ?",
            (exploration_id,),
        ).fetchall()

        judgments = self._conn.execute(
            "SELECT * FROM judgments WHERE exploration_id = ?",
            (exploration_id,),
        ).fetchall()

        return {
            "id": row["id"],
            "query": row["query"],
            "timestamp": row["timestamp"],
            "status": row["status"],
            "strategies": [
                {
                    "id": s["id"],
                    "description": s["description"],
                    "approach": s["approach"],
                    "confidence": s["confidence"],
                    "score": s["score"],
                }
                for s in strategies
            ],
            "judgments": [
                {
                    "strategy_id": j["strategy_id"],
                    "score": j["score"],
                    "reasoning": j["reasoning"],
                    "selected": bool(j["selected"]),
                }
                for j in judgments
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate exploration statistics."""
        assert self._conn is not None

        total_explorations = self._conn.execute(
            "SELECT COUNT(*) FROM explorations"
        ).fetchone()[0]

        completed = self._conn.execute(
            "SELECT COUNT(*) FROM explorations WHERE status = 'completed'"
        ).fetchone()[0]

        total_strategies = self._conn.execute(
            "SELECT COUNT(*) FROM strategies"
        ).fetchone()[0]

        total_judgments = self._conn.execute(
            "SELECT COUNT(*) FROM judgments"
        ).fetchone()[0]

        avg_score_row = self._conn.execute(
            "SELECT AVG(score) FROM judgments"
        ).fetchone()
        avg_score = avg_score_row[0] if avg_score_row[0] is not None else 0.0

        selected_count = self._conn.execute(
            "SELECT COUNT(*) FROM judgments WHERE selected = 1"
        ).fetchone()[0]

        return {
            "total_explorations": total_explorations,
            "completed_explorations": completed,
            "total_strategies": total_strategies,
            "total_judgments": total_judgments,
            "average_score": round(avg_score, 2),
            "selected_strategies": selected_count,
        }


# ---------------------------------------------------------------------------
# ParallelExplorationLoop -- reactive exploration loop
# ---------------------------------------------------------------------------


class ParallelExplorationLoop:
    """Generates multiple strategies, evaluates them, and selects the best.

    Subscribes to ``exploration.requested`` events on the :class:`EventBus`.
    For each request, generates candidate strategies, judges them, selects
    the winner, and persists the results to :class:`ExplorationDB`.

    Emits:
    - ``exploration.completed`` after an exploration finishes
    - ``exploration.strategy_selected`` with the winning strategy details

    Usage::

        bus = EventBus()
        loop = ParallelExplorationLoop(bus)
        await loop.start()
        result = await loop.explore("How to optimise DB queries?")
        await loop.stop()
    """

    def __init__(
        self,
        bus: EventBus,
        db: ExplorationDB | None = None,
    ) -> None:
        self._bus = bus
        self._db = db or ExplorationDB()

        self._subscription_ids: list[str] = []
        self._running = False

        # Stats
        self._explorations_run = 0
        self._strategies_generated = 0
        self._strategies_selected = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to exploration.requested events."""
        if self._running:
            return

        self._running = True

        sub_id = self._bus.subscribe(
            handler=self._on_exploration_requested,
            event_types={"exploration.requested"},
        )
        self._subscription_ids.append(sub_id)

        logger.info(
            "ParallelExplorationLoop started (subscriptions=%d)",
            len(self._subscription_ids),
        )

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("ParallelExplorationLoop stopped")

    # -- event handler -------------------------------------------------------

    async def _on_exploration_requested(self, event: Event) -> None:
        """Handle an exploration.requested event."""
        payload = event.payload or {}
        query = payload.get("query", "")
        if not query:
            return

        strategies_data = payload.get("strategies")
        strategies: list[ExplorationStrategy] | None = None
        if strategies_data and isinstance(strategies_data, list):
            strategies = [
                ExplorationStrategy(
                    description=s.get("description", ""),
                    approach=s.get("approach", ""),
                    confidence=s.get("confidence", 0.0),
                    pros=s.get("pros", []),
                    cons=s.get("cons", []),
                )
                for s in strategies_data
                if isinstance(s, dict)
            ]

        try:
            await self.explore(query, strategies)
        except Exception:
            logger.exception("Error during exploration for query: %s", query)

    # -- core exploration ----------------------------------------------------

    async def explore(
        self,
        query: str,
        strategies: list[ExplorationStrategy] | None = None,
    ) -> ExplorationResult | None:
        """Run a full exploration cycle for *query*.

        Parameters
        ----------
        query:
            The problem statement or question to explore.
        strategies:
            Optional pre-defined strategies.  If not provided,
            :meth:`_generate_strategies` is called to produce candidates.

        Returns
        -------
        The winning :class:`ExplorationResult`, or ``None`` if no viable
        strategy was found.
        """
        exploration_id = self._db.create_exploration(query)
        self._db.update_exploration_status(exploration_id, "in_progress")

        logger.info(
            "Exploration started: id=%s query=%s",
            exploration_id, query[:80],
        )

        # 1. Generate or accept strategies
        if strategies is None or len(strategies) == 0:
            strategies = self._generate_strategies(query)

        if not strategies:
            self._db.update_exploration_status(exploration_id, "completed")
            logger.warning("No strategies generated for query: %s", query[:80])
            return None

        # Persist strategies
        for strategy in strategies:
            self._db.add_strategy(exploration_id, strategy)
            self._strategies_generated += 1

        # 2. Judge all strategies
        results = self._judge_strategies(strategies)

        if not results:
            self._db.update_exploration_status(exploration_id, "completed")
            logger.warning("No judgment results for exploration %s", exploration_id)
            return None

        # 3. Select the best strategy
        winner = self._select_best(results)

        # 4. Record all judgments
        for result in results:
            self._db.record_judgment(exploration_id, result)

        # 5. Mark exploration as completed
        self._db.update_exploration_status(exploration_id, "completed")
        self._explorations_run += 1
        self._strategies_selected += 1

        # 6. Emit events
        winning_strategy = next(
            (s for s in strategies if s.id == winner.strategy_id), None
        )

        await self._bus.publish(Event(
            event_type="exploration.completed",
            source="parallel_exploration",
            payload={
                "exploration_id": exploration_id,
                "query": query,
                "strategies_count": len(strategies),
                "winner_id": winner.strategy_id,
                "winner_score": winner.score,
            },
        ))

        if winning_strategy is not None:
            await self._bus.publish(Event(
                event_type="exploration.strategy_selected",
                source="parallel_exploration",
                payload={
                    "exploration_id": exploration_id,
                    "strategy": winning_strategy.to_dict(),
                    "result": winner.to_dict(),
                },
            ))

        logger.info(
            "Exploration completed: id=%s winner=%s score=%.2f",
            exploration_id, winner.strategy_id, winner.score,
        )

        return winner

    # -- strategy generation (placeholder) -----------------------------------

    def _generate_strategies(self, query: str) -> list[ExplorationStrategy]:
        """Generate candidate strategies for *query*.

        This is a placeholder implementation that produces three default
        strategies.  A future version will dispatch to an LLM to generate
        context-aware strategies.
        """
        return [
            ExplorationStrategy(
                description="Direct approach",
                approach="Tackle the problem head-on with the most straightforward solution.",
                confidence=0.7,
                pros=["Simple to implement", "Easy to understand"],
                cons=["May miss edge cases"],
            ),
            ExplorationStrategy(
                description="Incremental approach",
                approach="Break the problem into smaller steps and solve each incrementally.",
                confidence=0.8,
                pros=["Lower risk", "Easier to test", "Reversible"],
                cons=["Slower to complete", "May over-decompose"],
            ),
            ExplorationStrategy(
                description="Research-first approach",
                approach="Investigate existing solutions and best practices before implementing.",
                confidence=0.6,
                pros=["Leverages existing knowledge", "Avoids common pitfalls"],
                cons=["Takes more time upfront", "Risk of analysis paralysis"],
            ),
        ]

    # -- strategy judging ----------------------------------------------------

    def _judge_strategies(
        self, strategies: list[ExplorationStrategy]
    ) -> list[ExplorationResult]:
        """Score each strategy and produce judgment results.

        This is a heuristic implementation that uses the strategy's
        confidence and pros/cons counts.  A future version will use
        LLM-based evaluation.
        """
        results: list[ExplorationResult] = []
        for strategy in strategies:
            # Heuristic scoring: base from confidence, adjusted by pros/cons
            pros_bonus = len(strategy.pros) * 0.05
            cons_penalty = len(strategy.cons) * 0.03
            raw_score = strategy.confidence + pros_bonus - cons_penalty

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, raw_score))

            results.append(ExplorationResult(
                strategy_id=strategy.id,
                score=round(score, 4),
                reasoning=(
                    f"Confidence={strategy.confidence:.2f}, "
                    f"pros={len(strategy.pros)}, cons={len(strategy.cons)} "
                    f"-> raw={raw_score:.4f}, clamped={score:.4f}"
                ),
                selected=False,
            ))

        return results

    # -- selection -----------------------------------------------------------

    def _select_best(self, results: list[ExplorationResult]) -> ExplorationResult:
        """Pick the highest-scoring result and mark it as selected.

        In case of ties, the first encountered is chosen.
        """
        best = max(results, key=lambda r: r.score)
        best.selected = True
        return best

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current loop status and statistics."""
        db_stats = self._db.get_stats()
        return {
            "running": self._running,
            "subscriptions": len(self._subscription_ids),
            "explorations_run": self._explorations_run,
            "strategies_generated": self._strategies_generated,
            "strategies_selected": self._strategies_selected,
            "db": db_stats,
        }
