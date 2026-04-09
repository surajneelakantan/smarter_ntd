"""
SMARTER.AI - Database Manager
NTD-grounded persistent memory layer for LLM interactions.

This is the ONLY file that touches SQL directly.
Every other component calls this interface.
"""

import os
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL interface for the NTD memory system.
    Uses a threaded connection pool for concurrent access.
    All configuration is read from environment variables.
    """

    def __init__(self, min_conn: int = 2, max_conn: int = 10):
        self.db_config = {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": int(os.environ.get("POSTGRES_PORT", 5432)),
            "dbname": os.environ.get("POSTGRES_DB", "smarter_ai"),
            "user": os.environ.get("POSTGRES_USER", "smarter_user"),
            "password": os.environ.get("POSTGRES_PASSWORD", "localdev123"),
        }
        try:
            self.pool = ThreadedConnectionPool(
                min_conn, max_conn, **self.db_config
            )
            log.info("Database connection pool created (%d-%d connections)", min_conn, max_conn)
        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to create connection pool: {e}")

    @contextmanager
    def _get_conn(self):
        """Context manager: get connection from pool, auto commit/rollback."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def _get_cursor(self):
        """Context manager: get a RealDictCursor for query results as dicts."""
        with self._get_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cur
            finally:
                cur.close()

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            log.info("Database connection pool closed")

    # =========================================================
    # USERS
    # =========================================================

    def create_user(self, metadata: Optional[Dict] = None) -> str:
        """Create a new user. Returns user_id."""
        with self._get_cursor() as cur:
            cur.execute(
                """INSERT INTO users (metadata)
                   VALUES (%s)
                   RETURNING user_id""",
                (json.dumps(metadata or {}),)
            )
            user_id = str(cur.fetchone()["user_id"])
            log.info("Created user: %s", user_id)
            return user_id

    def get_or_create_user(self, identifier: str) -> Dict:
        """
        Get existing user by identifier (stored in metadata->>'identifier')
        or create a new one. Returns full user row as dict.
        """
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT * FROM users
                   WHERE metadata->>'identifier' = %s""",
                (identifier,)
            )
            row = cur.fetchone()
            if row:
                # Update last_seen
                cur.execute(
                    """UPDATE users SET last_seen = NOW()
                       WHERE user_id = %s""",
                    (row["user_id"],)
                )
                log.info("Returning user found: %s", row["user_id"])
                return dict(row)

            # Create new user
            cur.execute(
                """INSERT INTO users (metadata)
                   VALUES (%s)
                   RETURNING *""",
                (json.dumps({"identifier": identifier}),)
            )
            new_user = dict(cur.fetchone())
            log.info("New user created: %s", new_user["user_id"])
            return new_user

    def update_last_seen(self, user_id: str) -> None:
        """Update user's last_seen timestamp."""
        with self._get_cursor() as cur:
            cur.execute(
                "UPDATE users SET last_seen = NOW() WHERE user_id = %s",
                (user_id,)
            )

    # =========================================================
    # SESSIONS
    # =========================================================

    def create_session(
        self,
        user_id: str,
        topic: Optional[str] = None,
        profession: Optional[str] = None,
        hours_budget: Optional[float] = None,
        learning_format: Optional[str] = None,
        professional_context: Optional[str] = None,
    ) -> str:
        """Create a new session. Returns session_id."""
        with self._get_cursor() as cur:
            cur.execute(
                """INSERT INTO sessions
                   (user_id, topic, profession, hours_budget,
                    learning_format, professional_context)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING session_id""",
                (user_id, topic, profession, hours_budget,
                 learning_format, professional_context)
            )
            session_id = str(cur.fetchone()["session_id"])
            log.info("Created session: %s for user: %s", session_id, user_id)
            return session_id

    def update_session_slots(self, session_id: str, topic: str,
                              profession: str, hours_budget: float,
                              learning_format: str) -> None:
        """Update session with slot-filling results."""
        with self._get_cursor() as cur:
            cur.execute(
                """UPDATE sessions
                   SET topic = %s, profession = %s,
                       hours_budget = %s, learning_format = %s,
                       last_active = NOW()
                   WHERE session_id = %s""",
                (topic, profession, hours_budget, learning_format, session_id)
            )

    def update_session_state(self, session_id: str, state: str) -> None:
        """Update session state (e.g., slot_filling -> plan_presented -> module_qa)."""
        with self._get_cursor() as cur:
            cur.execute(
                """UPDATE sessions
                   SET session_state = %s, last_active = NOW()
                   WHERE session_id = %s""",
                (state, session_id)
            )

    def store_recommended_modules(self, session_id: str, modules: List[Dict]) -> None:
        """Store the knapsack-selected modules as JSONB in the session."""
        with self._get_cursor() as cur:
            cur.execute(
                """UPDATE sessions
                   SET recommended_modules = %s, last_active = NOW()
                   WHERE session_id = %s""",
                (json.dumps(modules), session_id)
            )

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a single session by ID."""
        with self._get_cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get all sessions for a user, ordered by last_active descending."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT session_id, started_at, last_active, topic,
                          session_state, hours_budget, profession
                   FROM sessions
                   WHERE user_id = %s
                   ORDER BY last_active DESC
                   LIMIT %s""",
                (user_id, limit)
            )
            return [dict(row) for row in cur.fetchall()]

    def touch_session(self, session_id: str) -> None:
        """Update last_active timestamp (call on session end / browser close)."""
        with self._get_cursor() as cur:
            cur.execute(
                "UPDATE sessions SET last_active = NOW() WHERE session_id = %s",
                (session_id,)
            )

    # =========================================================
    # MESSAGES
    # =========================================================

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_type: str = "general",
        module_ref: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save a message. Returns message_id."""
        with self._get_cursor() as cur:
            cur.execute(
                """INSERT INTO messages
                   (session_id, role, content, message_type, module_ref, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING message_id""",
                (session_id, role, content, message_type,
                 json.dumps(module_ref) if module_ref else None,
                 json.dumps(metadata or {}))
            )
            return str(cur.fetchone()["message_id"])

    def get_session_messages(
        self, session_id: str, limit: int = 20
    ) -> List[Dict]:
        """Get recent messages for a session, oldest first."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT * FROM messages
                   WHERE session_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (session_id, limit)
            )
            rows = [dict(row) for row in cur.fetchall()]
            rows.reverse()  # Return oldest first
            return rows

    def get_module_messages(
        self, session_id: str, module_idx: int
    ) -> List[Dict]:
        """Get all messages referencing a specific module in a session."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT * FROM messages
                   WHERE session_id = %s
                     AND module_ref->>'module_idx' = %s
                   ORDER BY created_at ASC""",
                (session_id, str(module_idx))
            )
            return [dict(row) for row in cur.fetchall()]

    # =========================================================
    # MODULE INTERACTIONS (THE CORE NTD TABLE)
    # =========================================================

    def create_or_update_module_interaction(
        self, session_id: str, module_idx: int, module_pdf_name: str
    ) -> str:
        """
        Get existing interaction for this module in this session,
        or create a new one. Returns interaction_id.
        """
        with self._get_cursor() as cur:
            # Check if interaction already exists for this module in this session
            cur.execute(
                """SELECT interaction_id FROM module_interactions
                   WHERE session_id = %s AND module_idx = %s""",
                (session_id, module_idx)
            )
            row = cur.fetchone()
            if row:
                # Update last_asked_at
                cur.execute(
                    """UPDATE module_interactions
                       SET last_asked_at = NOW()
                       WHERE interaction_id = %s""",
                    (row["interaction_id"],)
                )
                return str(row["interaction_id"])

            # Create new interaction
            cur.execute(
                """INSERT INTO module_interactions
                   (session_id, module_idx, module_pdf_name)
                   VALUES (%s, %s, %s)
                   RETURNING interaction_id""",
                (session_id, module_idx, module_pdf_name)
            )
            return str(cur.fetchone()["interaction_id"])

    def increment_question_count(self, interaction_id: str) -> None:
        """Increment question_count for a module interaction (new questions only)."""
        with self._get_cursor() as cur:
            cur.execute(
                """UPDATE module_interactions
                   SET question_count = question_count + 1,
                       last_asked_at = NOW()
                   WHERE interaction_id = %s""",
                (interaction_id,)
            )

    def add_concept(self, interaction_id: str, concept: str) -> None:
        """
        Add a concept to concepts_asked JSONB array (distinct only).
        NTD: new concepts increment SS. Repeated concepts do not.
        """
        with self._get_cursor() as cur:
            # Only add if not already present
            cur.execute(
                """UPDATE module_interactions
                   SET concepts_asked = concepts_asked || %s::jsonb
                   WHERE interaction_id = %s
                     AND NOT concepts_asked @> %s::jsonb""",
                (json.dumps([concept]), interaction_id, json.dumps([concept]))
            )

    def add_professional_app(self, interaction_id: str, app_text: str) -> None:
        """
        Add a professional application to the interaction.
        NTD: associative links increase SS through additional encoding pathways.
        """
        with self._get_cursor() as cur:
            cur.execute(
                """UPDATE module_interactions
                   SET professional_apps = professional_apps || %s::jsonb
                   WHERE interaction_id = %s""",
                (json.dumps([app_text]), interaction_id)
            )

    def get_module_interaction(self, interaction_id: str) -> Optional[Dict]:
        """Get a single module interaction by ID."""
        with self._get_cursor() as cur:
            cur.execute(
                "SELECT * FROM module_interactions WHERE interaction_id = %s",
                (interaction_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_session_module_interactions(self, session_id: str) -> List[Dict]:
        """Get all module interactions for a session."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT * FROM module_interactions
                   WHERE session_id = %s
                   ORDER BY first_asked_at ASC""",
                (session_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_user_module_interactions(self, user_id: str) -> List[Dict]:
        """
        Get ALL module interactions across ALL sessions for a user.
        This is what the NTD engine needs to compute SS and RS.
        """
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT mi.*, s.user_id, s.profession
                   FROM module_interactions mi
                   JOIN sessions s ON mi.session_id = s.session_id
                   WHERE s.user_id = %s
                   ORDER BY mi.first_asked_at ASC""",
                (user_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    # =========================================================
    # PROFESSIONAL PROBLEMS
    # =========================================================

    def create_professional_problem(
        self,
        session_id: str,
        problem_text: str,
        related_modules: Optional[List] = None,
        related_concepts: Optional[List] = None,
    ) -> str:
        """Create a professional problem entry. Returns problem_id."""
        with self._get_cursor() as cur:
            cur.execute(
                """INSERT INTO professional_problems
                   (session_id, problem_text, related_modules, related_concepts)
                   VALUES (%s, %s, %s, %s)
                   RETURNING problem_id""",
                (session_id, problem_text,
                 json.dumps(related_modules or []),
                 json.dumps(related_concepts or []))
            )
            return str(cur.fetchone()["problem_id"])

    def get_session_problems(self, session_id: str) -> List[Dict]:
        """Get all professional problems for a session."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT * FROM professional_problems
                   WHERE session_id = %s
                   ORDER BY created_at ASC""",
                (session_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_user_problems(self, user_id: str) -> List[Dict]:
        """Get all professional problems across all sessions for a user."""
        with self._get_cursor() as cur:
            cur.execute(
                """SELECT pp.*, s.user_id
                   FROM professional_problems pp
                   JOIN sessions s ON pp.session_id = s.session_id
                   WHERE s.user_id = %s
                   ORDER BY pp.created_at ASC""",
                (user_id,)
            )
            return [dict(row) for row in cur.fetchall()]