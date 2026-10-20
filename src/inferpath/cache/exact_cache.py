"""SQLite-backed exact cache engine with TTL enforcement, purge routine, and hit tracking."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from inferpath.models.domain import HandlerOutput, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.security.hashing import scrub_api_keys

CURRENT_SCHEMA_VERSION = 1

CACHE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS exact_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,
    tier TEXT NOT NULL,
    response_payload TEXT NOT NULL,
    requirements_snapshot TEXT NOT NULL,
    version TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exact_cache_expires ON exact_cache(expires_at);
"""


class ExactCache:
    """
    SQLite-backed thread-safe exact-response cache.
    Stores and serves validated responses with strict TTL checks, atomic hit count increments,
    and safe serialization.
    """

    def __init__(
        self,
        db_path: str | Path = "data/storage/inferpath.db",
        max_entries: int = 10_000,
    ):
        self.db_path = str(db_path)
        self.max_entries = max_entries
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _enforce_capacity(self, conn: sqlite3.Connection) -> None:
        """Evicts expired entries and enforces max_entries capacity via LRU/FIFO."""
        now = datetime.now(UTC).isoformat()
        conn.execute("DELETE FROM exact_cache WHERE expires_at <= ?", (now,))
        cursor = conn.execute("SELECT COUNT(*) FROM exact_cache")
        count = cursor.fetchone()[0]
        if count >= self.max_entries:
            excess = count - self.max_entries + 1
            conn.execute(
                f"""
                DELETE FROM exact_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM exact_cache
                    ORDER BY hit_count ASC, created_at ASC
                    LIMIT {excess}
                )
                """
            )

    async def _enforce_capacity_async(self, db: aiosqlite.Connection) -> None:
        """Asynchronously evicts expired entries and enforces max_entries capacity."""
        now = datetime.now(UTC).isoformat()
        await db.execute("DELETE FROM exact_cache WHERE expires_at <= ?", (now,))
        cursor = await db.execute("SELECT COUNT(*) FROM exact_cache")
        row = await cursor.fetchone()
        count = row[0] if row else 0
        if count >= self.max_entries:
            excess = count - self.max_entries + 1
            await db.execute(
                f"""
                DELETE FROM exact_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM exact_cache
                    ORDER BY hit_count ASC, created_at ASC
                    LIMIT {excess}
                )
                """
            )

    def init_schema(self) -> None:
        """Initializes exact cache table and indexes synchronously with busy timeout."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.executescript(CACHE_SCHEMA_SQL)
            conn.commit()

    async def init_schema_async(self) -> None:
        """Initializes exact cache table and indexes asynchronously with busy timeout."""
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA busy_timeout = 5000;")
            await db.executescript(CACHE_SCHEMA_SQL)
            await db.commit()

    def get_schema_version(self) -> int:
        """Returns the current database schema version, handling legacy tables gracefully."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA busy_timeout = 5000;")
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT MAX(version) FROM schema_version")
                row = cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
            except sqlite3.OperationalError:
                # Legacy table without schema_version table
                return 0

    async def get_schema_version_async(self) -> int:
        """Asynchronously returns the current database schema version."""
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 5000;")
            try:
                cursor = await db.execute("SELECT MAX(version) FROM schema_version")
                row = await cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
            except sqlite3.OperationalError:
                return 0

    def migrate_if_needed(self) -> int:
        """Applies schema migrations for legacy databases without schema_version."""
        ver = self.get_schema_version()
        if ver < CURRENT_SCHEMA_VERSION:
            self.init_schema()
            return CURRENT_SCHEMA_VERSION
        return ver

    def lookup(self, cache_key: str) -> HandlerOutput | None:
        """
        Synchronously retrieves a cached response by exact key.
        Expired entries are NEVER returned and are treated as cache misses.
        """
        sql = """
        SELECT response_payload, expires_at, hit_count
        FROM exact_cache
        WHERE cache_key = ?
        """
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA busy_timeout = 5000;")
            cursor = conn.cursor()
            cursor.execute(sql, (cache_key,))
            row = cursor.fetchone()
            if not row:
                return None

            payload_json, expires_at_str, hit_count = row

            # Validate expiration against current UTC time
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
            except Exception:
                return None

            now = datetime.now(UTC)
            if now >= expires_at:
                return None  # Expired entry is treated as a miss

            # Increment hit count atomically
            conn.execute(
                "UPDATE exact_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()

            # Safely deserialize response payload
            try:
                data = json.loads(payload_json)
                output = HandlerOutput(**data)
                output.cache_hit = True
                return output
            except Exception:
                # Corrupted or malformed payload detected - purge row to restore cache integrity
                try:
                    conn.execute("DELETE FROM exact_cache WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                except Exception:
                    pass
                return None

    get = lookup

    async def lookup_async(self, cache_key: str) -> HandlerOutput | None:
        """
        Asynchronously retrieves a cached response by exact key.
        Expired entries are NEVER returned.
        """
        sql = """
        SELECT response_payload, expires_at, hit_count
        FROM exact_cache
        WHERE cache_key = ?
        """
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 5000;")
            cursor = await db.execute(sql, (cache_key,))
            row = await cursor.fetchone()
            if not row:
                return None

            payload_json, expires_at_str, hit_count = row

            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
            except Exception:
                return None

            now = datetime.now(UTC)
            if now >= expires_at:
                return None

            await db.execute(
                "UPDATE exact_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,),
            )
            await db.commit()

            try:
                data = json.loads(payload_json)
                output = HandlerOutput(**data)
                output.cache_hit = True
                return output
            except Exception:
                try:
                    await db.execute("DELETE FROM exact_cache WHERE cache_key = ?", (cache_key,))
                    await db.commit()
                except Exception:
                    pass
                return None

    get_async = lookup_async

    def store(
        self,
        cache_key: str,
        request: UserRequest,
        output: HandlerOutput,
        route: RouteTier | str,
        ttl_seconds: int = 86400,
        validation_status: str = "passed",
        requirements_snapshot: str | None = None,
    ) -> bool:
        """
        Synchronously stores or updates a validated response payload in the cache.
        Returns True on successful insertion/update.
        """
        # Strict privacy enforcement: non-public or sensitive requests are NEVER stored in cache
        if getattr(request, "privacy_class", None) != PrivacyClass.PUBLIC:
            return False

        # Strict privacy enforcement: non-public or sensitive requests are NEVER stored in cache
        if getattr(request, "privacy_class", None) != PrivacyClass.PUBLIC:
            return False

        route_str = route.value if isinstance(route, RouteTier) else str(route)
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=max(1, ttl_seconds))

        payload_dict = output.model_dump(mode="json")
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        req_snapshot = requirements_snapshot or "{}"

        sql = """
        INSERT INTO exact_cache (
            cache_key, normalized_prompt, output_text, tier, response_payload,
            requirements_snapshot, version, validation_status, hit_count,
            input_tokens, output_tokens, latency_ms, expires_at
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, 0,
            ?, ?, ?, ?
        )
        ON CONFLICT(cache_key) DO UPDATE SET
            response_payload = excluded.response_payload,
            output_text = excluded.output_text,
            version = excluded.version,
            validation_status = excluded.validation_status,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            latency_ms = excluded.latency_ms,
            expires_at = excluded.expires_at;
        """
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA busy_timeout = 5000;")
            try:
                self._enforce_capacity(conn)
                prompt_preview = (
                    request.prompt[:500]
                    if getattr(request, "privacy_class", None) == PrivacyClass.PUBLIC
                    else "[REDACTED_CONFIDENTIAL]"
                )
                conn.execute(
                    sql,
                    (
                        cache_key,
                        prompt_preview,
                        output.output_text,
                        route_str,
                        payload_json,
                        req_snapshot,
                        output.model_version,
                        validation_status,
                        output.input_tokens.value,
                        output.output_tokens.value,
                        output.latency_ms.value,
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def set(
        self,
        cache_key: str,
        output: HandlerOutput,
        ttl_seconds: int = 86400,
        request: UserRequest | None = None,
        route: RouteTier | str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Alias for store matching key-value cache conventions."""
        req = request or UserRequest(request_id=f"cache_{cache_key[:8]}", prompt=output.output_text)
        rt = route or output.tier
        return self.store(
            cache_key=cache_key,
            request=req,
            output=output,
            route=rt,
            ttl_seconds=ttl_seconds,
        )

    async def store_async(
        self,
        cache_key: str,
        request: UserRequest,
        output: HandlerOutput,
        route: RouteTier | str,
        ttl_seconds: int = 86400,
        validation_status: str = "passed",
        requirements_snapshot: str | None = None,
    ) -> bool:
        """
        Asynchronously stores or updates a validated response payload in the cache.
        """
        route_str = route.value if isinstance(route, RouteTier) else str(route)
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=max(1, ttl_seconds))

        payload_dict = output.model_dump(mode="json")
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        req_snapshot = requirements_snapshot or "{}"

        sql = """
        INSERT INTO exact_cache (
            cache_key, normalized_prompt, output_text, tier, response_payload,
            requirements_snapshot, version, validation_status, hit_count,
            input_tokens, output_tokens, latency_ms, expires_at
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, 0,
            ?, ?, ?, ?
        )
        ON CONFLICT(cache_key) DO UPDATE SET
            response_payload = excluded.response_payload,
            output_text = excluded.output_text,
            version = excluded.version,
            validation_status = excluded.validation_status,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            latency_ms = excluded.latency_ms,
            expires_at = excluded.expires_at;
        """
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 5000;")
            try:
                await self._enforce_capacity_async(db)
                prompt_preview = (
                    request.prompt[:500]
                    if getattr(request, "privacy_class", None) == PrivacyClass.PUBLIC
                    else "[REDACTED_CONFIDENTIAL]"
                )
                await db.execute(
                    sql,
                    (
                        cache_key,
                        prompt_preview,
                        output.output_text,
                        route_str,
                        payload_json,
                        req_snapshot,
                        output.model_version,
                        validation_status,
                        output.input_tokens.value,
                        output.output_tokens.value,
                        output.latency_ms.value,
                        expires_at.isoformat(),
                    ),
                )
                await db.commit()
                return True
            except Exception:
                return False

    async def set_async(
        self,
        cache_key: str,
        output: HandlerOutput,
        ttl_seconds: int = 86400,
        request: UserRequest | None = None,
        route: RouteTier | str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Async alias for store_async matching key-value cache conventions."""
        req = request or UserRequest(request_id=f"cache_{cache_key[:8]}", prompt=output.output_text)
        rt = route or output.tier
        return await self.store_async(
            cache_key=cache_key,
            request=req,
            output=output,
            route=rt,
            ttl_seconds=ttl_seconds,
        )

    def purge_expired(self) -> int:
        """
        Synchronously purges all expired entries from the cache.
        Returns the number of deleted records.
        """
        now = datetime.now(UTC).isoformat()
        sql = "DELETE FROM exact_cache WHERE expires_at <= ?"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, (now,))
            conn.commit()
            return cursor.rowcount

    async def purge_expired_async(self) -> int:
        """
        Asynchronously purges all expired entries from the cache.
        Returns the number of deleted records.
        """
        now = datetime.now(UTC).isoformat()
        sql = "DELETE FROM exact_cache WHERE expires_at <= ?"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(sql, (now,))
            await db.commit()
            return cursor.rowcount

    def get_entry_metadata(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieves metadata of a cached entry for diagnostics without returning private prompt text."""
        sql = """
        SELECT cache_key, tier, version, validation_status, hit_count, created_at, expires_at
        FROM exact_cache
        WHERE cache_key = ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, (cache_key,))
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            return {k: scrub_api_keys(str(v)) for k, v in result.items()}
