"""Unit tests for ExactCache SQLite storage, TTL expiration, purge, and safety."""

import sqlite3
import time

import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.models.domain import HandlerOutput, UserRequest
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


def _create_sample_output(text: str = "Cached answer text") -> HandlerOutput:
    return HandlerOutput(
        tier=RouteTier.SMALL_MODEL_C,
        output_text=text,
        validated=True,
        needs_clarification=False,
        input_tokens=ProvenanceValue(value=15, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=30, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=42.0, provenance=ProvenanceLabel.MEASURED, unit="ms"),
        model_version="gemini-2.5-flash-v1",
    )


def test_cache_store_and_lookup_hit(exact_cache: ExactCache):
    """Verifies that storing a validated response yields an exact cache hit with incremented hit count."""
    req = UserRequest(request_id="req-c-1", prompt="What is photosynthesis?")
    output = _create_sample_output("Photosynthesis converts light into chemical energy.")
    key = "test_key_001"

    # Store
    stored = exact_cache.store(
        cache_key=key,
        request=req,
        output=output,
        route=RouteTier.SMALL_MODEL_C,
        ttl_seconds=3600,
    )
    assert stored is True

    # Lookup hit
    hit = exact_cache.lookup(key)
    assert hit is not None
    assert hit.output_text == "Photosynthesis converts light into chemical energy."
    assert hit.cache_hit is True
    assert hit.input_tokens.value == 15
    assert hit.output_tokens.value == 30

    # Verify hit count incremented
    meta = exact_cache.get_entry_metadata(key)
    assert meta is not None
    assert int(meta["hit_count"]) == 1

    # Second hit increments again
    hit2 = exact_cache.lookup(key)
    assert hit2 is not None
    meta2 = exact_cache.get_entry_metadata(key)
    assert int(meta2["hit_count"]) == 2


def test_expired_entry_is_never_returned(exact_cache: ExactCache):
    """An expired entry must strictly return None (cache miss)."""
    req = UserRequest(request_id="req-exp-1", prompt="Short lived data")
    output = _create_sample_output("This data expires quickly.")
    key = "short_ttl_key"

    # Store with 2 second TTL
    exact_cache.store(
        cache_key=key,
        request=req,
        output=output,
        route=RouteTier.SMALL_MODEL_C,
        ttl_seconds=2,
    )

    # Immediately lookup: should hit
    hit_immediate = exact_cache.lookup(key)
    assert hit_immediate is not None

    # Wait 2.1s for expiration
    time.sleep(2.1)

    # Lookup after expiration must strictly be None
    hit_expired = exact_cache.lookup(key)
    assert hit_expired is None


def test_purge_expired_entries(exact_cache: ExactCache):
    """Purge routine removes expired entries from SQLite and returns deleted row count."""
    req = UserRequest(request_id="req-p-1", prompt="Purge test prompt")
    output = _create_sample_output()

    key_expired = "key_to_purge"
    key_valid = "key_to_keep"

    # Store one with 1s TTL and one with 1hr TTL
    exact_cache.store(key_expired, req, output, RouteTier.SMALL_MODEL_C, ttl_seconds=1)
    exact_cache.store(key_valid, req, output, RouteTier.SMALL_MODEL_C, ttl_seconds=3600)

    time.sleep(1.1)

    # Purge
    deleted_count = exact_cache.purge_expired()
    assert deleted_count >= 1

    # Expired key should no longer exist in DB metadata
    assert exact_cache.get_entry_metadata(key_expired) is None
    # Valid key remains
    assert exact_cache.get_entry_metadata(key_valid) is not None


def test_malformed_cache_payload_fails_safely(exact_cache: ExactCache):
    """Corrupted response_payload in SQLite returns None safely without raising an unhandled exception."""
    req = UserRequest(request_id="req-corrupt-1", prompt="Corrupt test")
    output = _create_sample_output()
    key = "corrupt_key"

    exact_cache.store(key, req, output, RouteTier.SMALL_MODEL_C, ttl_seconds=3600)

    # Directly corrupt response_payload in SQLite
    with sqlite3.connect(exact_cache.db_path) as conn:
        conn.execute(
            "UPDATE exact_cache SET response_payload = 'INVALID_NON_JSON{' WHERE cache_key = ?",
            (key,),
        )
        conn.commit()

    # Lookup should safely fail and return None
    result = exact_cache.lookup(key)
    assert result is None


def test_repeated_writes_update_cleanly(exact_cache: ExactCache):
    """Repeated writes for the same cache key update the entry cleanly without unique constraint failure."""
    req = UserRequest(request_id="req-up-1", prompt="Updatable prompt")
    out1 = _create_sample_output("Version 1")
    out2 = _create_sample_output("Version 2")
    key = "update_key"

    s1 = exact_cache.store(key, req, out1, RouteTier.SMALL_MODEL_C, ttl_seconds=3600)
    assert s1 is True
    hit1 = exact_cache.lookup(key)
    assert hit1.output_text == "Version 1"

    s2 = exact_cache.store(key, req, out2, RouteTier.SMALL_MODEL_C, ttl_seconds=3600)
    assert s2 is True
    hit2 = exact_cache.lookup(key)
    assert hit2.output_text == "Version 2"


@pytest.mark.asyncio
async def test_async_cache_operations(exact_cache: ExactCache):
    """Verifies async store, lookup, and purge functions."""
    req = UserRequest(request_id="req-async-1", prompt="Async test prompt")
    output = _create_sample_output("Async output text")
    key = "async_test_key"

    stored = await exact_cache.store_async(
        key, req, output, RouteTier.SMALL_MODEL_C, ttl_seconds=3600
    )
    assert stored is True

    hit = await exact_cache.lookup_async(key)
    assert hit is not None
    assert hit.output_text == "Async output text"
    assert hit.cache_hit is True
