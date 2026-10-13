"""Unit and property-based tests for canonical cache key builder and normalization."""

from hypothesis import given
from hypothesis import strategies as st

from inferpath.cache.canonical_key import CanonicalKeyBuilder, normalize_prompt
from inferpath.models.domain import TaskParameters
from inferpath.models.enums import PrivacyClass, RouteTier


def test_identical_inputs_produce_identical_keys():
    """Identical parameters must produce identical SHA-256 cache keys."""
    params = TaskParameters(expected_format="json", language="en", audience="general")
    k1 = CanonicalKeyBuilder.build_cache_key(
        prompt="Explain photosynthesis",
        task_parameters=params,
        context_hash="ctx123",
        user_scope="tenant_a",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
        template_version="none",
        privacy_class=PrivacyClass.PUBLIC,
    )
    k2 = CanonicalKeyBuilder.build_cache_key(
        prompt="Explain photosynthesis",
        task_parameters=params,
        context_hash="ctx123",
        user_scope="tenant_a",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
        template_version="none",
        privacy_class=PrivacyClass.PUBLIC,
    )
    assert k1 == k2
    assert len(k1) == 64


def test_meaning_changing_prompts_produce_different_keys():
    """Length or number constraints in prompts must never be collapsed."""
    params = TaskParameters(expected_format="text")

    k_50 = CanonicalKeyBuilder.build_cache_key(
        prompt="summarize in 50 words",
        task_parameters=params,
        route=RouteTier.SMALL_MODEL_C,
    )
    k_500 = CanonicalKeyBuilder.build_cache_key(
        prompt="summarize in 500 words",
        task_parameters=params,
        route=RouteTier.SMALL_MODEL_C,
    )
    assert k_50 != k_500

    k_child = CanonicalKeyBuilder.build_cache_key(
        prompt="Explain photosynthesis to a child",
        task_parameters=params,
        route=RouteTier.SMALL_MODEL_C,
    )
    k_prof = CanonicalKeyBuilder.build_cache_key(
        prompt="Explain photosynthesis to a university professor",
        task_parameters=params,
        route=RouteTier.SMALL_MODEL_C,
    )
    assert k_child != k_prof


def test_parameter_changes_invalidate_key():
    """Variations in audience, format, context, scope, route, or version produce distinct keys."""
    base_prompt = "Translate this text"
    p1 = TaskParameters(expected_format="text", language="en", audience="general")
    p2 = TaskParameters(expected_format="json", language="en", audience="general")
    p3 = TaskParameters(expected_format="text", language="es", audience="general")
    p4 = TaskParameters(expected_format="text", language="en", audience="expert")

    k_base = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p1,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )

    # Format difference
    k_fmt = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p2,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )
    assert k_base != k_fmt

    # Language difference
    k_lang = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p3,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )
    assert k_base != k_lang

    # Audience difference
    k_aud = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p4,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )
    assert k_base != k_aud

    # Context hash difference
    k_ctx = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p1,
        context_hash="ctx2",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )
    assert k_base != k_ctx

    # Scope difference
    k_scope = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p1,
        context_hash="ctx1",
        user_scope="scope2",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v1",
    )
    assert k_base != k_scope

    # Route difference
    k_route = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p1,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.LARGE_MODEL_D,
        handler_version="v1",
    )
    assert k_base != k_route

    # Version difference
    k_ver = CanonicalKeyBuilder.build_cache_key(
        base_prompt,
        p1,
        context_hash="ctx1",
        user_scope="scope1",
        route=RouteTier.SMALL_MODEL_C,
        handler_version="v2",
    )
    assert k_base != k_ver


def test_conservative_whitespace_normalization():
    """Inconsequential whitespace differences collapse to identical keys."""
    params = TaskParameters(expected_format="text")
    p1 = "What is the capital   of France?  "
    p2 = "What is the capital of France?"
    p3 = "  What   is   the   capital   of   France?  "

    k1 = CanonicalKeyBuilder.build_cache_key(p1, params)
    k2 = CanonicalKeyBuilder.build_cache_key(p2, params)
    k3 = CanonicalKeyBuilder.build_cache_key(p3, params)

    assert k1 == k2
    assert k2 == k3
    assert normalize_prompt(p1) == "What is the capital of France?"


def test_unicode_normalization_nfc():
    """Decomposed and precomposed Unicode representations produce identical keys."""
    params = TaskParameters(expected_format="text")
    # "café" with precomposed é (\u00e9) vs decomposed e + combining acute (\u0065\u0301)
    p_precomposed = "caf\u00e9 menu"
    p_decomposed = "cafe\u0301 menu"

    assert p_precomposed != p_decomposed  # Raw strings differ

    k_pre = CanonicalKeyBuilder.build_cache_key(p_precomposed, params)
    k_dec = CanonicalKeyBuilder.build_cache_key(p_decomposed, params)

    assert k_pre == k_dec
    assert normalize_prompt(p_precomposed) == normalize_prompt(p_decomposed)


@given(st.text(min_size=1, max_size=100))
def test_hypothesis_property_key_determinism(prompt_text: str):
    """Hypothesis property test: key computation is strictly deterministic across runs."""
    params = TaskParameters(expected_format="text", language="en")
    k1 = CanonicalKeyBuilder.build_cache_key(prompt_text, params, route=RouteTier.SMALL_MODEL_C)
    k2 = CanonicalKeyBuilder.build_cache_key(prompt_text, params, route=RouteTier.SMALL_MODEL_C)
    assert k1 == k2
    assert len(k1) == 64
