from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_meowagenet_idea079_spatial_patch_adapter.py"
SPEC = importlib.util.spec_from_file_location("idea079_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
idea079 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea079
SPEC.loader.exec_module(idea079)


def protocol() -> dict:
    return idea079.read_json(idea079.PROTOCOL_PATH)


def test_resolved_protocol_locks_method_budget_and_new_seed_bank() -> None:
    value = protocol()
    idea079.verify_protocol(value)
    assert value["status"] == idea079.LOCKED_STATUS
    assert tuple(value["model"]["pipelines"]) == idea079.PIPELINES
    assert tuple(value["model"]["base_seeds"]) == idea079.BASE_SEEDS
    assert value["model"]["total_fits"] == 108
    assert value["model"]["primary_A0_S1_fits"] == 72
    assert value["model"]["outer_test_predictions"] is False
    assert value["relationship"]["idea077_frozen"] is True
    assert value["relationship"]["idea078_shared_cache_only"] is True


def test_seed_digest_and_all_full_seeds_exclude_idea078() -> None:
    value = protocol()["model"]
    digest = hashlib.sha256(value["seed_derivation_text"].encode("utf-8")).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10_000
        for offset in range(0, len(digest), 4)
    ]
    assert candidates[:3] == list(idea079.BASE_SEEDS)
    derived = idea079._all_full_seeds(idea079.BASE_SEEDS)
    idea078 = idea079._all_full_seeds([9763, 3230, 9726])
    excluded = idea079._all_full_seeds(
        value["excluded_meow_base_seeds_IDEA065_through_IDEA078"]
    )
    assert len(derived) == 36
    assert derived.isdisjoint(idea078)
    assert derived.isdisjoint(excluded)


def test_source_provenance_explains_843_segments_for_792_calls() -> None:
    audit = idea079.audit_source_fbank(protocol())
    assert audit["calls"] == 792
    assert audit["cats"] == 111
    assert audit["segments"] == 843
    assert audit["segment_count_range"] == [1, 6]
    assert audit["multi_segment_calls"] == 42
    assert audit["segment_count_sum"] == 843
    assert audit["source_path_rows"] == 792
    assert audit["window_seconds"] == 1.28
    assert audit["window_hop_seconds"] == 0.64
    assert audit["fbank_sha256"] == "007d07f7c236ba44ae76a1e867cee5cc49b774ddde2f54064c68190cd01d60c7"
    assert audit["audio_manifest_sha256"] == "68e5131dc5d3cd611ecdda30e5176a6dcc90c0ea2500a6d8a4d9b066ce11a72f"
    assert audit["audio_checksums_sha256"] == "7c1b51ce1a18b1253d3099e9ce3ea034385cd1851ed121a1a5b82a50a20e6a12"


def test_adapter_parameter_match_and_paired_initial_draws() -> None:
    seed = idea079.BASE_SEEDS[0]
    torch.manual_seed(seed)
    pointwise = idea079.PatchTokenAdapter("pointwise")
    torch.manual_seed(seed)
    spatial = idea079.PatchTokenAdapter("spatial")
    assert idea079.trainable_parameter_count(pointwise) == idea079.ADAPTER_PARAMETERS
    assert idea079.trainable_parameter_count(spatial) == idea079.ADAPTER_PARAMETERS
    assert sum(p.numel() for p in pointwise.mixer.parameters()) == 90
    assert sum(p.numel() for p in spatial.mixer.parameters()) == 90
    assert pointwise.mixer.weight.shape == (9, 9, 1, 1)
    assert spatial.mixer.weight.shape == (9, 1, 3, 3)
    torch.testing.assert_close(pointwise.down.weight, spatial.down.weight, rtol=0, atol=0)
    torch.testing.assert_close(pointwise.down.bias, spatial.down.bias, rtol=0, atol=0)
    torch.testing.assert_close(
        pointwise.mixer.weight.reshape(-1),
        spatial.mixer.weight.reshape(-1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(pointwise.mixer.bias, spatial.mixer.bias, rtol=0, atol=0)
    assert torch.count_nonzero(pointwise.up.weight) == 0
    assert torch.count_nonzero(spatial.up.weight) == 0
    assert torch.count_nonzero(pointwise.up.bias) == 0
    assert torch.count_nonzero(spatial.up.bias) == 0


def test_patch_adapter_is_exact_identity_at_init_and_special_tokens_always_bypass() -> None:
    generator = torch.Generator().manual_seed(790)
    tokens = torch.randn(
        (3, idea079.TOTAL_TOKENS, idea079.HIDDEN_SIZE), generator=generator
    )
    for mode in ("pointwise", "spatial"):
        adapter = idea079.PatchTokenAdapter(mode)
        initialized = adapter(tokens)
        torch.testing.assert_close(initialized, tokens, rtol=0, atol=0)
        with torch.no_grad():
            adapter.up.weight.fill_(1.0e-3)
            adapter.up.bias.fill_(1.0e-3)
        adapted = adapter(tokens)
        torch.testing.assert_close(
            adapted[:, : idea079.SPECIAL_TOKENS],
            tokens[:, : idea079.SPECIAL_TOKENS],
            rtol=0,
            atol=0,
        )
        assert torch.max(
            torch.abs(
                adapted[:, idea079.SPECIAL_TOKENS :]
                - tokens[:, idea079.SPECIAL_TOKENS :]
            )
        ) > 0


def test_A0_C1_S1_heads_logits_and_losses_are_exactly_paired_without_cache() -> None:
    audit = idea079.paired_initialization_audit(
        protocol(), idea079.BASE_SEEDS[0], actual_tail=False
    )
    assert audit["trainable_parameters"] == idea079.EXPECTED_TRAINABLE_PARAMETERS
    assert all(audit["shared_head_state_equal"].values())
    assert all(
        value == 0.0
        for value in audit["max_initial_logit_difference_from_A0"].values()
    )
    assert all(
        value == 0.0
        for value in audit["initial_loss_difference_from_A0"].values()
    )
    assert all(
        value == 0.0
        for value in audit["max_special_token_difference_at_injection"].values()
    )
    assert all(audit["control_initialization_equal"].values())


def test_zero_up_gradient_schedule_is_reachable_for_both_controls() -> None:
    audit = idea079.gradient_reachability_audit(protocol(), idea079.BASE_SEEDS[0])
    assert set(audit) == {"C1_pointwise_patch", "S1_spatial_patch"}
    for row in audit.values():
        assert row["up_weight_max_gradient_at_zero"] > 0.0
        assert row["down_weight_max_gradient_at_zero"] == 0.0
        assert row["mixer_weight_max_gradient_at_zero"] == 0.0
        assert row["down_weight_max_gradient_after_up_probe"] > 0.0
        assert row["mixer_weight_max_gradient_after_up_probe"] > 0.0


def test_spatial_mixer_has_local_3x3_support_while_pointwise_does_not() -> None:
    pointwise = idea079.PatchTokenAdapter("pointwise").mixer
    spatial = idea079.PatchTokenAdapter("spatial").mixer
    with torch.no_grad():
        pointwise.weight.zero_()
        pointwise.bias.zero_()
        pointwise.weight[0, 0, 0, 0] = 1.0
        spatial.weight.zero_()
        spatial.bias.zero_()
        spatial.weight[0, 0, :, :] = 1.0
    impulse = torch.zeros((1, 9, 12, 12))
    impulse[0, 0, 5, 5] = 1.0
    point_support = torch.nonzero(pointwise(impulse)[0, 0], as_tuple=False)
    spatial_support = torch.nonzero(spatial(impulse)[0, 0], as_tuple=False)
    assert point_support.tolist() == [[5, 5]]
    assert sorted(spatial_support.tolist()) == sorted(
        [[frequency, time] for frequency in (4, 5, 6) for time in (4, 5, 6)]
    )


def test_shared_cache_manifest_is_resolved_without_changing_method_fields() -> None:
    value = protocol()
    assert value["shared_cache"]["cache_manifest_sha256"] == (
        "af14eb7b8a2f9c362f37484d06af4381fe89f1a497a89c62f1142ab49baf8e94"
    )
    resolved = idea079._resolved_cache(value)
    assert resolved["tokens_sha256"] == (
        "9fb7b32973560d31ad1906c07349e745054683d5a6a3486e3e33182fc6947ca9"
    )
    assert resolved["index_sha256"] == (
        "aa46820aa15a0812ef4437c80c29f28fd515f0228a9d0fd10bac266b734b9bb2"
    )
    assert resolved["tokens_format"] == "npy_float32_mmap"
    assert resolved["reconstruction_vs_locked_A0"] == {
        "mean_absolute_difference": 0.0000004022432733563619,
        "maximum_absolute_difference": 0.000008463859558105469,
    }


def test_idea065_nonduplication_boundary_is_explicitly_locked() -> None:
    value = protocol()
    nonduplicate = value["relationship"]["nonduplication"]
    assert "block 11" in nonduplicate["placement"]
    assert "144 patch tokens" in nonduplicate["token_scope"]
    assert "3x3 depthwise" in nonduplicate["mechanism"]
    assert value["adapter"]["special_tokens_bypassed"] == 2
    assert value["adapter"]["placement_after_block_one_based"] == 11
    assert value["adapter"]["placement_before_block_one_based"] == 12


def test_cache_budget_and_compute_are_locked_without_allocating_cache() -> None:
    value = protocol()
    cache = value["shared_cache"]
    assert cache["required_tensor_shape"] == [843, 146, 768]
    assert cache["required_tensor_dtype"] == "float32"
    assert cache["raw_token_bytes"] == 843 * 146 * 768 * 4
    assert value["adapter"]["mac_per_segment_each"] == (
        144 * 768 * 9 + 144 * 9 * 9 + 144 * 9 * 768
    )
