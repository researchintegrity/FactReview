from __future__ import annotations

from typing import Any

import pytest

from review.report.claim_audit import (
    _cap_status,
    _extract_self_tag,
    _normalize_status,
    _verdict_to_label,
    audit_axis_self_selection,
    audit_review_markdown,
    inject_weaknesses,
)

# ---------------------------------------------------------------------------
# Status normalization + capping (small structural helpers)


def test_normalize_status_handles_pending_as_empty() -> None:
    # "Pending" is the placeholder the runner emits for claims whose status
    # is decided by the post-hoc audit. It must normalize to "" so that
    # _cap_status treats the LLM verdict as the new status rather than
    # capping toward the placeholder.
    assert _normalize_status("Pending") == ""
    assert _normalize_status("✓ Supported") == "supported"
    assert _normalize_status("⚠ Inconclusive") == "inconclusive"
    assert _normalize_status("✗ In conflict") == "in conflict"
    assert _normalize_status("partially supported") == "partially supported"


def test_cap_status_takes_more_conservative() -> None:
    # supported < partially supported < inconclusive < in conflict.
    assert _cap_status("supported", "inconclusive") == "inconclusive"
    assert _cap_status("inconclusive", "supported") == "inconclusive"
    assert _cap_status("partially supported", "in conflict") == "in conflict"
    # Pending acts as no-status: capping fills it with the cap.
    assert _cap_status("Pending", "supported") == "supported"
    assert _cap_status("", "inconclusive") == "inconclusive"


def test_verdict_to_label_normalizes_provider_variants() -> None:
    assert _verdict_to_label("supported") == "supported"
    assert _verdict_to_label("partially_supported") == "partially supported"
    assert _verdict_to_label("partially-supported") == "partially supported"
    assert _verdict_to_label("partial") == "partially supported"
    assert _verdict_to_label("inconclusive") == "inconclusive"
    assert _verdict_to_label("unclear") == "inconclusive"
    assert _verdict_to_label("in_conflict") == "in conflict"
    assert _verdict_to_label("in conflict") == "in conflict"
    assert _verdict_to_label("conflict") == "in conflict"
    assert _verdict_to_label("garbage") == ""


# ---------------------------------------------------------------------------
# Self-tag parser


def test_extract_self_tag_strips_bracketed_verdict() -> None:
    text = (
        "The evidence supports the qualitative direction. "
        "[verdict: partially_supported; reason: gap is below 2 sigma]"
    )
    verdict, reason, cleaned = _extract_self_tag(text)
    assert verdict == "partially supported"
    assert reason == "gap is below 2 sigma"
    assert "[verdict:" not in cleaned
    assert cleaned.strip().endswith("direction.")


def test_extract_self_tag_returns_empty_when_absent() -> None:
    verdict, reason, cleaned = _extract_self_tag("plain assessment text")
    assert verdict == "" and reason == ""
    assert cleaned == "plain assessment text"


# ---------------------------------------------------------------------------
# Axis self-selection (deterministic, structural)


def test_axis_self_selection_flagged_when_majority_exclusive_wins() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "Caption.\n\n"
        "| Research domain | Method | Context comp. | Persistent notes | Modular SDK | Repo issues |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| SWE agents | SWE-Agent | × | × | × | √ |\n"
        "| SWE agents | OpenHands | × | × | × | √ |\n"
        "| Test-time | Live-SWE-Agent | × | × | × | √ |\n"
        "| This Work | CCA | √ | √ | √ | √ |\n"
        "## 3. Claims\n"
    )
    ratio, bullet = audit_axis_self_selection(md)
    # 3 of 4 niche cols are exclusive wins => ratio = 0.75.
    assert ratio is not None and ratio >= 0.6
    assert bullet is not None
    assert "favor the proposed system" in bullet


def test_axis_self_selection_quiet_when_ratio_low() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "| Research domain | Method | A | B |\n"
        "| --- | --- | --- | --- |\n"
        "| Domain | Other | √ | √ |\n"
        "| This Work | Ours | √ | × |\n"
        "## 3. Claims\n"
    )
    _, bullet = audit_axis_self_selection(md)
    assert bullet is None


# ---------------------------------------------------------------------------
# Weakness injection


def test_inject_weaknesses_appends_audit_bullets() -> None:
    md = (
        "## 4. Summary\n"
        "Some summary text.\n\n"
        "**Strengths:**\n"
        "- A strength\n\n"
        "**Weaknesses:**\n"
        "- Existing weakness\n\n"
        "## 5. Experiment\n"
    )
    out = inject_weaknesses(md, ["First audit weakness", "Second audit weakness"])
    assert "- Existing weakness" in out
    assert "- [audit] First audit weakness" in out
    assert "- [audit] Second audit weakness" in out
    pos_existing = out.find("Existing weakness")
    pos_audit = out.find("[audit] First")
    assert pos_existing < pos_audit
    assert out.count("**Weaknesses:**") == 1


def test_inject_weaknesses_handles_inline_weakness_label() -> None:
    md = (
        "## 4. Summary\n"
        "Summary.\n\n"
        "**Strengths:** - A strength.\n"
        "- Another strength.\n\n"
        "**Weaknesses:** - First weakness.\n"
        "- Second weakness.\n\n"
        "## 5. Experiment\n"
    )
    out = inject_weaknesses(md, ["audit one"])
    assert out.count("**Weaknesses:**") == 1
    assert "- [audit] audit one" in out
    assert out.index("Second weakness") < out.index("[audit] audit one")
    assert "## 5. Experiment" in out


# ---------------------------------------------------------------------------
# Batched LLM audit (audit_review_markdown)


def _make_llm_call(
    *,
    verdicts: list[dict[str, Any]],
    missing: list[str] | None = None,
    captured: list[str] | None = None,
) -> Any:
    """Build a fake llm_call returning a fixed response and (optionally)
    capturing the prompt sent to it."""
    payload = {
        "verdicts": verdicts,
        "ablation_missing_components": missing or [],
    }

    def _call(prompt: str) -> dict[str, Any]:
        if captured is not None:
            captured.append(prompt)
        return payload

    return _call


def test_audit_review_markdown_caps_supported_to_inconclusive() -> None:
    md = (
        "## 3. Claims\n"
        "(legend)\n\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| CCA achieves leading reported performance on SWE-Bench-Pro. | "
        "Table 1 reports 59.0 +/- 1.9 vs. 57.7. | "
        "ok | "
        '<span style="color: green;">✓ Supported</span> | Table 1 |\n'
        "## 4. Summary\n"
        "Summary.\n\n"
        "**Strengths:**\n- s\n\n"
        "**Weaknesses:**\n- w\n\n"
        "## 5. Experiment\n"
    )
    captured: list[str] = []
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "inconclusive", "reason": "gap within 1 sigma"}],
        captured=captured,
    )
    new_md, outcome = audit_review_markdown(md, llm_call=llm_call)

    assert len(outcome.claim_results) == 1
    result = outcome.claim_results[0]
    assert result.original_status == "supported"
    assert result.final_status == "inconclusive"
    assert result.llm_verdict == "inconclusive"
    assert result.llm_reason == "gap within 1 sigma"

    # Markdown reflects the cap.
    assert "⚠ Inconclusive" in new_md
    claims_chunk = new_md.split("## 4.")[0]
    assert "✓ Supported" not in claims_chunk

    # Audit weakness bullet injected.
    assert any("inconclusive" in b.lower() for b in outcome.extra_weaknesses)
    assert "[audit] Status downgraded to Inconclusive" in new_md

    # The prompt carried both claim id and ablation marker.
    assert len(captured) == 1
    assert "claim id=0" in captured[0]
    assert "Ablation section to compare against" in captured[0]


def test_audit_review_markdown_promotes_pending_to_llm_verdict() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Method M describes a new attention block. | "
        "Section 3.1 introduces M with ablation in Table 2. | "
        "ok | Pending | Section 3.1 |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
    )
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "supported", "reason": "method-design anchored in section + ablation"}],
    )
    new_md, outcome = audit_review_markdown(md, llm_call=llm_call)

    result = outcome.claim_results[0]
    # Pending normalizes to "" in the original status field.
    assert result.original_status == ""
    assert result.final_status == "supported"
    assert "✓ Supported" in new_md
    # No status change vs. original means no audit downgrade bullet for this row.
    assert not any(
        "Status downgraded" in b for b in outcome.extra_weaknesses
    )


def test_audit_review_markdown_does_not_upgrade_supported_when_llm_says_supported() -> None:
    # When the LLM agrees with the existing Supported status, capping is a
    # no-op and the markdown is unchanged for the status cell.
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Trivial. | Evidence. | ok | "
        '<span style="color: green;">✓ Supported</span> | Loc |\n'
        "## 4. Summary\n"
    )
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "supported", "reason": "fine"}],
    )
    new_md, outcome = audit_review_markdown(md, llm_call=llm_call)
    assert outcome.claim_results[0].final_status == "supported"
    assert "✓ Supported" in new_md


def test_audit_review_markdown_does_not_upgrade_inconclusive_when_llm_says_supported() -> None:
    # Capping is one-way (only toward more conservative). If the LLM is more
    # positive than the existing status, the status stays put.
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Trivial. | Evidence. | ok | "
        '<span style="color: #E6B800;">⚠ Inconclusive</span> | Loc |\n'
        "## 4. Summary\n"
    )
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "supported", "reason": "looks fine"}],
    )
    new_md, _ = audit_review_markdown(md, llm_call=llm_call)
    assert "⚠ Inconclusive" in new_md
    claims_chunk = new_md.split("## 4.")[0]
    assert "✓ Supported" not in claims_chunk


def test_audit_review_markdown_uses_agent_self_tag_as_extra_cap() -> None:
    # Agent self-tag is a free per-claim verdict the agent appends to the
    # Assessment cell. It must be reconciled with the LLM verdict by taking
    # the more conservative of the two (and stripped from the visible cell).
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Trivial. | Evidence. | "
        "ok [verdict: in_conflict; reason: paper value below comparator] | "
        '<span style="color: green;">✓ Supported</span> | Loc |\n'
        "## 4. Summary\n"
    )
    # LLM is more lenient than the agent self-tag - final cap should still be
    # the more conservative "in conflict".
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "partially_supported", "reason": "weak"}],
    )
    new_md, outcome = audit_review_markdown(md, llm_call=llm_call)
    result = outcome.claim_results[0]
    assert result.agent_self_verdict == "in conflict"
    assert result.llm_verdict == "partially supported"
    assert result.final_status == "in conflict"
    assert "✗ In conflict" in new_md
    # The bracketed self-tag is stripped from the assessment cell.
    assert "[verdict:" not in new_md


def test_audit_review_markdown_injects_missing_components_bullet() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Method M with A, B, and C. | Sec 2. | ok | Pending | Sec 2 |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
        "## 5. Experiment\n"
        "### Ablation Result\n"
        "| Dim | Cfg | Full | Paper | Δ |\n"
        "|---|---|---|---|---|\n"
        "| A | no | 1.0 | 0.5 | -0.5 |\n"
    )
    llm_call = _make_llm_call(
        verdicts=[{"id": 0, "verdict": "partially_supported", "reason": "B and C not ablated"}],
        missing=["B", "C"],
    )
    new_md, outcome = audit_review_markdown(md, llm_call=llm_call)
    assert outcome.ablation_components_missing == ["B", "C"]
    # Weakness bullet enumerates the missing components.
    assert any(
        "B" in b and "C" in b and "ablation" in b.lower()
        for b in outcome.extra_weaknesses
    )
    assert "[audit]" in new_md.split("## 5.")[0]


def test_audit_review_markdown_propagates_llm_failure() -> None:
    # The LLM call is mandatory: when the wrapper raises, the audit must not
    # swallow the exception (no graceful degradation).
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| C. | E. | ok | Pending | L |\n"
        "## 4. Summary\n"
    )

    def boom(prompt: str) -> dict[str, Any]:
        raise RuntimeError("LLM unreachable")

    with pytest.raises(RuntimeError, match="LLM unreachable"):
        audit_review_markdown(md, llm_call=boom)


def test_audit_review_markdown_skips_llm_when_no_claims_table() -> None:
    # No Claims section at all: we must not invoke the LLM and must still
    # run the structural axis audit.
    md = (
        "## 2. Technical Positioning\n"
        "| Research domain | Method | A | B | C |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Other | X | × | × | × |\n"
        "| Other | Y | × | × | × |\n"
        "| Other | Z | × | × | × |\n"
        "| This Work | Ours | √ | √ | √ |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
    )

    def must_not_be_called(prompt: str) -> dict[str, Any]:
        raise AssertionError("LLM should not be called when there are no claims")

    new_md, outcome = audit_review_markdown(md, llm_call=must_not_be_called)
    assert outcome.claim_results == []
    assert outcome.axis_self_selection_ratio is not None
    # Axis bullet still gets injected.
    assert any("favor the proposed system" in b for b in outcome.extra_weaknesses)
    assert "[audit]" in new_md


def test_audit_review_markdown_handles_garbage_llm_response() -> None:
    # Defensive: when the LLM returns an unexpected dict shape, the audit
    # leaves statuses untouched rather than crash.
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| C. | E. | ok | "
        '<span style="color: green;">✓ Supported</span> | L |\n'
        "## 4. Summary\n"
    )

    def garbage(prompt: str) -> dict[str, Any]:
        return {"unexpected": "shape"}

    new_md, outcome = audit_review_markdown(md, llm_call=garbage)
    # No verdicts parsed; status stays Supported.
    assert "✓ Supported" in new_md
    result = outcome.claim_results[0]
    assert result.llm_verdict == ""
    assert result.final_status == "supported"
