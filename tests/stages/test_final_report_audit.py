from __future__ import annotations

from typing import Any

from review.report import final_report_audit


def test_final_report_audit_continues_after_rejected_revision(monkeypatch) -> None:
    calls: list[str] = []

    def fake_llm_json(*, prompt: str, system: str, cfg: Any, module: str) -> dict[str, Any]:
        del prompt, cfg, module
        if "strict paper-review fact auditor" in system:
            calls.append("audit")
            return {
                "audit_summary": "one issue",
                "issues": [
                    {
                        "problem_type": "format",
                        "severity": "medium",
                        "section": "2. Technical Positioning",
                        "review_excerpt": "bad",
                        "paper_evidence": "paper",
                        "suggested_fix": "fix it",
                        "should_fix": True,
                    }
                ],
            }
        calls.append("revision")
        return {"revision_summary": "bad revision", "revised_markdown": "## broken"}

    monkeypatch.setattr(final_report_audit, "llm_json", fake_llm_json)

    result = final_report_audit.audit_and_refine_final_report(
        final_markdown=(
            "## 1. Metadata\n"
            "- **Title:** Example\n"
            "- **Task:** Example task\n"
            "- **Code:** Not found in manuscript\n\n"
            "## 2. Technical Positioning\n"
            "| Research domain | Method | Capability |\n"
            "|---|---|---|\n"
            "| This Work | Method | √ |\n\n"
            "## 3. Claims\n"
            "| Claim | Evidence | Assessment | Status | Location |\n"
            "|---|---|---|---|---|\n"
            "| C | E | A | Pending | L |\n\n"
            "## 4. Summary\n"
            "Summary.\n\n"
            "Strengths:\n"
            "- S\n\n"
            "Weaknesses:\n"
            "- W\n\n"
            "## 5. Experiment\n"
            "### Main Result\n"
            "Location: Table 1.\n\n"
            "| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |\n"
            "|---|---|---|---|---|---|\n"
            "| T | D | M | 1 | 1 | 0 |\n\n"
            "### Ablation Result\n"
            "Location: Table 2.\n\n"
            "| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |\n"
            "|---|---|---|---|---|\n"
            "| Optimal setup | Full | 1 | 1 | 0 |\n"
        ),
        source_markdown="source paper text",
        max_iterations=3,
        max_source_chars=8000,
        max_review_chars=4000,
        model="gpt-5.5",
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=False,
    )

    assert result.iterations_run == 3
    assert calls == ["audit", "revision", "audit", "revision", "audit", "revision"]
    assert result.applied is False
    assert result.stop_reason == "revision_changed_fixed_format"
    assert all(not iteration.compatibility_ok for iteration in result.iterations)
