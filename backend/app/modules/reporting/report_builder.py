"""
Assembles the final Responsible AI Report as a PDF, reading only
already-persisted Run/ModelVersion records -- performs no new
computation, per SDD Sec.16. Sections follow the SDD's fixed order.

Handles an arbitrary number of mitigated versions per run (V1 plus
one or more V2s, one per mitigation method that ran) -- earlier this
module assumed exactly one V2 and always picked whichever mitigated
version was created first (in practice, always Reweighing, since it
ran first in every batch). That silently hid every other mitigation
method from the report. Every mitigated version now gets its own
subsection, and the closing comparison/recommendation sections treat
them as siblings rather than singling one out as "the" debiased model.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from app.core.exceptions import DatasetValidationError
from app.db import repository as repo
from app.modules.storage.report_storage import get_report_path
from app.modules.narrative.narrative_generator import generate_fairness_narrative
from app.schemas.fairness import FairnessFinding

MITIGATION_DESCRIPTIONS = {
    "Reweighing": (
        "Reweighing is a pre-processing bias mitigation technique that "
        "assigns higher training weight to underrepresented "
        "(group, outcome) combinations, so the model learns from a "
        "fairness-adjusted view of the training data without changing "
        "the raw records themselves. It trains a new model."
    ),
    "Disparate Impact Remover": (
        "Disparate Impact Remover is a pre-processing technique that edits "
        "the non-protected feature distributions so they become "
        "statistically similar across groups, reducing the extent to "
        "which the protected attribute can be reconstructed from other "
        "features, before a new model is trained on the edited data."
    ),
    "Calibrated Equalized Odds Postprocessing": (
        "Calibrated Equalized Odds Postprocessing adjusts the already-"
        "trained model's output probabilities per group, after training, "
        "to equalize false positive and false negative rates between "
        "groups while preserving calibration. It does not train a new "
        "model -- the underlying estimator is unchanged; only its output "
        "is corrected at inference time."
    ),
    "Reject Option Classification": (
        "Reject Option Classification adjusts predictions that fall "
        "within a band of low confidence around the decision boundary, "
        "favoring the unprivileged group in that band, to reduce "
        "disparate treatment near the threshold. It does not train a new "
        "model -- the underlying estimator is unchanged; only its output "
        "is corrected at inference time."
    ),
}

styles = getSampleStyleSheet()
H1 = styles["Heading1"]
H2 = styles["Heading2"]
H3 = styles["Heading3"]
BODY = styles["Normal"]
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, textColor=colors.grey)


def _fmt(value) -> str:
    """Formats a metric that may be None (undefined -- e.g. division by
    zero in disparate impact when a group's selection rate is 0)."""
    if value is None:
        return "undefined"
    return f"{value:.4f}"


def _metrics_table(headers: list[str], rows: list[list[str]]) -> Table:
    data = [headers] + rows
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _spd_improvement(v1, v2) -> float:
    """Positive means v2 reduced |statistical_parity_difference| vs v1."""
    f1, f2 = v1.fairness_metrics, v2.fairness_metrics
    return abs(f1.get("statistical_parity_difference", 0)) - abs(
        f2.get("statistical_parity_difference", 0)
    )


def _add_mitigated_version_section(story, v1, v2, section_no: int):
    finding1 = FairnessFinding.model_validate(v1.fairness_finding)
    finding2 = FairnessFinding.model_validate(v2.fairness_finding)

    method = v2.mitigation_method or "Unknown method"
    category_label = {"pre": "pre-processing", "in": "in-processing", "post": "post-processing"}.get(
        v2.mitigation_category, v2.mitigation_category or "-"
    )

    story.append(Paragraph(f"{section_no}. Mitigation: {method}", H2))
    story.append(Paragraph(f"<b>Category:</b> {category_label}", BODY))
    description = MITIGATION_DESCRIPTIONS.get(method)
    if description:
        story.append(Paragraph(description, BODY))
    if v2.runtime_seconds is not None:
        story.append(Paragraph(f"<b>Runtime:</b> {v2.runtime_seconds:.2f}s", BODY))
    story.append(Spacer(1, 8))

    p1, p2 = v1.performance_metrics, v2.performance_metrics
    story.append(Paragraph("Performance (original vs this version)", H3))
    story.append(_metrics_table(
        ["Metric", "Original", method],
        [
            ["Accuracy", f"{p1.get('accuracy', 0):.4f}", f"{p2.get('accuracy', 0):.4f}"],
            ["Precision", f"{p1.get('precision', 0):.4f}", f"{p2.get('precision', 0):.4f}"],
            ["Recall", f"{p1.get('recall', 0):.4f}", f"{p2.get('recall', 0):.4f}"],
            ["F1 Score", f"{p1.get('f1_score', 0):.4f}", f"{p2.get('f1_score', 0):.4f}"],
        ],
    ))
    story.append(Spacer(1, 8))

    f1, f2 = v1.fairness_metrics, v2.fairness_metrics
    story.append(Paragraph("Fairness (original vs this version)", H3))
    story.append(_metrics_table(
        ["Metric", "Original", method],
        [
            ["Statistical Parity Difference", f"{f1.get('statistical_parity_difference', 0):.4f}",
             f"{f2.get('statistical_parity_difference', 0):.4f}"],
            ["Disparate Impact Ratio", _fmt(f1.get('disparate_impact_ratio')),
             _fmt(f2.get('disparate_impact_ratio'))],
            ["Equal Opportunity Difference", _fmt(f1.get('equal_opportunity_difference')),
             _fmt(f2.get('equal_opportunity_difference'))],
            ["Average Odds Difference", _fmt(f1.get('average_odds_difference')),
             _fmt(f2.get('average_odds_difference'))],
            ["Theil Index", _fmt(f1.get('theil_index')), _fmt(f2.get('theil_index'))],
        ],
    ))
    story.append(Spacer(1, 6))

    accuracy_delta = p2.get("accuracy", 0) - p1.get("accuracy", 0)
    spd_improvement = _spd_improvement(v1, v2)
    story.append(Paragraph(
        f"Verdict: original was <b>{finding1.status.replace('_', ' ')}</b>, "
        f"this version is <b>{finding2.status.replace('_', ' ')}</b>. "
        f"Accuracy change: <b>{accuracy_delta:+.4f}</b>. "
        f"|Statistical Parity Difference| change: "
        f"<b>{-spd_improvement:+.4f}</b> ({'improved' if spd_improvement > 0 else 'did not improve'}).",
        BODY,
    ))
    story.append(Spacer(1, 14))


def build_report(session, run_id: str) -> str:
    run = repo.get_run(session, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    versions = sorted(repo.list_versions_for_run(session, run_id), key=lambda v: v.created_at)
    if len(versions) < 2:
        raise DatasetValidationError(
            "Report generation requires both an original and at least one "
            "debiased version to exist for this run.",
            details={"versions_found": len(versions)},
        )

    v1 = next(v for v in versions if v.mitigation_method is None)
    mitigated_versions = [v for v in versions if v.mitigation_method is not None]

    finding1 = FairnessFinding.model_validate(v1.fairness_finding)
    narrative_v1 = generate_fairness_narrative(finding1)

    story = []

    # --- 1. Executive Summary ---------------------------------------
    story.append(Paragraph("Responsible AI Report", H1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("1. Executive Summary", H2))
    story.append(Paragraph(narrative_v1.replace("\n", "<br/>"), BODY))
    story.append(Spacer(1, 6))
    method_list = ", ".join(v.mitigation_method for v in mitigated_versions)
    story.append(Paragraph(
        f"{len(mitigated_versions)} mitigation method(s) were evaluated against this "
        f"original model: <b>{method_list}</b>. Each is treated as an independent "
        f"alternative below -- see Section {3 + len(mitigated_versions)} for a "
        f"side-by-side comparison and Section {4 + len(mitigated_versions)} for guidance "
        f"on choosing between them.",
        BODY,
    ))
    story.append(Spacer(1, 12))

    # --- 2. Model & Data Overview -------------------------------------
    story.append(Paragraph("2. Model & Data Overview", H2))
    story.append(_metrics_table(
        ["Field", "Value"],
        [
            ["Algorithm", v1.algorithm_name],
            ["Model source", v1.source],
            ["Preprocessing", v1.preprocessing_status],
            ["Protected attribute", run.protected_attribute or "-"],
            ["Privileged group", run.privileged_value or "-"],
            ["Unprivileged group", run.unprivileged_value or "-"],
            ["Target column", run.target_column or "-"],
            ["Test set size (original)", str(v1.performance_metrics.get("test_set_size", "-"))],
        ],
    ))
    story.append(Spacer(1, 12))

    # --- 3. Original Model: Performance, Explainability, Fairness -------
    story.append(Paragraph("3. Original Model", H2))
    p1 = v1.performance_metrics
    story.append(Paragraph("Performance", H3))
    story.append(_metrics_table(
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [[f"{p1.get('accuracy', 0):.4f}", f"{p1.get('precision', 0):.4f}",
          f"{p1.get('recall', 0):.4f}", f"{p1.get('f1_score', 0):.4f}"]],
    ))
    story.append(Spacer(1, 8))

    top_features_v1 = v1.explainability_results.get("top_features", [])[:5]
    if top_features_v1:
        story.append(Paragraph("Top features by SHAP importance:", BODY))
        story.append(_metrics_table(
            ["Feature", "Importance"],
            [[f["feature_name"], f"{f['importance_score']:.5f}"] for f in top_features_v1],
        ))
        story.append(Spacer(1, 8))

    worst_subgroups = v1.error_analysis_results.get("worst_subgroups", [])[:3]
    if worst_subgroups:
        story.append(Paragraph("Underperforming subgroups:", BODY))
        story.append(_metrics_table(
            ["Column", "Subgroup", "Accuracy", "Gap vs Overall"],
            [[s["column"], s["subgroup"], f"{s['subgroup_accuracy']:.3f}", f"{s['accuracy_gap']:.3f}"]
             for s in worst_subgroups],
        ))
        story.append(Spacer(1, 8))

    f1 = v1.fairness_metrics
    story.append(Paragraph("Fairness", H3))
    story.append(_metrics_table(
        ["Metric", "Value"],
        [
            ["Statistical Parity Difference", f"{f1.get('statistical_parity_difference', 0):.4f}"],
            ["Disparate Impact Ratio", _fmt(f1.get('disparate_impact_ratio'))],
            ["Equal Opportunity Difference", _fmt(f1.get('equal_opportunity_difference'))],
            ["Average Odds Difference", _fmt(f1.get('average_odds_difference'))],
            ["Theil Index", _fmt(f1.get('theil_index'))],
        ],
    ))
    story.append(Paragraph(f"Verdict: <b>{finding1.status.replace('_', ' ')}</b>", BODY))
    story.append(Spacer(1, 14))

    # --- One section per mitigated version -------------------------------
    for i, v2 in enumerate(mitigated_versions):
        _add_mitigated_version_section(story, v1, v2, section_no=4 + i)

    # --- Comparison across all versions -----------------------------------
    comparison_section_no = 4 + len(mitigated_versions)
    story.append(Paragraph(f"{comparison_section_no}. Comparison Across All Versions", H2))
    comparison_rows = []
    for v in versions:
        label = v.mitigation_method or "Original"
        f = v.fairness_metrics
        p = v.performance_metrics
        comparison_rows.append([
            label,
            f"{p.get('accuracy', 0):.4f}",
            f"{f.get('statistical_parity_difference', 0):.4f}",
            f"{v.runtime_seconds:.2f}s" if v.runtime_seconds is not None else "-",
        ])
    story.append(_metrics_table(
        ["Version", "Accuracy", "Statistical Parity Diff.", "Runtime"],
        comparison_rows,
    ))
    story.append(Spacer(1, 12))

    # --- Recommendation ----------------------------------------------
    recommendation_section_no = comparison_section_no + 1
    story.append(Paragraph(f"{recommendation_section_no}. Recommendation", H2))
    fair_versions = [v for v in mitigated_versions
                      if FairnessFinding.model_validate(v.fairness_finding).status == "fair"]
    if fair_versions:
        best = max(fair_versions, key=lambda v: v.performance_metrics.get("accuracy", 0))
        recommendation = (
            f"<b>{best.mitigation_method}</b> shows no significant disparity on this test "
            f"set and retains the highest accuracy ({best.performance_metrics.get('accuracy', 0):.4f}) "
            f"among the methods that reached a fair verdict. Review the full comparison "
            f"above before deciding whether to deploy it in place of the original model -- "
            f"other methods may better suit different accuracy/fairness priorities."
        )
    else:
        improved = [(v, _spd_improvement(v1, v)) for v in mitigated_versions]
        improved = [pair for pair in improved if pair[1] > 0]
        if improved:
            best, improvement = max(improved, key=lambda pair: pair[1])
            recommendation = (
                f"No mitigation method reached a fully \"fair\" verdict on this test set. "
                f"<b>{best.mitigation_method}</b> reduced |Statistical Parity Difference| the "
                f"most (by {improvement:.4f}), at an accuracy change of "
                f"{best.performance_metrics.get('accuracy', 0) - p1.get('accuracy', 0):+.4f}. "
                f"Review the full comparison above before deciding whether that trade-off is "
                f"acceptable, or consider a different protected-attribute configuration."
            )
        else:
            recommendation = (
                "None of the evaluated mitigation methods meaningfully reduced disparity "
                "for this model/dataset combination. Consider reviewing the protected "
                "attribute configuration or exploring alternative approaches."
            )
    story.append(Paragraph(recommendation, BODY))
    story.append(Spacer(1, 12))

    # --- Version Lineage -----------------------------------------------
    lineage_section_no = recommendation_section_no + 1
    story.append(Paragraph(f"{lineage_section_no}. Version Lineage", H2))
    story.append(_metrics_table(
        ["Version", "Mitigation", "Created At", "Version ID"],
        [[str(v.version_number), v.mitigation_method or "-",
          v.created_at.strftime("%Y-%m-%d %H:%M UTC") if v.created_at else "-",
          v.version_id[:8] + "..."] for v in versions],
    ))
    story.append(Spacer(1, 12))

    # --- Methodology & Limitations --------------------------------------
    methodology_section_no = lineage_section_no + 1
    story.append(Paragraph(f"{methodology_section_no}. Methodology & Limitations", H2))
    lib_versions = v1.library_versions or {}
    lib_line = ", ".join(f"{k} {v}" for k, v in lib_versions.items())
    story.append(Paragraph(f"<b>Library versions:</b> {lib_line}", BODY))
    story.append(Paragraph(
        f"<b>Preprocessing:</b> {v1.preprocessing_status}. "
        + (
            "This model was uploaded as a bare estimator; preprocessing of "
            "the input datasets is assumed to match the original training "
            "process and could not be independently verified."
            if v1.preprocessing_status == "user_responsibility"
            else "Preprocessing is fully managed by the model's own Pipeline."
        ),
        BODY,
    ))
    any_small_group = any(
        v.fairness_metrics.get("small_group_warning") for v in versions
    )
    if any_small_group:
        story.append(Paragraph(
            "<b>Note:</b> one or more protected-attribute subgroups fell "
            "below the recommended minimum size in this dataset; fairness "
            "metrics for that group should be treated as low-confidence.",
            BODY,
        ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated by an internal Responsible AI platform "
        "and is intended to support, not replace, human review. It is not "
        "a certified fairness or compliance audit.", SMALL,
    ))

    report_path = get_report_path(run_id)
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    doc.build(story)

    repo.set_report_path(session, run_id, report_path)
    return report_path
