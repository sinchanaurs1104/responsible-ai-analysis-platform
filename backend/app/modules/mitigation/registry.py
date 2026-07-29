"""
Registry mapping a mitigation method name (matching
fairness.thresholds.MITIGATION_LOOKUP's values, e.g. "Reweighing") to
its intervention category and strategy instance.

Adding a new method later (Phase 3+) is one new strategy file + one new
entry here -- no changes to the orchestrator or retraining logic, since
the orchestrator only ever branches on category, never on method name.
"""

from dataclasses import dataclass

from app.core.exceptions import ConfigValidationError
from app.modules.mitigation.preprocessing.reweighing import ReweighingStrategy
from app.modules.mitigation.preprocessing.disparate_impact_remover import DisparateImpactRemoverStrategy
from app.modules.mitigation.postprocessing.calibrated_eq_odds import CalibratedEqualizedOddsStrategy
from app.modules.mitigation.postprocessing.reject_option_classification import RejectOptionClassificationStrategy

CATEGORY_PREPROCESSING = "pre"
CATEGORY_INPROCESSING = "in"
CATEGORY_POSTPROCESSING = "post"


@dataclass
class MitigationRegistration:
    category: str
    strategy: object  # PreprocessingStrategy | InprocessingStrategy | PostprocessingStrategy


MITIGATION_REGISTRY: dict[str, MitigationRegistration] = {
    "Reweighing": MitigationRegistration(
        category=CATEGORY_PREPROCESSING,
        strategy=ReweighingStrategy(),
    ),
    "Disparate Impact Remover": MitigationRegistration(
        category=CATEGORY_PREPROCESSING,
        strategy=DisparateImpactRemoverStrategy(),
    ),
    "Calibrated Equalized Odds Postprocessing": MitigationRegistration(
        category=CATEGORY_POSTPROCESSING,
        strategy=CalibratedEqualizedOddsStrategy(),
    ),
    "Reject Option Classification": MitigationRegistration(
        category=CATEGORY_POSTPROCESSING,
        strategy=RejectOptionClassificationStrategy(),
    ),
}


def get_registration(method_name: str) -> MitigationRegistration:
    registration = MITIGATION_REGISTRY.get(method_name)
    if registration is None:
        raise ConfigValidationError(
            f"Mitigation method '{method_name}' is not registered.",
            details={"available_methods": list(MITIGATION_REGISTRY.keys())},
        )
    return registration
