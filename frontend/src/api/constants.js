// Mirrors app/modules/mitigation/registry.py -- keep in sync manually,
// there is no backend endpoint that lists methods.
export const MITIGATION_METHODS = [
  { name: "Reweighing", category: "pre", label: "Reweighing" },
  { name: "Disparate Impact Remover", category: "pre", label: "Disparate Impact Remover" },
  { name: "Calibrated Equalized Odds Postprocessing", category: "post", label: "Calibrated Equalized Odds" },
  { name: "Reject Option Classification", category: "post", label: "Reject Option Classification" },
];

// Mirrors app/modules/training/trainer.py supported algorithms.
export const ALGORITHMS = [
  { name: "LogisticRegression", label: "Logistic Regression" },
  { name: "DecisionTreeClassifier", label: "Decision Tree" },
  { name: "RandomForestClassifier", label: "Random Forest" },
  { name: "GradientBoostingClassifier", label: "Gradient Boosting" },
];

export const WORKFLOW_STAGES = [
  { key: "upload", label: "Upload & Train" },
  { key: "configure", label: "Configure" },
  { key: "mitigate", label: "Mitigate" },
  { key: "versions", label: "Versions" },
  { key: "compare", label: "Compare" },
];
