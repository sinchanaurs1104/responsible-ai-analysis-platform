import { createContext, useContext, useState, useCallback, useMemo } from "react";

const RunContext = createContext(null);

const initialState = {
  runId: null,
  workflowType: null, // "build_and_analyze" | "analyze_existing"
  datasetName: null,
  algorithmName: null,
  targetColumn: null,
  protectedAttribute: null,
  privilegedValue: null,
  unprivilegedValue: null,
  validationWarnings: [],
  status: null, // "pending" | "running" | "completed" | "failed"
  currentStage: null,
  errorMessage: null,
  mitigationMethods: [],
  failedMethods: [],
  versions: [],
  furthestStage: "upload",
};

const STAGE_ORDER = ["upload", "configure", "mitigate", "versions", "compare"];

export function RunProvider({ children }) {
  const [state, setState] = useState(initialState);

  const patch = useCallback((fields) => {
    setState((prev) => ({ ...prev, ...fields }));
  }, []);

  const markStageReached = useCallback((stageKey) => {
    setState((prev) => {
      const prevIdx = STAGE_ORDER.indexOf(prev.furthestStage);
      const nextIdx = STAGE_ORDER.indexOf(stageKey);
      if (nextIdx <= prevIdx) return prev;
      return { ...prev, furthestStage: stageKey };
    });
  }, []);

  const reset = useCallback(() => setState(initialState), []);

  const value = useMemo(
    () => ({ ...state, patch, markStageReached, reset, STAGE_ORDER }),
    [state, patch, markStageReached, reset]
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun() {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used within a RunProvider");
  return ctx;
}
