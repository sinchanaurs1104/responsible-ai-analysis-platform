import { NavLink } from "react-router-dom";
import { WORKFLOW_STAGES } from "../api/constants";
import { useRun } from "../api/RunContext";
import ScaleGlyph from "./ScaleGlyph";
import "./Sidebar.css";

export default function Sidebar() {
  const { furthestStage, STAGE_ORDER, runId, datasetName } = useRun();
  const furthestIdx = STAGE_ORDER.indexOf(furthestStage);

  return (
    <nav className="sidebar">
      <div className="sidebar__brand">
        <ScaleGlyph value={0.15} size="sm" />
        <div>
          <div className="sidebar__brand-name">Responsible AI</div>
          <div className="sidebar__brand-sub">Platform</div>
        </div>
      </div>

      <ol className="sidebar__stages">
        {WORKFLOW_STAGES.map((stage, idx) => {
          const reachable = idx <= furthestIdx;
          return (
            <li key={stage.key}>
              <NavLink
                to={reachable ? `/${stage.key}` : "#"}
                className={({ isActive }) =>
                  [
                    "sidebar__stage",
                    isActive ? "sidebar__stage--active" : "",
                    reachable ? "" : "sidebar__stage--locked",
                  ].join(" ")
                }
                onClick={(e) => {
                  if (!reachable) e.preventDefault();
                }}
              >
                <span className="sidebar__stage-index numeric">{idx + 1}</span>
                <span>{stage.label}</span>
              </NavLink>
            </li>
          );
        })}
      </ol>

      <div className="sidebar__footer">
        {runId ? (
          <>
            <div className="sidebar__footer-label">Current run</div>
            <div className="sidebar__footer-value" title={datasetName || runId}>
              {datasetName || runId}
            </div>
          </>
        ) : (
          <div className="sidebar__footer-label">No run started</div>
        )}
      </div>
    </nav>
  );
}
