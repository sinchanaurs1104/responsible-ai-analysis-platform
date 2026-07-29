import { getDownloadUrl } from "../../api/client";
import { fmtPct } from "../version-analysis/PerformanceSection";
import { Button, Badge } from "../ui";

export default function SummaryTable({ versions }) {
  return (
    <table className="data-table compare-table">
      <thead>
        <tr>
          <th>Version</th>
          <th>Category</th>
          <th>Accuracy</th>
          <th>Fairness status</th>
          <th>Runtime</th>
          <th>Download</th>
        </tr>
      </thead>
      <tbody>
        {versions.map((v) => (
          <tr key={v.version_id}>
            <td>{v.mitigation_method || "Original"}</td>
            <td>
              {v.mitigation_category ? (
                <Badge tone="accent">{v.mitigation_category}</Badge>
              ) : (
                <Badge tone="neutral">baseline</Badge>
              )}
            </td>
            <td className="numeric">{fmtPct(v.performance_metrics?.accuracy)}</td>
            <td>
              <FairnessStatusBadge status={v.fairness_finding?.status} />
            </td>
            <td className="numeric">
              {v.runtime_seconds != null ? `${v.runtime_seconds.toFixed(2)}s` : "—"}
            </td>
            <td>
              {v.has_downloadable_model && v.mitigation_category !== "post" ? (
                <a href={getDownloadUrl(v.version_id)} target="_blank" rel="noreferrer">
                  <Button variant="ghost">Download</Button>
                </a>
              ) : (
                "—"
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const STATUS_TONE = { fair: "positive", moderate_disparity: "neutral", high_disparity: "negative" };
const STATUS_LABEL = { fair: "Fair", moderate_disparity: "Moderate", high_disparity: "High disparity" };

export function FairnessStatusBadge({ status }) {
  if (!status) return "—";
  return <Badge tone={STATUS_TONE[status] || "neutral"}>{STATUS_LABEL[status] || status}</Badge>;
}
