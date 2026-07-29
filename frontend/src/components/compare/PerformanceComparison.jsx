import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function PerformanceComparison({ versions }) {
  const data = versions.map((v) => ({
    name: v.mitigation_method || "Original",
    Accuracy: v.performance_metrics?.accuracy ?? 0,
    Precision: v.performance_metrics?.precision ?? 0,
    Recall: v.performance_metrics?.recall ?? 0,
    F1: v.performance_metrics?.f1_score ?? 0,
  }));

  return (
    <div className="chart-block">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={60} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
          <Legend />
          <Bar dataKey="Accuracy" fill="var(--color-blue)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Precision" fill="var(--color-sage-strong)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Recall" fill="var(--color-coral-strong)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="F1" fill="var(--color-ink-faint)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
