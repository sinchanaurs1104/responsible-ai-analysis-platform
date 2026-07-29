/**
 * Signature glyph: two horizontal bars diverging from a center line,
 * representing privileged vs unprivileged group standing on some
 * metric. Reused as: nav icon, version-card thumbnail (mini, static),
 * and full-size on the Compare page (data-driven, per version).
 *
 * value: -1..1, where 0 is perfect parity. Negative = unprivileged
 * group favored, positive = privileged group favored (matches
 * statistical_parity_difference sign convention used by fairness.metrics).
 */
export default function ScaleGlyph({ value = 0, size = "md", labelled = false }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const dims = { sm: { w: 28, h: 16 }, md: { w: 56, h: 32 }, lg: { w: 160, h: 64 } }[size];
  const { w, h } = dims;
  const midX = w / 2;
  const barH = size === "lg" ? 10 : size === "md" ? 6 : 3;
  const maxReach = midX - 4;
  const reach = Math.abs(clamped) * maxReach;

  const barColor =
    Math.abs(clamped) < 0.05
      ? "var(--color-neutral-accent)"
      : Math.abs(clamped) < 0.2
      ? "var(--color-sage-strong)"
      : "var(--color-negative)";

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="group parity indicator">
      <line x1={midX} y1={0} x2={midX} y2={h} stroke="var(--color-border)" strokeWidth="1.5" />
      <line x1={4} y1={h / 2} x2={w - 4} y2={h / 2} stroke="var(--color-border)" strokeWidth="1" />
      {clamped >= 0 ? (
        <rect x={midX} y={h / 2 - barH / 2} width={reach} height={barH} rx={barH / 2} fill={barColor} />
      ) : (
        <rect x={midX - reach} y={h / 2 - barH / 2} width={reach} height={barH} rx={barH / 2} fill={barColor} />
      )}
      <circle cx={midX} cy={h / 2} r={size === "lg" ? 4 : 2.5} fill="var(--color-ink)" />
      {labelled && (
        <>
          <text x={4} y={h - 2} fontSize="9" fill="var(--color-ink-faint)" fontFamily="var(--font-mono)">
            unpriv.
          </text>
          <text x={w - 4} y={h - 2} fontSize="9" fill="var(--color-ink-faint)" textAnchor="end" fontFamily="var(--font-mono)">
            priv.
          </text>
        </>
      )}
    </svg>
  );
}
