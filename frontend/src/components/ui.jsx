import "./ui.css";

export function Card({ children, className = "", as: Tag = "div", ...rest }) {
  return (
    <Tag className={`card ${className}`} {...rest}>
      {children}
    </Tag>
  );
}

export function PageHeader({ eyebrow, title, description }) {
  return (
    <header className="page-header">
      {eyebrow && <div className="page-header__eyebrow">{eyebrow}</div>}
      <h1 className="page-header__title">{title}</h1>
      {description && <p className="page-header__desc">{description}</p>}
    </header>
  );
}

export function Button({ variant = "primary", children, className = "", ...rest }) {
  return (
    <button className={`btn btn--${variant} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function Badge({ tone = "neutral", children }) {
  return <span className={`badge badge--${tone}`}>{children}</span>;
}

export function WarningBanner({ warnings }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div className="warning-banner">
      <div className="warning-banner__title">Scope adjustments</div>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="error-banner">
      <div className="error-banner__title">Something went wrong</div>
      <p>{message}</p>
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}

export function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className="metric">
      <div className="metric__label">{label}</div>
      <div className={`metric__value numeric metric__value--${tone}`}>{value}</div>
    </div>
  );
}
