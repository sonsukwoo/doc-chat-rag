import type { StageStatus } from "../types";

interface StageBadgeProps {
  label: string;
  status?: StageStatus;
}

export function StageBadge({ label, status = "not_started" }: StageBadgeProps) {
  return (
    <span className={`stage-badge stage-${status}`}>
      <span className="stage-badge-label">{label}</span>
      <span className="stage-badge-status">{status.replace(/_/g, " ")}</span>
    </span>
  );
}
