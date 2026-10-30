/**
 * Inline SVG icons for ligature-critical spots.
 *
 * The icon font (Material Symbols Outlined) renders ligature text such as
 * "sparkles" until the webfont arrives; on slow/blocked font loads the
 * fallback word is visible and half-clipped by fixed-size tiles (the
 * "RKLES" garble) and the generating-button spinner shows a spinning
 * "C"/"refresh" word. These inline SVGs never depend on font delivery,
 * so the Generate Notes tile, generating spinners, and route/viewer
 * loading states always render. Decorative small status icons elsewhere
 * keep ligatures (font delivery is fixed via index.html link+preconnect
 * and the index.css fallback stack).
 */

interface IconProps {
  size?: number;
  color?: string;
  className?: string;
  label?: string;
}

export function SparklesIcon({ size = 32, color = "#edfee9", className, label }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path
        d="M12 2.5l2.1 6.4 6.4 2.1-6.4 2.1L12 19.5l-2.1-6.4-6.4-2.1 6.4-2.1L12 2.5z"
        fill={color}
      />
      <path
        d="M19 2.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2z"
        fill={color}
        opacity={0.7}
      />
    </svg>
  );
}

export function SpinnerIcon({ size = 18, color = "#c6c6c7", className, label }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className ? `animate-spin ${className}` : "animate-spin"}
      role={label ? "img" : undefined}
      aria-label={label ?? "Loading"}
    >
      <circle cx="12" cy="12" r="10" stroke={color} strokeOpacity={0.25} strokeWidth={3} />
      <path
        d="M22 12a10 10 0 0 0-10-10"
        stroke={color}
        strokeWidth={3}
        strokeLinecap="round"
      />
    </svg>
  );
}
