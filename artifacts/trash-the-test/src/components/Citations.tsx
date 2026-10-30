import { Link } from "wouter";
import { citationLabel, type Citation } from "@/lib/api-services";

interface CitationsProps {
  citations?: Citation[];
  sources?: string[];
}

/**
 * Citations UI: renders chunk citation span/page/method, never bare ids.
 * Page-aware cites (document_id + page) are clickable links into the
 * EXISTING document viewer (?page=N citation jump); cites without a
 * resolvable page stay as descriptive spans. Used in tutor, study, and
 * website contexts.
 */
export default function Citations({ citations, sources }: CitationsProps) {
  if (citations && citations.length > 0) {
    return (
      <div className="mt-2 flex gap-2 flex-wrap">
        {citations.map((c, i) => {
          const label = citationLabel(c, i);
          const title = c.span || c.chunk_id;
          const style = {
            backgroundColor: "rgba(140,161,170,0.1)",
            color: "#8ca1aa",
            fontFamily: "Space Grotesk, sans-serif",
          };
          const className = "px-2 py-0.5 rounded text-xs hover:opacity-80";
          // Clickable + page-aware where supported: jump the existing
          // viewer to the cited page. Otherwise a descriptive span.
          if (c.document_id && c.page !== undefined && c.page !== null && c.page > 0) {
            return (
              <Link
                key={`${c.chunk_id}-${i}`}
                href={`/documents/${c.document_id}?page=${c.page}`}
                title={`${title} — open source page ${c.page}`}
                className={className}
                style={{ ...style, textDecoration: "underline" }}
              >
                {label}
              </Link>
            );
          }
          return (
            <span key={`${c.chunk_id}-${i}`} title={title} className={className} style={style}>
              {label}
            </span>
          );
        })}
      </div>
    );
  }
  if (sources && sources.length > 0) {
    return (
      <div className="mt-2 flex gap-2 flex-wrap">
        {sources.map((s, i) => (
          <span
            key={`${s}-${i}`}
            title={s}
            className="px-2 py-0.5 rounded text-xs"
            style={{
              backgroundColor: "rgba(140,161,170,0.1)",
              color: "#8ca1aa",
              fontFamily: "Space Grotesk, sans-serif",
            }}
          >
            chunk {String(s).slice(0, 8)} (id only — detail unavailable)
          </span>
        ))}
      </div>
    );
  }
  return null;
}
