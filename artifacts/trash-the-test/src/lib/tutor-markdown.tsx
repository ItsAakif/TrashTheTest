import { Fragment, type ElementType, type ReactNode } from "react";

const SAFE_HREF = /^https?:\/\//i;

function renderInline(text: string, keyBase: string): ReactNode[] {
    // Order matters: code (so '*' inside backticks is not bolded),
    // then bold, then italic, then links.  Backslash-escaped
    // markers are not supported (deliberate small scope).
    const out: ReactNode[] = [];
    let i = 0;
    let buf = "";
    const flush = () => {
        if (buf) { out.push(buf); buf = ""; }
    };
    while (i < text.length) {
        if (text[i] === "`") {
            const end = text.indexOf("`", i + 1);
            if (end > i) {
                flush();
                out.push(
                    <code key={`${keyBase}-c-${i}`}
                          style={{ background: "#252626",
                                   padding: "0 4px",
                                   borderRadius: 4,
                                   fontFamily: "JetBrains Mono, monospace" }}>
                        {text.slice(i + 1, end)}
                    </code>
                );
                i = end + 1;
                continue;
            }
        }
        if (text.startsWith("**", i)) {
            const end = text.indexOf("**", i + 2);
            if (end > i + 2) {
                flush();
                out.push(<strong key={`${keyBase}-b-${i}`}>
                            {text.slice(i + 2, end)}</strong>);
                i = end + 2;
                continue;
            }
        }
        if (text[i] === "*") {
            const end = text.indexOf("*", i + 1);
            if (end > i) {
                flush();
                out.push(<em key={`${keyBase}-i-${i}`}>
                            {text.slice(i + 1, end)}</em>);
                i = end + 1;
                continue;
            }
        }
        if (text[i] === "[") {
            const close = text.indexOf("]", i + 1);
            const paren = close > i ? text.indexOf("(", close + 1) : -1;
            const parenEnd = paren > 0 ? text.indexOf(")", paren + 1) : -1;
            if (close > i && paren === close + 1 && parenEnd > paren) {
                const label = text.slice(i + 1, close);
                const href = text.slice(paren + 1, parenEnd);
                flush();
                if (SAFE_HREF.test(href)) {
                    out.push(
                        <a key={`${keyBase}-a-${i}`} href={href}
                           target="_blank" rel="noopener noreferrer"
                           style={{ color: "#8ca1aa",
                                    textDecoration: "underline" }}>
                            {label}
                        </a>
                    );
                } else {
                    // Unsafe/unknown scheme: keep the label, drop the link
                    // syntax entirely (ADR-008 acceptance: the node renders
                    // plain text, never a javascript: href).
                    out.push(<span key={`${keyBase}-t-${i}`}>{label}</span>);
                }
                i = parenEnd + 1;
                continue;
            }
        }
        buf += text[i];
        i += 1;
    }
    flush();
    return out;
}

function renderBlock(block: string, idx: number): ReactNode {
    const trimmed = block.trim();
    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
    if (heading) {
        const level = heading[1].length;
        // React 19 types moved JSX out of the global namespace, so the tag
        // is typed as ElementType instead of keyof JSX.IntrinsicElements.
        const Tag = (`h${Math.min(6, level)}` as unknown) as ElementType;
        return <Tag key={`h-${idx}`}>{renderInline(heading[2],
                                                    `h-${idx}`)}</Tag>;
    }
    if (/^>\s+/.test(trimmed)) {
        return <blockquote key={`q-${idx}`}
                           style={{ borderLeft: "3px solid #454747",
                                    paddingLeft: 12, margin: "8px 0",
                                    color: "#acabaa" }}>
                   {renderInline(trimmed.replace(/^>\s+/, ""),
                                 `q-${idx}`)}
               </blockquote>;
    }
    if (/^[-*]\s+/.test(trimmed)) {
        return <li key={`li-${idx}`}>
                {renderInline(trimmed.replace(/^[-*]\s+/, ""),
                               `li-${idx}`)}
               </li>;
    }
    if (/^\d+\.\s+/.test(trimmed)) {
        return <li key={`ol-${idx}`}>
                {renderInline(trimmed.replace(/^\d+\.\s+/, ""),
                               `ol-${idx}`)}
               </li>;
    }
    const lines = trimmed.split("\n");
    return <p key={`p-${idx}`}>
           {lines.map((line, j) => (
               <Fragment key={`p-${idx}-${j}`}>
                   {j > 0 ? <br /> : null}
                   {renderInline(line, `p-${idx}-${j}`)}
               </Fragment>
           ))}
           </p>;
}

function listItemKey(node: ReactNode): "ul" | "ol" | null {
    if (!node || typeof node !== "object" || !("key" in node)) return null;
    const key = String(node.key);
    // `li-*` keys come from the unordered branch of `renderBlock`,
    // `ol-*` from the ordered one; the list TAG differs from the key prefix.
    if (key.startsWith("li-")) return "ul";
    if (key.startsWith("ol-")) return "ol";
    return null;
}

export function TutorMarkdown({ text }: { text: string }) {
    if (!text) return null;
    const blocks = text.replace(/\r\n/g, "\n").split(/\n{2,}/);
    // Group consecutive list items into <ul>/<ol>.
    const grouped: ReactNode[] = [];
    let listBuffer: ReactNode[] = [];
    let listKind: "ul" | "ol" | null = null;
    const flushList = () => {
        if (listKind && listBuffer.length) {
            const Tag = listKind;
            grouped.push(<Tag key={`list-${grouped.length}`}>
                             {listBuffer}</Tag>);
        }
        listBuffer = [];
        listKind = null;
    };
    blocks.forEach((b, i) => {
        const node = renderBlock(b, i);
        const kind = listItemKey(node);
        if (kind) {
            if (listKind !== kind) flushList();
            listKind = kind;
            listBuffer.push(node);
        } else {
            flushList();
            grouped.push(node);
        }
    });
    flushList();
    return <div className="space-y-2 leading-relaxed text-sm"
                style={{ wordBreak: "break-word" }}>{grouped}</div>;
}
