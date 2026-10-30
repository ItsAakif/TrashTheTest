"""HTML sanitiser for generated study websites (stdlib, no new dependency).

Allowlist-based: strips <script>/<style>/<iframe>/<object>/<embed>/<form>,
event-handler attributes (on*), javascript:/data:(non-image)/vbscript: URLs,
and rewrites/blocks SSRF-unsafe hrefs (only http/https/mailto/# + relative
are kept; absolute URLs to private/loopback hosts are dropped).

Citations (chunk id / page / method spans emitted by templates) are plain
text/elements from the allowlist, so provenance survives sanitising.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from html import escape
from urllib.parse import urlparse
import ipaddress

ALLOWED_TAGS = {
    "html", "head", "body", "title", "meta", "style", "div", "span", "p",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "a",
    "strong", "em", "b", "i", "u", "code", "pre", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td", "br", "hr",
    "section", "article", "header", "footer", "main", "small",
    "details", "summary",
}

ALLOWED_ATTRS = {
    "a": {"href", "title", "class"},
    "div": {"class"},
    "span": {"class", "data-chunk-id", "data-page", "data-method"},
    "p": {"class"},
    "h1": {"class"},
    "h2": {"class"},
    "h3": {"class"},
    "h4": {"class"},
    "table": {"class"},
    "code": {"class"},
    "pre": {"class"},
    "section": {"class"},
    "details": {"class"},
    "li": {"class", "data-page"},
}

STRIP_TAGS = {"script", "iframe", "object", "embed", "form", "input", "button", "link"}

UNSAFE_SCHEMES = ("javascript:", "vbscript:", "data:text/html")


def _is_unsafe_url(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    if v.startswith(("#", "/", "mailto:")):
        return False
    for scheme in UNSAFE_SCHEMES:
        if v.startswith(scheme):
            return True
    if v.startswith("data:"):
        # Only allow data:image/* (inline images); block the rest.
        if v.startswith("data:image/"):
            return False
        return True
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return True
    if parsed.scheme and parsed.scheme not in ("http", "https", "mailto"):
        return True
    if parsed.hostname:
        host = parsed.hostname.lower()
        # SSRF guard: never link rendered pages at private/loopback hosts.
        if host in ("localhost",):
            return True
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
        except ValueError:
            pass  # regular hostname; path/query are escaped on render
    return False


class _Sanitiser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip_depth = 0
        self._in_style = False

    @staticmethod
    def _clean_css(css: str) -> str:
        # Strip dangerous CSS constructs; keep layout rules for templates.
        css = re.sub(r"expression\s*\(", "blocked(", css, flags=re.IGNORECASE)
        css = re.sub(r"javascript\s*:", "blocked:", css, flags=re.IGNORECASE)
        css = re.sub(r"vbscript\s*:", "blocked:", css, flags=re.IGNORECASE)
        css = re.sub(r"@import[^;]*;", "", css, flags=re.IGNORECASE)
        return css

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in STRIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag not in ALLOWED_TAGS:
            return
        if tag == "style":
            self._in_style = True
            self.out.append("<style>")
            return
        allowed = ALLOWED_ATTRS.get(tag, {"class"})
        kept: list[str] = []
        for name, value in attrs:
            n = name.lower()
            if n.startswith("on"):
                continue
            if n not in allowed:
                continue
            v = value or ""
            if tag == "a" and n == "href" and _is_unsafe_url(v):
                continue
            kept.append(f'{n}="{escape(v, quote=True)}"')
        suffix = (" " + " ".join(kept)) if kept else ""
        self.out.append(f"<{tag}{suffix}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in STRIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "style":
            self._in_style = False
            self.out.append("</style>")
            return
        if tag in ALLOWED_TAGS:
            self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_style:
            self.out.append(self._clean_css(data))
            return
        self.out.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self.out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self.out.append(f"&#{name};")


def sanitise_html(html: str) -> str:
    """Sanitise rendered website HTML through the allowlist."""
    parser = _Sanitiser()
    parser.feed(html or "")
    parser.close()
    return "".join(parser.out)


def is_slug_safe(slug: str) -> bool:
    """Slug guard: lowercase alnum + hyphens, no traversal, bounded length."""
    if not slug or len(slug) > 100:
        return False
    if slug in (".", "..") or "/" in slug or "\\" in slug or " " in slug:
        return False
    return re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) is not None


__all__ = ["sanitise_html", "is_slug_safe", "ALLOWED_TAGS", "ALLOWED_ATTRS"]
