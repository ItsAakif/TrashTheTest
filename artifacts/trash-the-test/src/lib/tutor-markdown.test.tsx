import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { TutorMarkdown } from "./tutor-markdown";

// NOTE: no test runner is installed in this workspace (see the note in
// api-services.test.ts). These tests are written to the T020 contract so the
// next session only has to add vitest + @testing-library/react to run them.
describe("TutorMarkdown", () => {
  it("renders inline bold/italic/code/links", () => {
    const { container } = render(
      <TutorMarkdown text="**hi** *there* `code` [link](https://example.com)" />);
    expect(container.querySelector("strong")?.textContent).toBe("hi");
    expect(container.querySelector("em")?.textContent).toBe("there");
    expect(container.querySelector("code")?.textContent).toBe("code");
    const a = container.querySelector("a");
    expect(a?.getAttribute("href")).toBe("https://example.com");
  });

  it("strips raw HTML", () => {
    const { container } = render(
      <TutorMarkdown text='<script>alert(1)</script> safe' />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toContain("safe");
  });

  it("rejects javascript: URLs", () => {
    const { container } = render(
      <TutorMarkdown text="[x](javascript:alert(1))" />);
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("x");
  });

  it("renders headings and lists", () => {
    // Blocks are split on BLANK lines (T012: `text.split(/\n{2,}/)`), so the
    // items below are separated by blank lines; with a single newline the two
    // items would be one block and therefore one <li>.
    const { container } = render(
      <TutorMarkdown text={"# H1\n\n- a\n\n- b\n\n1. one\n\n2. two"} />);
    expect(container.querySelector("h1")?.textContent).toBe("H1");
    expect(container.querySelectorAll("ul li").length).toBe(2);
    expect(container.querySelectorAll("ol li").length).toBe(2);
  });

  it("keeps a javascript: label as plain text with no href anywhere", () => {
    const { container } = render(
      <TutorMarkdown text="[steal](javascript:alert(document.cookie)) and [ok](https://example.com)" />);
    const anchors = container.querySelectorAll("a");
    expect(anchors.length).toBe(1);
    expect(anchors[0].getAttribute("href")).toBe("https://example.com");
    expect(container.textContent).toContain("steal");
    expect(container.innerHTML).not.toContain("javascript:");
  });
});
