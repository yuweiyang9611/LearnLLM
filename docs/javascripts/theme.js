// Load synchronously in <head>: sets the colour scheme before the first paint.
(() => {
  "use strict";

  const root = document.documentElement;
  const storageKey = "site-theme";
  const paper = { light: "#f4efe4", dark: "#1c1a17" };
  const schemes = [["light", "白天"], ["dark", "晚上"]];

  function saved() {
    try {
      const value = localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : null;
    } catch {
      return null;
    }
  }

  function apply(theme) {
    root.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", paper[theme]);
    document.querySelectorAll(".nb-theme button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.scheme === theme));
    });
  }

  // Chromium squeezes most touching Chinese marks, but leaves a curly quote beside a comma or full stop
  // a full em apart, as in ”， and ，“. The font's `halt` feature sets the first of the two at half width.
  const looseMarks = /[”’](?=[，。、；：？！])|[，。、；：？！](?=[“‘])/g;
  const notProse = "code, kbd, pre, samp, script, style, textarea, svg, [contenteditable], .nb-halt";

  function squeeze(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const loose = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.data.search(looseMarks) !== -1 && !node.parentElement?.closest(notProse)) loose.push(node);
    }
    for (const node of loose) {
      const parts = document.createDocumentFragment();
      let start = 0;
      for (const match of node.data.matchAll(looseMarks)) {
        const mark = document.createElement("span");
        mark.className = "nb-halt";
        mark.textContent = match[0];
        if (match.index > start) parts.append(node.data.slice(start, match.index));
        parts.append(mark);
        start = match.index + 1;
      }
      if (start < node.data.length) parts.append(node.data.slice(start));
      node.replaceWith(parts);
    }
  }

  const system = window.matchMedia("(prefers-color-scheme: dark)");
  apply(saved() ?? (system.matches ? "dark" : "light"));
  system.addEventListener("change", (event) => {
    if (!saved()) apply(event.matches ? "dark" : "light");
  });

  // Fonts are shared across the family, so a course site's local test server has none to serve.
  if (location.hostname === "yuweiyang9611.github.io") {
    const fonts = document.createElement("link");
    fonts.rel = "stylesheet";
    fonts.href = "/kit/v1/fonts.css";
    document.head.append(fonts);
  }

  document.addEventListener("DOMContentLoaded", () => {
    squeeze(document.body);
    // Text that scripts add later, such as search results, gets the same treatment.
    const later = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) {
          const root = node.nodeType === Node.TEXT_NODE ? node.parentNode : node;
          if (root?.isConnected) squeeze(root);
        }
      }
      later.takeRecords(); // the wrapping above is not new text
    });
    later.observe(document.body, { childList: true, subtree: true });
  });

  document.addEventListener("DOMContentLoaded", () => {
    const nav = document.querySelector(".nb-nav");
    if (!nav) return;
    const group = document.createElement("div");
    group.className = "nb-theme";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "配色方案");
    for (const [scheme, word] of schemes) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.scheme = scheme;
      button.textContent = word;
      button.addEventListener("click", () => {
        apply(scheme);
        try {
          localStorage.setItem(storageKey, scheme);
        } catch {
          // The scheme still applies for this visit if storage is unavailable.
        }
      });
      group.append(button);
    }
    nav.after(group);
    apply(root.dataset.theme);
  });
})();
