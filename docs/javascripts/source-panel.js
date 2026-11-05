// The source panel: files bundled at build time, read in place with the kit's .nb-code form.
// window.SourcePanel.create builds one; the page decides what to run and where the output goes.
(() => {
  "use strict";

  function element(tag, className, attributes = {}) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    return node;
  }

  const fileName = (path) => String(path).split("/").pop();

  // Splits a coloured fragment into lines, keeping each token's class on every line it spans.
  function splitLines(fragment) {
    const lines = [[]];
    for (const node of [...fragment.childNodes]) {
      const className = node.nodeType === Node.ELEMENT_NODE ? node.className : "";
      const pieces = node.textContent.split("\n");
      pieces.forEach((piece, index) => {
        if (index > 0) lines.push([]);
        if (!piece) return;
        if (className) {
          const span = element("span", className);
          span.textContent = piece;
          lines.at(-1).push(span);
        } else {
          lines.at(-1).push(document.createTextNode(piece));
        }
      });
    }
    if (lines.length > 1 && lines.at(-1).length === 0) lines.pop();
    return lines;
  }

  function create({ root, base, files, active, blobBase, onSelect }) {
    root.classList.add("nb-code");
    const tabs = element("div", "nb-code-tabs", { role: "tablist", "aria-label": "文件" });
    const note = element("span", "nb-code-note");
    const view = element("pre", "nb-code-view", { tabindex: "0", role: "tabpanel" });
    const code = element("code");
    view.append(code);
    const actions = element("div", "nb-code-actions");
    const copy = element("button", "nb-button", { type: "button" });
    copy.textContent = "复制";
    const link = element("a", "nb-code-link");
    link.textContent = "在 GitHub 打开";
    const status = element("p", "nb-code-status", { role: "status", "aria-live": "polite" });
    actions.append(copy, link, status);
    root.replaceChildren(tabs, view, actions);

    const cache = new Map();
    const buttons = new Map();
    let current = null;
    let marked = [];
    let override = null;

    function load(path) {
      if (!cache.has(path)) {
        cache.set(path, fetch(`${base}${path}.txt`).then((response) => {
          if (!response.ok) throw new Error(`无法读取 ${path}（${response.status}）`);
          return response.text();
        }).then((text) => text.replace(/\r\n/g, "\n")));
      }
      return cache.get(path);
    }

    function renderText(path, text) {
      const colours = window.CodeColors;
      const fragment = colours ? colours.highlight(text, colours.languageFor(path)) : document.createDocumentFragment();
      if (!colours) fragment.append(text);
      const lines = splitLines(fragment);
      const output = document.createDocumentFragment();
      lines.forEach((nodes, index) => {
        const line = element("div", "nb-code-line", { "data-line": String(index + 1), id: `L${index + 1}` });
        if (marked.some(([from, to]) => index + 1 >= from && index + 1 <= to)) line.setAttribute("data-marked", "");
        // One child beside the line number: a grid lays out every child, so the tokens must share a span.
        const text = element("span", "nb-code-text");
        text.append(...nodes);
        line.append(text);
        output.append(line);
      });
      code.replaceChildren(output);
      note.textContent = `${lines.length} 行`;
    }

    async function select(path) {
      if (!buttons.has(path)) return;
      current = path;
      for (const [key, button] of buttons) button.setAttribute("aria-selected", String(key === path));
      link.href = blobBase ? blobBase + path.split("/").map(encodeURIComponent).join("/") : "";
      link.hidden = !blobBase;
      view.hidden = false;
      if (override?.path === path && override.element) {
        view.hidden = true;
        override.element.hidden = false;
      } else if (override?.element) {
        override.element.hidden = true;
      }
      status.textContent = "";
      try {
        const text = await load(path);
        if (current !== path) return;
        renderText(path, text);
        onSelect?.(path, text);
      } catch (error) {
        code.replaceChildren();
        note.textContent = "";
        status.textContent = `${error.message}。可以在 GitHub 打开这个文件。`;
      }
    }

    for (const file of files) {
      const button = element("button", "nb-code-tab", { type: "button", role: "tab", "aria-selected": "false", "data-path": file.path });
      button.textContent = file.label ?? fileName(file.path);
      button.title = file.path;
      button.addEventListener("click", () => select(file.path));
      buttons.set(file.path, button);
      tabs.append(button);
    }
    tabs.append(note);

    copy.addEventListener("click", async () => {
      if (!current) return;
      const text = override?.path === current && override.value ? override.value() : await load(current);
      try {
        await navigator.clipboard.writeText(text);
        status.textContent = `已复制 ${fileName(current)}`;
      } catch {
        status.textContent = "无法访问剪贴板，请手动选中复制。";
      }
    });

    function mark(ranges) {
      marked = ranges;
      if (current) load(current).then((text) => renderText(current, text)).catch(() => {});
    }

    // The page swaps the view for an editor on one file; the panel keeps the tabs honest.
    function setOverride(next) {
      if (override?.element && override.element !== next?.element) override.element.hidden = true;
      override = next;
      if (current) select(current);
    }

    select(active ?? files[0]?.path);

    return Object.freeze({
      element: root,
      view,
      actions,
      status,
      select,
      load,
      mark,
      setOverride,
      current: () => current,
      scrollToLine(line) {
        const target = code.querySelector(`[data-line="${line}"]`);
        if (target) target.scrollIntoView({ block: "center" });
      },
    });
  }

  function parseRange(hash) {
    const match = String(hash).match(/^#?L(\d+)(?:-L?(\d+))?$/);
    if (!match) return null;
    const from = Number(match[1]);
    const to = Number(match[2] ?? match[1]);
    return [Math.min(from, to), Math.max(from, to)];
  }

  window.SourcePanel = Object.freeze({ create, parseRange, fileName });
})();
