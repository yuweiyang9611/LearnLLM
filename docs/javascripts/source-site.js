// Source read in place: the panel a chapter embeds for its experiment scripts, and the 源码 page that shows
// any bundled file (?path=, with #L5-L9 for a line range) or a directory of them (?dir=). The files come from
// assets/source/, copied from the repository at the site's own revision by scripts/bundle_source.py.
(() => {
  "use strict";

  const SOURCE_PAGE = "%E6%BA%90%E7%A0%81/";
  // Material's base_url is "." on the root page, ".." below it and "/LearnLLM/" on the 404 page.
  const siteBase = (document.querySelector('meta[name="llm-base"]')?.content || ".").replace(/\/*$/, "/");
  const bundleBase = `${siteBase}assets/source/`;

  let indexPromise = null;
  function loadIndex() {
    indexPromise ??= fetch(`${bundleBase}index.json`).then((response) => {
      if (!response.ok) throw new Error(`无法读取源码索引（${response.status}）`);
      return response.json();
    });
    return indexPromise;
  }

  function sourceUrl(path, directory = false) {
    return `${siteBase}${SOURCE_PAGE}?${directory ? "dir" : "path"}=${encodeURIComponent(path)}`;
  }

  function link(path, text, directory = false) {
    const anchor = document.createElement("a");
    anchor.href = sourceUrl(path, directory);
    anchor.textContent = text ?? path;
    return anchor;
  }

  // ---------------------------------------------------------------- Chapter panels

  function mountChapterPanels() {
    const roots = document.querySelectorAll(".llm-source[data-source]");
    if (!roots.length || !window.SourcePanel) return;
    loadIndex().then((index) => {
      const bundled = new Set(index.files.map((file) => file.path));
      for (const root of roots) {
        const paths = root.dataset.source.split(/\s+/).filter(Boolean);
        const files = paths.map((path) => ({ path, label: window.SourcePanel.fileName(path) }));
        const panel = window.SourcePanel.create({
          root,
          base: bundleBase,
          files,
          active: root.dataset.active || paths[0],
          blobBase: index.blobBase,
        });
        // A second small link beside the GitHub one: the same file on the 源码 page, with its siblings.
        const inSite = document.createElement("a");
        inSite.className = "llm-source-open";
        inSite.textContent = "在源码页打开";
        panel.actions.insertBefore(inSite, panel.actions.querySelector(".nb-code-link"));
        const follow = (path) => {
          inSite.href = sourceUrl(path);
          inSite.hidden = !bundled.has(path);
        };
        follow(panel.current() ?? paths[0]);
        root.querySelector(".nb-code-tabs").addEventListener("click", () => follow(panel.current()));
      }
    }).catch((error) => {
      for (const root of roots) {
        const status = root.querySelector(".nb-code-status") ?? root.appendChild(document.createElement("p"));
        status.className = "nb-code-status";
        status.textContent = `${error.message}。可以在 GitHub 打开这些文件。`;
      }
    });
  }

  // ---------------------------------------------------------------- The 源码 page

  function mountSourcePage() {
    const panelRoot = document.querySelector("#source-panel");
    const tree = document.querySelector("#source-tree");
    if (!panelRoot || !tree || !window.SourcePanel) return;

    const params = new URL(location.href).searchParams;
    const requestedPath = params.get("path")?.replace(/^\/+/, "") || null;
    const requestedDir = (params.get("dir") ?? "").replace(/^\/+|\/+$/g, "");
    const crumbs = document.querySelector("#source-crumbs");
    const title = document.querySelector("#source-title") ?? document.querySelector(".md-content h1");
    const note = document.querySelector("#source-note");
    const siblings = document.querySelector("#source-siblings");
    const siblingList = document.querySelector("#source-sibling-list");

    function setTitle(text) {
      if (!title) return;
      title.textContent = text;
      document.title = `${text} · 源码 · LearnLLM`;
    }

    function renderCrumbs(path) {
      crumbs.replaceChildren(link("", "源码", true));
      const parts = path.split("/").filter(Boolean);
      parts.forEach((part, index) => {
        crumbs.append(" / ");
        const isLast = index === parts.length - 1;
        if (isLast && requestedPath) {
          const span = document.createElement("span");
          span.textContent = part;
          crumbs.append(span);
        } else {
          crumbs.append(link(parts.slice(0, index + 1).join("/"), part, true));
        }
      });
      crumbs.hidden = false;
    }

    function renderTree(index, dir) {
      const prefix = dir ? `${dir}/` : "";
      const children = new Map();
      for (const file of index.files) {
        if (!file.path.startsWith(prefix)) continue;
        const rest = file.path.slice(prefix.length);
        const slash = rest.indexOf("/");
        const name = slash === -1 ? rest : rest.slice(0, slash);
        const entry = children.get(name) ?? { name, directory: slash !== -1, files: 0, bytes: 0, lines: 0 };
        entry.files += 1;
        entry.bytes += file.bytes;
        entry.lines += file.lines;
        children.set(name, entry);
      }
      tree.hidden = false;
      if (!children.size) {
        tree.textContent = "没有这个目录。";
        return;
      }
      const list = document.createElement("ol");
      list.className = "llm-source-rows";
      const sorted = [...children.values()].sort((a, b) => Number(b.directory) - Number(a.directory) || a.name.localeCompare(b.name));
      for (const entry of sorted) {
        const row = document.createElement("li");
        row.className = "nb-entry llm-source-row";
        const target = prefix + entry.name;
        const name = link(target, entry.name + (entry.directory ? "/" : ""), entry.directory);
        const meta = document.createElement("span");
        meta.textContent = entry.directory
          ? `${entry.files} 个文件 · ${entry.lines} 行`
          : `${entry.lines} 行 · ${(entry.bytes / 1024).toFixed(1)} KB`;
        row.append(name, meta);
        list.append(row);
      }
      tree.replaceChildren(list);
    }

    function renderFile(index, path) {
      const file = index.files.find((item) => item.path === path);
      if (!file) {
        setTitle("没有这个文件");
        note.textContent = `${path} 不在站内的源码包里，可能不是课程用到的文件。`;
        renderCrumbs("");
        renderTree(index, "");
        return;
      }
      const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
      renderCrumbs(path);
      setTitle(window.SourcePanel.fileName(path));
      note.textContent = `${path} · ${file.lines} 行 · 仓库提交 ${index.revision.slice(0, 7)}`;
      tree.hidden = true;
      panelRoot.hidden = false;
      const panel = window.SourcePanel.create({ root: panelRoot, base: bundleBase, files: [{ path }], active: path, blobBase: index.blobBase });
      const range = window.SourcePanel.parseRange(location.hash);
      if (range) {
        panel.mark([range]);
        panel.load(path).then(() => panel.scrollToLine(range[0])).catch(() => {});
      }
      const others = index.files.filter((item) =>
        item.path !== path
        && item.path.startsWith(dir ? `${dir}/` : "")
        && !item.path.slice(dir ? dir.length + 1 : 0).includes("/"));
      if (others.length) {
        siblings.hidden = false;
        siblingList.replaceChildren(...others.map((item) => {
          const row = document.createElement("li");
          row.append(link(item.path, window.SourcePanel.fileName(item.path)));
          return row;
        }));
      }
    }

    loadIndex().then((index) => {
      if (requestedPath) {
        renderFile(index, requestedPath);
      } else {
        renderCrumbs(requestedDir ? `${requestedDir}/` : "");
        if (requestedDir) {
          setTitle(`${requestedDir.split("/").pop()}/`);
          note.textContent = `${requestedDir} · 仓库提交 ${index.revision.slice(0, 7)}`;
        } else {
          note.textContent = `${index.files.length} 个文件 · 仓库提交 ${index.revision.slice(0, 7)} · 和这个站点出自同一个提交。从章节里的面板或下面的目录进入。`;
        }
        renderTree(index, requestedDir);
      }
    }).catch((error) => {
      setTitle("源码暂时读不到");
      note.textContent = error.message;
    });
  }

  // Material mounts its own copy button on every pre > code it finds at load; the panel's pre is created after
  // that pass, so the page keeps the panel's own 复制 word.
  function mount() {
    mountChapterPanels();
    mountSourcePage();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
