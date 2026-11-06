// Run in the browser: the chapter panel gets 运行, 改一改再跑, 恢复原文 and 停止 for the scripts that run on
// Pyodide (python-runner.js), with the output under the actions row; a script that needs torch says 需要本机
// with its local command instead. The panel's data-run attribute names the runnable files; RUNS lists what each
// one reads from the bundle. A finished run sets the chapter's ranAt in the learning record.
(() => {
  "use strict";

  // The files a script needs beside itself, by bundle path: the data it reads and the pure modules it imports.
  const RUNS = {
    "experiments/03_bigram_language_model.py": ["data/tiny_corpus.txt"],
    "experiments/09_tiny_rag.py": ["data/rag_documents.json", "src/learn_llm/rag.py"],
    "experiments/11_local_agent.py": ["data/rag_documents.json", "src/learn_llm/rag.py", "src/learn_llm/agent.py"],
  };
  // The two files the browser swaps for versions without torch; everything else is the bundle's copy, byte for byte.
  const BROWSER_FILES = [
    { path: "experiments/_common.py", url: "assets/browser/_common.py" },
    { path: "src/learn_llm/__init__.py", url: "assets/browser/learn_llm/__init__.py" },
  ];
  const DIFFERENCE = "浏览器里 _common.py 和 learn_llm/__init__.py 换成不含 torch 的版本；其余逐字相同。";

  const siteBase = (document.querySelector('meta[name="llm-base"]')?.content || ".").replace(/\/*$/, "/");
  const chapter = document.querySelector('meta[name="llm-chapter"]')?.content ?? null;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(id, text, solid) {
    const node = element("button", solid ? "nb-button nb-button--solid" : "nb-button", text);
    node.type = "button";
    node.dataset.run = id;
    return node;
  }

  async function fetchText(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`无法读取 ${url}（${response.status}）`);
    return (await response.text()).replace(/\r\n/g, "\n");
  }

  function attach(root, panel, bundleBase) {
    if (root.dataset.llmRun === "ready" || !window.PythonRunner) return;
    root.dataset.llmRun = "ready";
    const runnable = new Set((root.dataset.run ?? "").split(/\s+/).filter((path) => RUNS[path]));

    const runButton = button("run", "运行", true);
    const editButton = button("edit", "改一改再跑", false);
    const resetButton = button("reset", "恢复原文", false);
    const stopButton = button("stop", "停止", false);
    resetButton.hidden = true;
    stopButton.hidden = true;
    panel.actions.prepend(runButton, editButton, resetButton, stopButton);

    const note = element("p", "llm-run-note", DIFFERENCE);
    const local = element("p", "llm-run-local");
    const output = element("div", "nb-code-output");
    output.hidden = true;
    const outputLabel = element("p", "nb-code-output-label");
    const outputText = element("pre");
    outputText.tabIndex = 0;
    output.append(outputLabel, outputText);
    root.append(note, local, output);

    const editors = new Map();
    const originals = new Map();
    let overridePath = null;
    let handle = null;

    function editor(path) {
      if (!editors.has(path)) {
        const area = element("textarea", "nb-code-editor");
        area.spellcheck = false;
        area.setAttribute("aria-label", `${window.SourcePanel.fileName(path)} 的可编辑副本`);
        area.hidden = true;
        panel.view.after(area);
        editors.set(path, area);
      }
      return editors.get(path);
    }

    function isEditing(path) {
      return editors.has(path) && editors.get(path).dataset.editing === "true";
    }

    function setOverride(path) {
      const next = path && isEditing(path) ? path : null;
      if (next === overridePath) return;
      overridePath = next;
      panel.setOverride(next ? { path: next, element: editor(next), value: () => editor(next).value } : null);
    }

    function localCommand(path) {
      return `python ${path}`;
    }

    // The controls follow the selected tab: a runnable file shows the run words, the rest say 需要本机.
    function refresh() {
      const path = panel.current();
      const canRun = runnable.has(path);
      runButton.hidden = !canRun;
      editButton.hidden = !canRun || isEditing(path);
      resetButton.hidden = !canRun || !isEditing(path);
      note.hidden = !canRun;
      local.hidden = canRun;
      if (!canRun && path) {
        local.replaceChildren("需要本机：", element("code", null, localCommand(path)), "（这个脚本要 PyTorch，浏览器里跑不了）");
      }
      setOverride(path);
    }

    function showOutput(state, label, text) {
      output.hidden = false;
      output.dataset.state = state;
      outputLabel.textContent = label;
      outputText.textContent = text;
    }

    async function collect(path) {
      const files = [];
      for (const dependency of RUNS[path]) files.push({ path: dependency, text: await fetchText(`${bundleBase}${dependency}.txt`) });
      for (const file of BROWSER_FILES) files.push({ path: file.path, text: await fetchText(`${siteBase}${file.url}`) });
      files.push({ path, text: isEditing(path) ? editor(path).value : await panel.load(path) });
      return files;
    }

    async function runScript() {
      const path = panel.current();
      if (handle || !runnable.has(path)) return;
      const edited = isEditing(path);
      runButton.disabled = true;
      stopButton.hidden = false;
      showOutput("running", "运行中", "");
      let files;
      try {
        files = await collect(path);
      } catch (error) {
        runButton.disabled = false;
        stopButton.hidden = true;
        showOutput("wrong", "运行失败", error.message);
        return;
      }
      handle = window.PythonRunner.run({
        files,
        entry: path,
        onStage: (name, label) => { panel.status.textContent = `${label}中…`; outputLabel.textContent = `${label}中`; },
        onOutput: (text) => { outputText.textContent = text; },
      });
      const result = await handle.done;
      handle = null;
      runButton.disabled = false;
      stopButton.hidden = true;
      if (result.ok) {
        showOutput("done", `运行结果 · 浏览器内 Python${edited ? " · 已改动的副本" : ""}`, result.output.trimEnd() || "（没有输出）");
        panel.status.textContent = "运行完成。对照章节里写的预期输出。";
        if (chapter) window.LearnProgress?.mark(chapter, "ranAt");
      } else {
        const label = result.timedOut ? "运行超时" : result.stopped ? "已停止" : "运行失败";
        showOutput("wrong", label, [result.output.trimEnd(), result.error].filter(Boolean).join("\n"));
        panel.status.textContent = result.timedOut ? "超时了。可以改代码后重新运行。" : result.stopped ? "已停止。" : "运行失败，看输出里的回溯。";
      }
    }

    editButton.addEventListener("click", async () => {
      const path = panel.current();
      if (!runnable.has(path)) return;
      if (!originals.has(path)) originals.set(path, await panel.load(path));
      const area = editor(path);
      if (area.dataset.editing !== "true") {
        area.value = originals.get(path);
        area.dataset.editing = "true";
      }
      refresh();
      area.focus();
      panel.status.textContent = "这是副本，改完点运行；原文件不会变。";
    });

    resetButton.addEventListener("click", () => {
      const path = panel.current();
      const area = editors.get(path);
      if (area) {
        area.value = originals.get(path) ?? "";
        area.dataset.editing = "false";
      }
      refresh();
      panel.status.textContent = "已恢复原文。";
    });

    runButton.addEventListener("click", runScript);
    stopButton.addEventListener("click", () => handle?.stop());
    window.addEventListener("pagehide", () => handle?.stop());

    // Opening a file counts as looking at the code: a tab click, or a click into the code itself.
    const tabs = root.querySelector(".nb-code-tabs");
    tabs?.addEventListener("click", (event) => {
      if (!event.target.closest(".nb-code-tab")) return;
      if (chapter) window.LearnProgress?.mark(chapter, "codeAt");
      refresh();
    });
    panel.view.addEventListener("click", () => { if (chapter) window.LearnProgress?.mark(chapter, "codeAt"); });

    refresh();
    root.llmRun = Object.freeze({ run: runScript, refresh, isEditing: () => isEditing(panel.current()) });
  }

  document.addEventListener("llm-source-panel", (event) => {
    attach(event.target, event.detail.panel, event.detail.bundleBase);
  });
  for (const root of document.querySelectorAll(".llm-source[data-source]")) {
    if (root.llmPanel) attach(root, root.llmPanel, `${siteBase}assets/source/`);
  }

  window.LlmRun = Object.freeze({ runs: RUNS, browserFiles: BROWSER_FILES, difference: DIFFERENCE });
})();
