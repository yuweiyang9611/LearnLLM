// Runs one experiment script in the browser on Pyodide, in a Web Worker so the page never freezes.
// window.PythonRunner.run({ files, entry, onStage, onOutput }) → { done, stop }: files are {path, text} pairs
// written under the project root in the worker, entry is the script's path among them. Stages and their
// timeouts: 加载 Python 运行时 60 s (the first click downloads the runtime from jsDelivr), 准备文件 10 s,
// 执行 20 s; past a timeout the worker is terminated and the result says 超时. The worker is kept between
// runs on a page, so the second run skips the download; a stopped or timed-out worker is replaced.
(() => {
  "use strict";

  const workerUrl = new URL("python-worker.js", document.currentScript.src);
  const stages = { load: [60000, "加载 Python 运行时"], prepare: [10000, "准备文件"], execute: [20000, "执行"] };
  const OUTPUT_LIMIT = 64 * 1024;
  let shared = null;

  function spawn() {
    shared ??= new Worker(workerUrl, { type: "module" });
    return shared;
  }

  function discard(worker) {
    worker.terminate();
    if (shared === worker) shared = null;
  }

  function run({ files, entry, onStage, onOutput }) {
    let worker = null;
    let timer = 0;
    let settled = false;
    let output = "";
    let settle = null;
    const done = new Promise((resolve) => { settle = resolve; });

    function finish(result, { keep = true } = {}) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (worker) {
        worker.onmessage = null;
        worker.onerror = null;
        if (!keep) discard(worker);
      }
      settle({ output, ...result });
    }

    function stage(name) {
      clearTimeout(timer);
      const [timeout, label] = stages[name];
      onStage?.(name, label);
      timer = setTimeout(() => finish({ ok: false, timedOut: true, stage: name, label, error: `${label}超时（${timeout / 1000} 秒），已停止。` }, { keep: false }), timeout);
    }

    try {
      worker = spawn();
      worker.onerror = (event) => {
        event.preventDefault();
        finish({ ok: false, stage: "load", error: `运行失败：${event.message || "无法启动 Python 运行时"}` }, { keep: false });
      };
      worker.onmessage = ({ data }) => {
        if (settled) return;
        if (data.stage === "load" || data.stage === "prepare" || data.stage === "execute") { stage(data.stage); return; }
        if (data.stage === "output") {
          const room = OUTPUT_LIMIT - output.length;
          if (room > 0) {
            output += data.text.slice(0, room);
            onOutput?.(output);
          }
          return;
        }
        if (data.stage !== "done") return;
        output = data.output.slice(0, OUTPUT_LIMIT) + (data.truncated ? "\n[输出已达到 64 KB，后续输出已截断]" : "");
        if (data.ok) finish({ ok: true, stage: "execute" });
        else finish({ ok: false, stage: "execute", error: data.error });
      };
      stage("load");
      worker.postMessage({ type: "run", files, entry });
    } catch (error) {
      finish({ ok: false, stage: "load", error: `无法启动运行环境：${error.message}` }, { keep: false });
    }

    return {
      done,
      stop: () => finish({ ok: false, stopped: true, stage: "stopped", error: "已停止。" }, { keep: false }),
    };
  }

  window.PythonRunner = Object.freeze({ run, stages });
})();
