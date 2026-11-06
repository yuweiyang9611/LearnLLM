// Runs one experiment script on Pyodide, off the page: the runner posts the files to write and the script to
// run, this worker loads the runtime from jsDelivr on first use, writes the files into the Pyodide filesystem
// at /home/pyodide/LearnLLM/<path> and runs the script as __main__ with experiments/ and src/ on sys.path,
// streaming stdout and stderr back as {stage:"output"} messages. The page terminates the worker on timeout.
// It is a module worker: Pyodide 314 ships as an ES module and refuses classic workers.

// Pinned: the release is dated 2026-09-14 on npm; a newer release is taken on purpose, not by drift.
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const ROOT = "/home/pyodide/LearnLLM";
const OUTPUT_LIMIT = 64 * 1024;

let runtimePromise = null;
let output = "";
let truncated = false;

// Output is kept to 64 KB: past it the rest is dropped and the result says so.
function emit(text) {
  if (truncated) return;
  const room = OUTPUT_LIMIT - output.length;
  const piece = text.length > room ? text.slice(0, Math.max(room, 0)) : text;
  if (text.length > room) truncated = true;
  if (!piece) return;
  output += piece;
  self.postMessage({ stage: "output", text: piece });
}

async function runtime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const { loadPyodide } = await import(`${PYODIDE_BASE}pyodide.mjs`);
      const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });
      pyodide.setStdout({ batched: (line) => emit(`${line}\n`) });
      pyodide.setStderr({ batched: (line) => emit(`${line}\n`) });
      return pyodide;
    })();
    runtimePromise.catch(() => { runtimePromise = null; });
  }
  return runtimePromise;
}

const RUN_SCRIPT = `
import runpy
import sys

root = ${JSON.stringify(ROOT)}
# Every run starts from the files just written: drop the course modules a previous run imported.
for name in list(sys.modules):
    if name == "_common" or name == "learn_llm" or name.startswith("learn_llm."):
        del sys.modules[name]
for folder in (root + "/src", root + "/experiments"):
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)
sys.argv = [entry]
runpy.run_path(entry, run_name="__main__")
`;

self.onmessage = async ({ data }) => {
  if (data?.type !== "run") return;
  output = "";
  truncated = false;
  try {
    self.postMessage({ stage: "load" });
    const pyodide = await runtime();
    self.postMessage({ stage: "prepare" });
    for (const file of data.files) {
      const path = `${ROOT}/${file.path}`;
      pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      pyodide.FS.writeFile(path, file.text, { encoding: "utf8" });
    }
    self.postMessage({ stage: "execute" });
    const globals = pyodide.toPy({ entry: `${ROOT}/${data.entry}` });
    try {
      await pyodide.runPythonAsync(RUN_SCRIPT, { globals });
    } finally {
      globals.destroy();
    }
    self.postMessage({ stage: "done", ok: true, output, truncated });
  } catch (error) {
    self.postMessage({ stage: "done", ok: false, output, truncated, error: String(error?.message ?? error) });
  }
};
