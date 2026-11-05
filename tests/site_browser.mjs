// Browser checks for source read in place, against a built site/ (run scripts/bundle_source.py and
// `mkdocs build --strict` first). Needs Node and playwright-core, which this repository does not install:
//   NODE_PATH=<a node_modules with playwright-core> node tests/site_browser.mjs [--screenshots <dir>]
// It serves site/ under /LearnLLM/ as GitHub Pages does, opens a chapter and the 源码 page in Chromium
// (Edge or Chrome from the usual paths, or PLAYWRIGHT_BROWSER), and checks: the panel shows the file with
// its line numbers and colours, the 源码 page shows a file and a directory, no page links GitHub for a
// bundled file, nothing scrolls sideways at 390px, and the tabs still render without script.
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const siteRoot = join(repositoryRoot, "site");
const prefix = "/LearnLLM/";
const sourcePage = "%E6%BA%90%E7%A0%81/";
const screenshotDirectory = process.argv.includes("--screenshots")
  ? resolve(process.argv[process.argv.indexOf("--screenshots") + 1])
  : null;

const types = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".json": "application/json", ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".xml": "application/xml" };

function browserExecutable() {
  const candidates = [
    process.env.PLAYWRIGHT_BROWSER,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
  ].filter(Boolean);
  const found = candidates.find((path) => existsSync(path));
  if (!found) throw new Error("No Chromium executable found; set PLAYWRIGHT_BROWSER.");
  return found;
}

function serve() {
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    let path = decodeURIComponent(url.pathname);
    if (!path.startsWith(prefix)) { response.writeHead(404); response.end(); return; }
    let file = join(siteRoot, path.slice(prefix.length));
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
    if (!existsSync(file)) { response.writeHead(404); response.end("not found"); return; }
    const extension = file.slice(file.lastIndexOf(".")).toLowerCase();
    response.writeHead(200, { "content-type": types[extension] ?? "application/octet-stream" });
    response.end(readFileSync(file));
  });
  return new Promise((resolveServer) => server.listen(0, "127.0.0.1", () => resolveServer(server)));
}

async function assertNoDocumentOverflow(page, label) {
  const dimensions = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth }));
  assert.ok(dimensions.scrollWidth <= dimensions.clientWidth + 1, `${label} 横向溢出：${JSON.stringify(dimensions)}`);
}

async function shoot(page, name) {
  if (!screenshotDirectory) return;
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: join(screenshotDirectory, `${name}.png`), fullPage: false });
}

function htmlFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? htmlFiles(path) : entry.name.endsWith(".html") ? [path] : [];
  });
}

const server = await serve();
const baseUrl = `http://127.0.0.1:${server.address().port}${prefix}`;
const chapter = `${baseUrl}03_%E6%B3%A8%E6%84%8F%E5%8A%9B%E4%B8%8ETransformer/`;
const browser = await chromium.launch({ executablePath: browserExecutable(), headless: true });
try {
  const bundle = JSON.parse(readFileSync(join(siteRoot, "assets", "source", "index.json"), "utf8"));
  const bundled = new Set(bundle.files.map((file) => file.path));
  assert.ok(bundled.has("experiments/04_attention.py") && bundled.size >= 30, "源码包应带上课程文件");
  for (const file of htmlFiles(siteRoot)) {
    const html = readFileSync(file, "utf8").replace(/<noscript>[\s\S]*?<\/noscript>/gu, "");
    for (const match of html.matchAll(/href="https:\/\/github\.com\/[^"]+\/(blob|tree)\/[^/"]+\/([^"#?]+)/gu)) {
      const target = decodeURIComponent(match[2]).replace(/\/$/u, "");
      assert.ok(!bundled.has(target), `${file} 仍把 ${target} 链接到 GitHub`);
      assert.ok(![...bundled].some((path) => path.startsWith(`${target}/`)), `${file} 仍把目录 ${target} 链接到 GitHub`);
    }
  }

  // Day: the chapter's panel shows its two experiments with line numbers and colour.
  const day = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await day.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(chapter);
  const panel = page.locator(".llm-source");
  await panel.locator(".nb-code-line").first().waitFor();
  assert.equal(await panel.locator(".nb-code-tab").count(), 2, "第 03 章应有两个文件页签");
  assert.equal(await panel.locator(".nb-code-line").first().getAttribute("data-line"), "1");
  const lines = await panel.locator(".nb-code-line").count();
  assert.ok(lines >= 40, `04_attention.py 应完整显示，实际 ${lines} 行`);
  assert.ok(await panel.locator(".tok-keyword").count() > 0 && await panel.locator(".tok-call").count() > 0, "源码应着色");
  assert.equal(await panel.locator(".nb-code-line").first().locator(":scope > *").count(), 1, "每行只有一个文本 span");
  assert.match(await panel.locator(".nb-code-link").getAttribute("href"), /github\.com\/yuweiyang9611\/LearnLLM\/blob\/[0-9a-f]{40}\/experiments\/04_attention\.py$/u);
  assert.ok((await panel.locator(".llm-source-open").getAttribute("href")).includes(`${sourcePage}?path=experiments%2F04_attention.py`), "面板应链接到站内源码页");
  assert.equal(await panel.locator(".md-code__nav").count(), 0, "Material 的复制按钮不应进入面板");
  assert.match(await panel.locator(".nb-code-note").textContent(), /^\d+ 行$/u);
  await panel.locator(".nb-code-tab").nth(1).click();
  await page.waitForFunction(() => document.querySelector('.llm-source .nb-code-tab[aria-selected="true"]')?.textContent === "05_transformer_block.py" && document.querySelectorAll(".llm-source .nb-code-line").length > 10);
  assert.ok((await panel.locator(".llm-source-open").getAttribute("href")).includes("05_transformer_block.py"), "站内链接应跟着页签");
  await panel.locator(".nb-code-tab").nth(0).click();
  await page.waitForFunction(() => document.querySelector('.llm-source .nb-code-tab[aria-selected="true"]')?.textContent === "04_attention.py");
  await panel.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -120));
  await shoot(page, "chapter-03-day");

  // The 源码 page: a file with a marked range and its siblings, then a directory and the root.
  await page.goto(`${baseUrl}${sourcePage}?path=experiments%2F04_attention.py#L5-L9`);
  await page.locator("#source-panel .nb-code-line").first().waitFor();
  assert.equal(await page.locator("#source-panel .nb-code-line[data-marked]").count(), 5, "行号范围应标出 5 行");
  assert.equal(await page.locator(".md-content h1").textContent(), "04_attention.py");
  assert.ok(await page.locator("#source-sibling-list li").count() >= 10, "源码页应列出同目录文件");
  assert.ok((await page.locator("#source-crumbs").textContent()).includes("experiments"), "应有面包屑");
  assert.equal(await page.locator(".md-content h1").count(), 1);
  await shoot(page, "source-file-day");
  await page.goto(`${baseUrl}${sourcePage}?dir=examples%2Fcapstone_attention`);
  await page.locator(".llm-source-row").first().waitFor();
  assert.ok(await page.locator(".llm-source-row").count() >= 5, "目录页应列出文件");
  assert.match(await page.locator(".llm-source-row span").first().textContent(), /行/u);
  await shoot(page, "source-dir-day");
  await page.goto(`${baseUrl}${sourcePage}`);
  await page.locator(".llm-source-row").first().waitFor();
  assert.equal(await page.locator(".llm-source-row").count(), 3, "根目录应列出 examples、experiments、src");
  assert.match(await page.locator(".llm-source-row span").first().textContent(), /个文件/u);
  await page.goto(`${baseUrl}${sourcePage}?path=nope.py`);
  await page.locator(".llm-source-row").first().waitFor();
  assert.equal(await page.locator(".md-content h1").textContent(), "没有这个文件");
  assert.deepEqual(errors, [], "页面不应有脚本错误");
  await day.close();

  // Night: the same two pages with the stored scheme.
  const night = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await night.addInitScript(() => { try { localStorage.setItem("site-theme", "dark"); } catch {} });
  const nightPage = await night.newPage();
  await nightPage.goto(chapter);
  await nightPage.locator(".llm-source .nb-code-line").first().waitFor();
  assert.equal(await nightPage.evaluate(() => document.documentElement.dataset.theme), "dark");
  assert.equal(await nightPage.evaluate(() => document.body.dataset.mdColorScheme), "slate");
  await nightPage.locator(".llm-source").scrollIntoViewIfNeeded();
  await nightPage.evaluate(() => window.scrollBy(0, -120));
  await shoot(nightPage, "chapter-03-night");
  await nightPage.goto(`${baseUrl}${sourcePage}?path=experiments%2F04_attention.py#L5-L9`);
  await nightPage.locator("#source-panel .nb-code-line").first().waitFor();
  await shoot(nightPage, "source-file-night");
  await night.close();

  // 390px: the code scrolls inside the panel, never the page.
  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const mobilePage = await mobile.newPage();
  await mobilePage.goto(chapter);
  await mobilePage.locator(".llm-source .nb-code-line").first().waitFor();
  await assertNoDocumentOverflow(mobilePage, "移动端第 03 章");
  const view = await mobilePage.locator(".llm-source .nb-code-view").evaluate((element) => ({ scrollWidth: element.scrollWidth, clientWidth: element.clientWidth, overflowX: getComputedStyle(element).overflowX }));
  assert.ok(view.scrollWidth > view.clientWidth && view.overflowX === "auto", `代码应在面板内滚动：${JSON.stringify(view)}`);
  await mobilePage.locator(".llm-source").scrollIntoViewIfNeeded();
  await mobilePage.evaluate(() => window.scrollBy(0, -80));
  await shoot(mobilePage, "chapter-03-390");
  await mobilePage.goto(`${baseUrl}${sourcePage}?path=experiments%2F10_sft_tiny_gpt.py`);
  await mobilePage.locator("#source-panel .nb-code-line").first().waitFor();
  await assertNoDocumentOverflow(mobilePage, "移动端源码页");
  await shoot(mobilePage, "source-file-390");
  await mobilePage.goto(`${baseUrl}${sourcePage}?dir=src%2Flearn_llm`);
  await mobilePage.locator(".llm-source-row").first().waitFor();
  await assertNoDocumentOverflow(mobilePage, "移动端源码目录");
  await mobile.close();

  // Without script: the tabs render and the noscript links name the files on GitHub.
  const noScript = await browser.newContext({ viewport: { width: 1280, height: 900 }, javaScriptEnabled: false });
  const plainPage = await noScript.newPage();
  await plainPage.goto(chapter);
  assert.equal(await plainPage.locator(".llm-source .nb-code-tab").count(), 2, "无脚本时页签仍在");
  const fallbackLinks = await plainPage.locator(".llm-source noscript a").evaluateAll((anchors) => anchors.map((anchor) => anchor.getAttribute("href")));
  assert.deepEqual(fallbackLinks, [
    "https://github.com/yuweiyang9611/LearnLLM/blob/main/experiments/04_attention.py",
    "https://github.com/yuweiyang9611/LearnLLM/blob/main/experiments/05_transformer_block.py",
  ]);
  await noScript.close();

  console.log(`PASS: source read in place checked against ${baseUrl}`);
} finally {
  await browser.close();
  server.close();
}
