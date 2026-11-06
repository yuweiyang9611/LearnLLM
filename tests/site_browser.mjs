// Browser checks for source read in place, against a built site/ (run scripts/bundle_source.py and
// `mkdocs build --strict` first). Needs Node and playwright-core, which this repository does not install:
//   NODE_PATH=<a node_modules with playwright-core> node tests/site_browser.mjs [--screenshots <dir>]
// It serves site/ under /LearnLLM/ as GitHub Pages does, opens a chapter and the 源码 page in Chromium
// (Edge or Chrome from the usual paths, or PLAYWRIGHT_BROWSER), and checks: the panel shows the file with
// its line numbers and colours, the 源码 page shows a file and a directory, no page links GitHub for a
// bundled file, nothing scrolls sideways at 390px, and the tabs still render without script. Then the loop:
// the band under a chapter's title follows the record (read, 看代码, 跑, 自测, 记一笔), experiments 03, 09 and 11
// run on Pyodide (downloaded from jsDelivr on the first 运行, so allow a minute), a torch script says 需要本机,
// the self-check and the note are recorded, the opening screen continues at the earliest 停在这里 chapter, and
// 备份 / 恢复 round-trip the record. Screenshots go to --screenshots <dir> when given.
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
  // data/ joined the bundle for the browser runs (the corpus and the RAG documents the experiments read).
  assert.equal(await page.locator(".llm-source-row").count(), 4, "根目录应列出 data、examples、experiments、src");
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

  // ---------------------------------------------------------------- Run in the browser and the loop band

  const chapterUrl = (name) => `${baseUrl}${encodeURIComponent(name)}/`;
  const chapter00 = chapterUrl("00_学习路线与环境");
  const chapter02 = chapterUrl("02_Tokenizer与语言模型");
  const chapter07 = chapterUrl("07_评测_RAG与Agent");
  const bandStates = (page) => page.locator("ol.llm-steps li").evaluateAll((items) => items.map((item) => item.dataset.state));
  const outputDone = (page, pattern, timeout) => page.waitForFunction(
    (source) => new RegExp(source, "u").test(document.querySelector(".llm-source .nb-code-output pre")?.textContent ?? "") && document.querySelector(".llm-source .nb-code-output")?.dataset.state === "done",
    pattern,
    { timeout },
  );

  // Without script the band renders with every step pending (none where the chapter lacks the step).
  const plain = await browser.newContext({ viewport: { width: 1280, height: 900 }, javaScriptEnabled: false });
  const plainChapter = await plain.newPage();
  await plainChapter.goto(chapter02);
  assert.deepEqual(await plainChapter.locator("ol.llm-steps li").evaluateAll((items) => items.map((item) => `${item.dataset.step}:${item.dataset.state}`)), ["read:pending", "code:pending", "run:pending", "quiz:pending", "note:pending"], "无脚本时带子全是待做");
  await plainChapter.goto(chapter00);
  assert.deepEqual(await plainChapter.locator("ol.llm-steps li").evaluateAll((items) => items.map((item) => item.dataset.state)), ["pending", "pending", "none", "pending", "pending"], "00 章没有能在浏览器跑的脚本");
  await plain.close();

  const loop = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const loopPage = await loop.newPage();
  const loopErrors = [];
  loopPage.on("pageerror", (error) => loopErrors.push(error.message));
  await loopPage.goto(chapter02);
  await loopPage.locator(".llm-source .nb-code-line").first().waitFor();
  assert.equal(await loopPage.locator("ol.llm-steps li").count(), 5, "带子有五步");
  assert.deepEqual(await bandStates(loopPage), ["done", "now", "pending", "pending", "pending"], "打开页面：读完成，看代码停在这里");
  // The first tab, 02_tokenization.py, needs torch: 需要本机 with the command and no 运行 word.
  await loopPage.locator(".llm-source .llm-run-local").waitFor();
  assert.match(await loopPage.locator(".llm-source .llm-run-local").textContent(), /需要本机：python experiments\/02_tokenization\.py/u);
  assert.ok(await loopPage.locator('.llm-source [data-run="run"]').isHidden(), "02 没有运行键");
  await loopPage.locator(".llm-source .nb-code-tab").nth(1).click();
  await loopPage.waitForFunction(() => document.querySelector('.llm-source .nb-code-tab[aria-selected="true"]')?.textContent === "03_bigram_language_model.py");
  assert.deepEqual(await bandStates(loopPage), ["done", "done", "now", "pending", "pending"], "点了页签：看代码完成，跑停在这里");
  assert.ok(await loopPage.locator('.llm-source [data-run="run"]').isVisible(), "03 有运行键");
  assert.ok(await loopPage.locator(".llm-source .llm-run-local").isHidden(), "03 不写需要本机");
  assert.match(await loopPage.locator(".llm-source .llm-run-note").textContent(), /_common\.py 和 learn_llm\/__init__\.py 换成不含 torch 的版本；其余逐字相同/u);
  // Experiment 03 on Pyodide: the first click downloads the runtime from jsDelivr, so allow the load timeout.
  await loopPage.locator('.llm-source [data-run="run"]').click();
  await outputDone(loopPage, "PASS: 利用一个 token 的上下文，模型优于均匀随机猜测。", 150000);
  const output03 = await loopPage.locator(".llm-source .nb-code-output pre").textContent();
  assert.match(output03, /词表大小: \d+/u);
  assert.match(output03, /Bigram 训练语料困惑度: \d+\.\d+/u);
  assert.match(await loopPage.locator(".llm-source .nb-code-output-label").textContent(), /^运行结果 · 浏览器内 Python$/u);
  assert.deepEqual(await bandStates(loopPage), ["done", "done", "done", "now", "pending"], "跑完了：自测停在这里");
  await loopPage.evaluate(() => window.scrollTo(0, 0));
  await shoot(loopPage, "chapter-02-band-day");
  await loopPage.locator(".llm-source .nb-code-output").scrollIntoViewIfNeeded();
  await loopPage.evaluate(() => window.scrollBy(0, 160));
  await shoot(loopPage, "chapter-02-run-day");
  await loopPage.evaluate(() => { document.documentElement.dataset.theme = "dark"; });
  await loopPage.waitForFunction(() => document.body.dataset.mdColorScheme === "slate");
  await shoot(loopPage, "chapter-02-run-night");
  await loopPage.evaluate(() => window.scrollTo(0, 0));
  await shoot(loopPage, "chapter-02-band-night");
  await loopPage.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  // The self-check: the three questions from the bank; the band moves on when all are right.
  const answers = await loopPage.evaluate(() => window.LlmQuizBank.questionsFor("02").map((question) => [question.id, question.answer]));
  assert.equal(answers.length, 3, "每章三题");
  const first = loopPage.locator(`.llm-quiz-item[data-quiz-id="${answers[0][0]}"]`);
  await first.locator(`input[value="${(answers[0][1] + 1) % 4}"]`).check();
  await first.locator('button[type="submit"]').click();
  assert.equal(await first.getAttribute("data-state"), "wrong", "答错有说明");
  assert.match(await first.locator(".llm-quiz-feedback").textContent(), /^更合适的是：/u);
  await first.locator(".llm-quiz-retry").click();
  for (const [id, answer] of answers) {
    const item = loopPage.locator(`.llm-quiz-item[data-quiz-id="${id}"]`);
    await item.locator(`input[value="${answer}"]`).check();
    await item.locator('button[type="submit"]').click();
    assert.equal(await item.getAttribute("data-state"), "done", id);
  }
  assert.match(await loopPage.locator(".llm-quiz-summary").textContent(), /^3 \/ 3 题答对/u);
  assert.deepEqual(await bandStates(loopPage), ["done", "done", "done", "done", "now"], "自测通过：记一笔停在这里");
  // A note saved: the chapter is done.
  await loopPage.locator("#llm-note").fill("Bigram 只看前一个 token；浏览器里的 PPL 和本机一致。");
  await loopPage.locator("#llm-note-save").click();
  assert.match(await loopPage.locator(".llm-notepad-status").textContent(), /^已保存 · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/u);
  assert.deepEqual(await bandStates(loopPage), ["done", "done", "done", "done", "done"], "五步都完成");
  const record = await loopPage.evaluate(() => JSON.parse(localStorage.getItem("learn-llm-progress-v1")));
  assert.equal(record.version, 1);
  for (const field of ["readAt", "codeAt", "ranAt", "quizPassedAt", "quizRevision", "note"]) assert.ok(record.chapters["02"][field], `记录里有 ${field}`);
  // 改一改再跑: the edited copy runs, the label says so, and 恢复原文 brings the file back.
  await loopPage.locator('.llm-source [data-run="edit"]').click();
  await loopPage.locator(".llm-source .nb-code-editor").fill('print("改过的副本")\n');
  await loopPage.locator('.llm-source [data-run="run"]').click();
  await outputDone(loopPage, "改过的副本", 60000);
  assert.match(await loopPage.locator(".llm-source .nb-code-output-label").textContent(), /已改动的副本$/u);
  await loopPage.locator('.llm-source [data-run="reset"]').click();
  assert.ok(await loopPage.locator(".llm-source .nb-code-editor").isHidden(), "恢复原文后编辑框收起");
  // Chapter 07: experiments 09 and 11 both run.
  await loopPage.goto(chapter07);
  await loopPage.locator(".llm-source .nb-code-line").first().waitFor();
  await loopPage.locator('.llm-source [data-run="run"]').click();
  await outputDone(loopPage, "PASS: Tiny RAG 能引用来源、拒绝资料外问题", 150000);
  assert.match(await loopPage.locator(".llm-source .nb-code-output pre").textContent(), /端到端成功率:\s+100\.0%/u);
  await loopPage.locator(".llm-source .nb-code-tab").nth(1).click();
  await loopPage.waitForFunction(() => document.querySelector('.llm-source .nb-code-tab[aria-selected="true"]')?.textContent === "11_local_agent.py");
  await loopPage.locator('.llm-source [data-run="run"]').click();
  await outputDone(loopPage, "PASS: Agent 的工具、状态、错误和停止条件均由本地程序显式控制。", 60000);
  assert.match(await loopPage.locator(".llm-source .nb-code-output pre").textContent(), /最终回答: 62/u);
  assert.deepEqual(await bandStates(loopPage), ["done", "done", "done", "now", "pending"], "07 跑完，自测停在这里");
  // Chapter 00: its script needs torch, so the panel says 需要本机 and the run step is dashed.
  await loopPage.goto(chapter00);
  await loopPage.locator(".llm-source .llm-run-local").waitFor();
  assert.match(await loopPage.locator(".llm-source .llm-run-local").textContent(), /需要本机：python experiments\/00_environment_check\.py/u);
  assert.deepEqual(await bandStates(loopPage), ["done", "now", "none", "pending", "pending"]);
  // The opening screen: continue at the earliest chapter with a 停在这里 step (00 at 看代码), cards marked.
  await loopPage.goto(baseUrl);
  await loopPage.waitForFunction(() => document.querySelector(".llm-continue")?.textContent === "接着做");
  assert.match(await loopPage.locator(".llm-start .nb-banner-title").textContent(), /^00 /u);
  assert.match(await loopPage.locator(".llm-start .nb-banner-note").textContent(), /停在「看代码」· 第 2 步，共 5 步/u);
  assert.ok((await loopPage.locator(".llm-continue").getAttribute("href")).endsWith("#llm-source"), "接着做指向面板");
  assert.equal(await loopPage.locator('.llm-cards > .nb-entry[data-record="done"]').count(), 1, "02 章完成");
  assert.equal(await loopPage.locator('.llm-cards > .nb-entry[data-record="now"] .nb-card-eyebrow').textContent(), "开始学习 · 第 00 章");
  assert.deepEqual(await loopPage.locator(".llm-card-state").evaluateAll((chips) => chips.map((chip) => chip.textContent)), ["停在「看代码」", "完成", "停在「自测」"]);
  await shoot(loopPage, "index-continue-day");
  await loopPage.evaluate(() => { document.documentElement.dataset.theme = "dark"; });
  await loopPage.waitForFunction(() => document.body.dataset.mdColorScheme === "slate");
  await shoot(loopPage, "index-continue-night");
  await loopPage.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  // 备份 downloads the record; 恢复 with chapter 00 finished moves the banner to 01.
  const [download] = await Promise.all([loopPage.waitForEvent("download"), loopPage.locator("#llm-export").click()]);
  assert.equal(download.suggestedFilename(), "learn-llm-progress.json");
  const backup = JSON.parse(readFileSync(await download.path(), "utf8"));
  assert.equal(backup.format, "learn-llm-progress");
  assert.ok(backup.progress.chapters["02"].ranAt && backup.progress.chapters["07"].ranAt, "备份带着两章的运行记录");
  const restored = structuredClone(backup);
  const revision00 = await loopPage.evaluate(() => window.LlmQuizBank.revision("00"));
  restored.progress.chapters["00"] = { readAt: "2026-11-06T00:00:00.000Z", codeAt: "2026-11-06T00:00:00.000Z", quizPassedAt: "2026-11-06T00:00:00.000Z", quizRevision: revision00, note: "环境检查通过。" };
  await loopPage.locator("#llm-import").setInputFiles({ name: "backup.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(restored)) });
  await loopPage.waitForFunction(() => /^01 /u.test(document.querySelector(".llm-start .nb-banner-title")?.textContent ?? ""));
  assert.match(await loopPage.locator(".llm-backup-status").textContent(), /^已恢复 \d+ 章的记录/u);
  assert.equal(await loopPage.locator('.llm-cards > .nb-entry[data-record="done"]').count(), 2, "00 和 02 完成");
  await loopPage.locator("#llm-import").setInputFiles({ name: "other.json", mimeType: "application/json", buffer: Buffer.from('{"format":"something-else"}') });
  await loopPage.waitForFunction(() => document.querySelector(".llm-backup-status")?.textContent === "这不是本站的备份文件。");
  assert.deepEqual(loopErrors, [], "页面不应有脚本错误");
  await loop.close();

  // 390px: the band wraps onto more than one row and nothing scrolls sideways; the quiz and the note fit too.
  const loopMobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const loopMobilePage = await loopMobile.newPage();
  await loopMobilePage.goto(chapter02);
  await loopMobilePage.locator(".llm-source .nb-code-line").first().waitFor();
  const bandBox = await loopMobilePage.locator("ol.llm-steps").evaluate((band) => ({ height: band.getBoundingClientRect().height, item: band.firstElementChild.getBoundingClientRect().height, scrollWidth: band.scrollWidth, clientWidth: band.clientWidth }));
  assert.ok(bandBox.height > bandBox.item * 1.5, `带子应换行：${JSON.stringify(bandBox)}`);
  assert.ok(bandBox.scrollWidth <= bandBox.clientWidth + 1, `带子不横向溢出：${JSON.stringify(bandBox)}`);
  await assertNoDocumentOverflow(loopMobilePage, "移动端第 02 章（带子）");
  await loopMobilePage.evaluate(() => window.scrollTo(0, 0));
  await shoot(loopMobilePage, "chapter-02-band-390");
  await loopMobilePage.locator("#llm-quiz").scrollIntoViewIfNeeded();
  await assertNoDocumentOverflow(loopMobilePage, "移动端自测");
  await loopMobilePage.locator("#llm-notepad").scrollIntoViewIfNeeded();
  await assertNoDocumentOverflow(loopMobilePage, "移动端记一笔");
  await loopMobilePage.goto(baseUrl);
  await loopMobilePage.locator("#llm-backup-panel").waitFor();
  await assertNoDocumentOverflow(loopMobilePage, "移动端首页");
  await shoot(loopMobilePage, "index-continue-390");
  await loopMobile.close();

  console.log(`PASS: source read in place, the browser runs and the loop band checked against ${baseUrl}`);
} finally {
  await browser.close();
  server.close();
}
