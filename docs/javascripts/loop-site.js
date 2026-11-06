// The loop on every chapter and the opening screen, read from the learning record (progress.js):
// - the band under a chapter's title (ol.nb-steps.llm-steps, rendered by overrides/main.html with every step
//   pending) takes its states from the record; opening the page sets readAt;
// - the 记一笔 textarea at the end of a chapter saves the chapter's note;
// - the opening screen's banner moves to the earliest chapter with a 停在这里 step, the cards show where each
//   chapter stands, and the 备份 / 恢复 words export and import the record as JSON.
(() => {
  "use strict";

  const progress = window.LearnProgress;
  if (!progress) return;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const STEP_ANCHORS = { read: "", code: "#llm-source", run: "#llm-source", quiz: "#llm-quiz", note: "#llm-notepad" };

  function formatTime(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (value) => String(value).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  // ---------------------------------------------------------------- The band on a chapter

  function mountBand() {
    const band = document.querySelector("ol.llm-steps[data-chapter]");
    if (!band) return;
    const chapter = band.dataset.chapter;
    function render() {
      for (const item of progress.steps(chapter)) {
        const node = band.querySelector(`li[data-step="${item.step}"]`);
        if (node) node.dataset.state = item.state;
      }
    }
    progress.mark(chapter, "readAt");
    render();
    progress.subscribe(render);
  }

  // ---------------------------------------------------------------- 记一笔

  function mountNotepad() {
    const marker = document.querySelector(".llm-notepad[data-chapter]");
    if (!marker) return;
    const chapter = marker.dataset.chapter;
    const heading = element("h2", "llm-notepad-title", "记一笔");
    heading.id = "llm-notepad-heading";
    const lead = element("p", "llm-notepad-note", "这一章记下的东西只存在这台浏览器里；首页可以把全部记录备份成 JSON。");
    const area = document.createElement("textarea");
    area.className = "llm-notepad-text";
    area.id = "llm-note";
    area.rows = 6;
    area.spellcheck = false;
    area.setAttribute("aria-labelledby", "llm-notepad-heading");
    area.placeholder = "运行前的预测、实际输出、和预测不同的原因、只改了哪一个变量……";
    const actions = element("div", "llm-notepad-actions");
    const save = element("button", "nb-button nb-button--solid", "保存");
    save.type = "button";
    save.id = "llm-note-save";
    const status = element("p", "llm-notepad-status");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    actions.append(save, status);
    marker.replaceChildren(heading, lead, area, actions);

    function render() {
      const record = progress.chapter(chapter);
      if (document.activeElement !== area) area.value = record.note ?? "";
      status.textContent = record.note ? `已保存 · ${formatTime(record.noteAt)}` : "还没有记。";
    }
    save.addEventListener("click", () => {
      if (!progress.setNote(chapter, area.value)) {
        status.textContent = "太长了：一章的笔记最多 20 KB。";
        return;
      }
      render();
      if (!progress.isPersistent()) status.textContent = "这台浏览器不让保存记录（隐私模式或存储已满）。";
    });
    render();
    progress.subscribe(render);
  }

  // ---------------------------------------------------------------- The opening screen

  function mountIndex() {
    const cards = [...document.querySelectorAll(".llm-cards > .nb-entry")];
    const banner = document.querySelector(".llm-start");
    if (!cards.length || !banner) return;
    const chapters = progress.chapters;
    const byChapter = new Map(cards.map((card, index) => [chapters[index], card]));
    const eyebrow = banner.querySelector(".nb-banner-eyebrow");
    const title = banner.querySelector(".nb-banner-title");
    const note = banner.querySelector(".nb-banner-note");
    const link = banner.querySelector(".llm-continue");
    const original = { eyebrow: eyebrow?.textContent, title: title?.textContent, note: note?.innerHTML, href: link?.getAttribute("href"), text: link?.textContent };

    function cardLink(id) {
      return byChapter.get(id)?.querySelector("h3 a") ?? null;
    }

    function render() {
      const next = progress.continueAt();
      for (const [id, card] of byChapter) {
        const steps = progress.steps(id);
        const done = steps.filter((item) => item.state === "done").length;
        const now = steps.find((item) => item.state === "now");
        const complete = steps.every((item) => item.state === "done" || item.state === "none");
        let state = "";
        if (complete) state = "done";
        else if (next?.id === id) state = "now";
        if (state) card.dataset.record = state;
        else card.removeAttribute("data-record");
        const meta = card.querySelector(".nb-card-meta");
        let chip = meta?.querySelector(".llm-card-state");
        const text = complete ? "完成" : done ? `停在「${now.label}」` : "";
        if (text) {
          chip ??= meta.appendChild(element("span", "llm-card-state"));
          chip.textContent = text;
          chip.dataset.state = complete ? "done" : "now";
        } else {
          chip?.remove();
        }
      }
      if (!eyebrow || !title || !note || !link) return;
      if (!progress.hasAnyRecord()) {
        eyebrow.textContent = original.eyebrow;
        title.textContent = original.title;
        note.innerHTML = original.note;
        link.setAttribute("href", original.href);
        link.textContent = original.text;
        return;
      }
      if (!next) {
        eyebrow.textContent = "全部完成";
        title.textContent = "九章都做过一遍了";
        note.textContent = "从 00 再读一遍，或者回到结课项目把报告写完。";
        link.setAttribute("href", original.href);
        link.textContent = "再看一遍";
        return;
      }
      const anchor = cardLink(next.id);
      const index = next.steps.findIndex((item) => item.step === next.step) + 1;
      eyebrow.textContent = "接着做";
      title.textContent = anchor ? `${next.id} ${anchor.textContent}` : `第 ${next.id} 章`;
      note.textContent = `停在「${next.label}」· 第 ${index} 步，共 ${next.steps.length} 步。`;
      link.setAttribute("href", (anchor?.getAttribute("href") ?? original.href) + STEP_ANCHORS[next.step]);
      link.textContent = "接着做";
    }
    render();
    progress.subscribe(render);
  }

  // ---------------------------------------------------------------- 备份 / 恢复

  function mountBackup() {
    const root = document.querySelector("#llm-backup-panel");
    if (!root) return;
    const exportButton = root.querySelector("#llm-export");
    const importInput = root.querySelector("#llm-import");
    const importLabel = root.querySelector('label[for="llm-import"]');
    const resetButton = root.querySelector("#llm-reset");
    const status = root.querySelector(".llm-backup-status");
    for (const control of [exportButton, importInput, resetButton]) if (control) control.disabled = false;
    importLabel?.removeAttribute("aria-disabled");

    function say(text) { if (status) status.textContent = text; }

    exportButton?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(progress.exportData(), null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "learn-llm-progress.json";
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      say("已把记录备份成 learn-llm-progress.json。");
    });

    importInput?.addEventListener("change", async () => {
      const file = importInput.files?.[0];
      importInput.value = "";
      if (!file) return;
      try {
        const payload = JSON.parse(await file.text());
        if (!progress.importData(payload)) {
          say("这不是本站的备份文件。");
          return;
        }
        say(`已恢复 ${Object.keys(progress.snapshot().chapters).length} 章的记录。`);
      } catch {
        say("读不了这个文件。");
      }
    });

    resetButton?.addEventListener("click", () => {
      if (!window.confirm("清空这台浏览器里的全部学习记录？先备份再清。")) return;
      progress.reset();
      say("记录已清空。");
    });
  }

  function mount() {
    mountBand();
    mountNotepad();
    mountIndex();
    mountBackup();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
