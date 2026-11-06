// The site's learning record: one entry per chapter in this browser, read by the loop band on every chapter,
// the continue banner on the opening screen and the home site. Nothing here is a tick the reader makes on a
// band; each field is set by the thing the reader did (opening the page, opening the code, a browser run, the
// self-check, a note). The record is JSON in localStorage under one key, exported and restored as a file.
(() => {
  "use strict";

  const KEY = "learn-llm-progress-v1";
  const FORMAT = "learn-llm-progress";
  const VERSION = 1;
  const NOTE_LIMIT = 20 * 1024;
  const STEPS = ["read", "code", "run", "quiz", "note"];
  const STEP_LABELS = { read: "读", code: "看代码", run: "跑", quiz: "自测", note: "记一笔" };
  const TIMES = ["readAt", "codeAt", "ranAt", "quizPassedAt"];
  // What each chapter can do on the site: a code panel (04 is a paper chapter) and a script that runs in the
  // browser (chapter 02's experiment 03, chapter 07's experiments 09 and 11; the rest need torch).
  const CHAPTERS = {
    "00": { panel: true, run: false },
    "01": { panel: true, run: false },
    "02": { panel: true, run: true },
    "03": { panel: true, run: false },
    "04": { panel: false, run: false },
    "05": { panel: true, run: false },
    "06": { panel: true, run: false },
    "07": { panel: true, run: true },
    "08": { panel: true, run: false },
  };
  const IDS = Object.keys(CHAPTERS);

  const listeners = new Set();
  let state = empty();
  let persistent = true;

  function empty() {
    return { version: VERSION, updatedAt: null, chapters: {} };
  }

  function now() {
    return new Date().toISOString();
  }

  function isoOrNull(value) {
    return typeof value === "string" && !Number.isNaN(Date.parse(value)) ? value : null;
  }

  function normalize(raw) {
    if (!raw || raw.version !== VERSION || !raw.chapters || typeof raw.chapters !== "object" || Array.isArray(raw.chapters)) return null;
    const next = empty();
    for (const id of IDS) {
      const entry = raw.chapters[id];
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
      const chapter = {};
      for (const field of TIMES) {
        const value = isoOrNull(entry[field]);
        if (value) chapter[field] = value;
      }
      if (chapter.quizPassedAt && typeof entry.quizRevision === "string") chapter.quizRevision = entry.quizRevision;
      if (typeof entry.note === "string" && entry.note.trim() && new TextEncoder().encode(entry.note).length <= NOTE_LIMIT) {
        chapter.note = entry.note;
        chapter.noteAt = isoOrNull(entry.noteAt);
      }
      if (Object.keys(chapter).length) next.chapters[id] = chapter;
    }
    next.updatedAt = isoOrNull(raw.updatedAt);
    return next;
  }

  function report(failed) {
    persistent = !failed;
  }

  function read() {
    try {
      const raw = localStorage.getItem(KEY);
      state = (raw === null ? null : normalize(JSON.parse(raw))) ?? empty();
      report(false);
    } catch {
      state = empty();
      report(true);
    }
  }

  function write() {
    state.updatedAt = now();
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
      report(false);
    } catch {
      report(true);
    }
    notify();
  }

  function notify() {
    for (const listener of listeners) listener(snapshot());
  }

  function snapshot() {
    return JSON.parse(JSON.stringify(state));
  }

  function chapter(id) {
    return state.chapters[id] ? { ...state.chapters[id] } : {};
  }

  function entry(id) {
    state.chapters[id] ??= {};
    return state.chapters[id];
  }

  // readAt, codeAt and ranAt are set once: the first time counts, later visits keep the first date.
  function mark(id, field) {
    if (!CHAPTERS[id] || !["readAt", "codeAt", "ranAt"].includes(field)) return false;
    if (state.chapters[id]?.[field]) return false;
    entry(id)[field] = now();
    write();
    return true;
  }

  function passQuiz(id, revision) {
    if (!CHAPTERS[id]) return false;
    const target = entry(id);
    target.quizPassedAt = now();
    target.quizRevision = String(revision);
    write();
    return true;
  }

  function setNote(id, text) {
    if (!CHAPTERS[id] || typeof text !== "string") return false;
    if (new TextEncoder().encode(text).length > NOTE_LIMIT) return false;
    const target = entry(id);
    if (text.trim()) {
      target.note = text;
      target.noteAt = now();
    } else {
      delete target.note;
      delete target.noteAt;
    }
    if (!Object.keys(target).length) delete state.chapters[id];
    write();
    return true;
  }

  function quizCurrent(id, record) {
    if (!record.quizPassedAt) return false;
    const bank = window.LlmQuizBank;
    if (!bank) return true;
    return record.quizRevision === bank.revision(id);
  }

  // The band's reading of one chapter: done from the record, none for a step the chapter lacks, exactly one now
  // on the first step still to do, pending after it. Without any record every step is pending.
  function steps(id) {
    const meta = CHAPTERS[id];
    if (!meta) return [];
    const record = chapter(id);
    const done = {
      read: Boolean(record.readAt),
      code: Boolean(record.codeAt),
      run: Boolean(record.ranAt),
      quiz: quizCurrent(id, record),
      note: Boolean(record.note),
    };
    const lacks = { code: !meta.panel, run: !meta.run };
    let placed = false;
    return STEPS.map((step) => {
      let stateName;
      if (lacks[step]) stateName = "none";
      else if (done[step]) stateName = "done";
      else if (!placed) { stateName = "now"; placed = true; }
      else stateName = "pending";
      return { step, label: STEP_LABELS[step], state: stateName };
    });
  }

  function nowStep(id) {
    return steps(id).find((item) => item.state === "now") ?? null;
  }

  // Where to continue: the earliest chapter with a now step; null once every chapter is done.
  function continueAt() {
    for (const id of IDS) {
      const step = nowStep(id);
      if (step) return { id, step: step.step, label: step.label, steps: steps(id) };
    }
    return null;
  }

  function hasAnyRecord() {
    return Object.keys(state.chapters).length > 0;
  }

  function exportData() {
    return { format: FORMAT, version: VERSION, exportedAt: now(), progress: snapshot() };
  }

  function importData(payload) {
    const next = normalize(payload?.format === FORMAT ? payload.progress : payload);
    if (!next) return false;
    state = next;
    write();
    return true;
  }

  function reset() {
    state = empty();
    try {
      localStorage.removeItem(KEY);
      report(false);
    } catch {
      report(true);
    }
    notify();
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  read();
  window.addEventListener("storage", (event) => {
    if (event.key === null || event.key === KEY) { read(); notify(); }
  });

  window.LearnProgress = Object.freeze({
    key: KEY,
    format: FORMAT,
    chapters: Object.freeze(IDS.slice()),
    stepNames: Object.freeze(STEPS.slice()),
    stepLabels: Object.freeze({ ...STEP_LABELS }),
    noteLimit: NOTE_LIMIT,
    features: (id) => (CHAPTERS[id] ? { ...CHAPTERS[id] } : null),
    chapter,
    mark,
    passQuiz,
    setNote,
    steps,
    nowStep,
    continueAt,
    hasAnyRecord,
    exportData,
    importData,
    reset,
    subscribe,
    snapshot,
    isPersistent: () => persistent,
  });
})();
