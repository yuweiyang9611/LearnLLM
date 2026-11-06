// The chapter's self-check, rendered into the <div class="llm-quiz" data-chapter="NN"> marker before the
// chapter's last section: three questions from quiz-bank.js, answered in place. When every question has been
// answered correctly, the learning record gets the chapter's quizPassedAt with the bank's revision.
(() => {
  "use strict";

  const bank = window.LlmQuizBank;
  const progress = window.LearnProgress;
  const marker = document.querySelector(".llm-quiz[data-chapter]");
  if (!bank || !marker) return;
  const chapter = marker.dataset.chapter;
  const questions = bank.questionsFor(chapter);
  if (!questions.length) return;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const correct = new Set();
  const heading = element("h2", "llm-quiz-title", `自测 · ${questions.length} 题`);
  heading.id = "llm-quiz-heading";
  const summary = element("p", "llm-quiz-summary");
  summary.setAttribute("role", "status");
  summary.setAttribute("aria-live", "polite");
  const list = element("ol", "llm-quiz-list");

  function passed() {
    const record = progress?.chapter(chapter) ?? {};
    return Boolean(record.quizPassedAt) && record.quizRevision === bank.revision(chapter);
  }

  function renderSummary() {
    if (correct.size === questions.length) {
      summary.textContent = `${questions.length} / ${questions.length} 题答对，已记下。`;
    } else if (passed()) {
      summary.textContent = `上次已全部答对；再答一遍也可以。`;
    } else {
      summary.textContent = `${correct.size} / ${questions.length} 题答对。全部答对后这一章的自测记为完成。`;
    }
  }

  function renderQuestion(question, index) {
    const item = element("li", "llm-quiz-item nb-entry");
    item.dataset.quizId = question.id;
    const form = element("form", "llm-quiz-form");
    const fieldset = element("fieldset");
    const legend = element("legend", null, `第 ${index + 1} 题 · ${question.question}`);
    const options = element("div", "llm-quiz-options");
    question.options.forEach((text, optionIndex) => {
      const label = element("label", "llm-quiz-option");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `quiz-${question.id}`;
      input.value = String(optionIndex);
      input.required = true;
      label.append(input, element("span", null, text));
      options.append(label);
    });
    const submit = element("button", "nb-button", "提交答案");
    submit.type = "submit";
    submit.disabled = true;
    const feedback = element("p", "llm-quiz-feedback");
    feedback.hidden = true;
    feedback.setAttribute("role", "status");
    fieldset.append(legend, options);
    form.append(fieldset, submit, feedback);
    item.append(form);

    options.addEventListener("change", () => { submit.disabled = false; });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const chosen = Number(new FormData(form).get(`quiz-${question.id}`));
      if (!Number.isInteger(chosen)) return;
      const right = chosen === question.answer;
      for (const input of options.querySelectorAll("input")) {
        input.disabled = true;
        const label = input.closest("label");
        label.classList.toggle("correct", Number(input.value) === question.answer);
        label.classList.toggle("wrong", input.checked && !right);
      }
      submit.disabled = true;
      feedback.hidden = false;
      feedback.dataset.state = right ? "done" : "wrong";
      feedback.textContent = `${right ? "答对了。" : `更合适的是：${question.options[question.answer]}。`}${question.explanation}`;
      item.dataset.state = right ? "done" : "wrong";
      if (right) correct.add(question.id);
      else {
        // A wrong answer can be tried again: the question resets after the explanation is read.
        const retry = element("button", "nb-button llm-quiz-retry", "再答一次");
        retry.type = "button";
        retry.addEventListener("click", () => {
          for (const input of options.querySelectorAll("input")) {
            input.disabled = false;
            input.checked = false;
            input.closest("label").classList.remove("correct", "wrong");
          }
          feedback.hidden = true;
          feedback.textContent = "";
          retry.remove();
          delete item.dataset.state;
        });
        form.append(retry);
      }
      if (correct.size === questions.length && progress && !passed()) progress.passQuiz(chapter, bank.revision(chapter));
      renderSummary();
    });
    return item;
  }

  list.append(...questions.map(renderQuestion));
  marker.replaceChildren(heading, summary, list);
  marker.classList.add("llm-quiz--ready");
  renderSummary();
  progress?.subscribe(renderSummary);
})();
