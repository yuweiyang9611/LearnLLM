// Material opens the course contents and search through checkbox labels. These are the running head's
// 目录 and 搜索 words and the search panel's 关闭 word, so make each one a keyboard button.
function enhanceToggleButtons() {
  const selectors = [
    '.nb-nav label[for="__drawer"]',
    '.nb-nav label[for="__search"]',
    'label.md-search__close[for="__search"]'
  ];

  document.querySelectorAll(selectors.join(",")).forEach((label) => {
    const targetId = label.getAttribute("for");
    const target = document.getElementById(targetId);
    const controlledRegion = targetId === "__drawer"
      ? document.querySelector(".md-sidebar--primary")
      : document.querySelector('[data-md-component="search"]');
    const controlledId = targetId === "__drawer" ? "__drawer_panel" : "__search_dialog";

    if (controlledRegion) {
      controlledRegion.id = controlledId;
      label.setAttribute("aria-controls", controlledId);
    }

    if (target) {
      label.setAttribute("aria-expanded", String(target.checked));
    }

    if (label.dataset.keyboardReady === "true") return;

    // The visible word is the accessible name, so speech input can say what it reads.
    label.setAttribute("role", "button");
    label.setAttribute("tabindex", "0");
    if (targetId === "__search" && !label.classList.contains("md-search__close")) {
      label.setAttribute("aria-haspopup", "dialog");
    }
    label.dataset.keyboardReady = "true";

    if (target) {
      target.addEventListener("change", () => {
        label.setAttribute("aria-expanded", String(target.checked));
      });
    }

    label.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      event.stopPropagation();
      if (!target) return;
      target.checked = !target.checked;
      target.dispatchEvent(new Event("change", { bubbles: true }));
    });
  });
}

enhanceToggleButtons();

if (typeof document$ !== "undefined") {
  document$.subscribe(enhanceToggleButtons);
}
