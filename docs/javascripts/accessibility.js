function enhanceToggleButtons() {
  const selectors = [
    'label.md-header__button[for="__drawer"]',
    'label.md-header__button[for="__search"]',
    'label.md-search__icon[for="__search"]'
  ];

  document.querySelectorAll(selectors.join(",")).forEach((label) => {
    const targetId = label.getAttribute("for");
    const fallbackLabel = targetId === "__drawer"
      ? "打开导航"
      : (label.classList.contains("md-search__icon") ? "关闭搜索" : "打开搜索");
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

    label.setAttribute("role", "button");
    label.setAttribute("tabindex", "0");
    label.setAttribute("aria-label", label.getAttribute("title") || fallbackLabel);
    if (targetId === "__search" && !label.classList.contains("md-search__icon")) {
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
