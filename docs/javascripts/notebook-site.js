// Opening search from the 搜索 word puts the caret in the field, as the / key already does.
(() => {
  "use strict";

  const toggle = document.getElementById("__search");
  const query = document.querySelector('[data-md-component="search-query"]');
  if (!toggle || !query) return;
  toggle.addEventListener("change", () => {
    if (toggle.checked && document.activeElement !== query) requestAnimationFrame(() => query.focus());
  });
})();

// The running head's 菜单 word shows the collapsed links on narrow screens (the kit leaves this to each site).
(() => {
  "use strict";

  const toggle = document.querySelector(".nb-nav-toggle");
  const nav = document.getElementById("nb-site-nav");
  if (!toggle || !nav) return;

  function setOpen(open) {
    toggle.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("open", open);
  }

  toggle.addEventListener("click", () => {
    setOpen(toggle.getAttribute("aria-expanded") !== "true");
  });

  // 目录 and 搜索 open panels of their own, so the menu steps aside for them.
  nav.addEventListener("click", (event) => {
    if (event.target.closest("label")) setOpen(false);
  });
  // Capture, because accessibility.js stops Enter and Space on these labels once it has handled them.
  nav.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target.closest("label")) setOpen(false);
  }, true);
})();
