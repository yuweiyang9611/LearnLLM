// Colours source text on demand for the source panel, which loads files after the page.
// A port of the reference code-colors.js, with the kit's token classes: keyword and number purple,
// type cyan, call blue, string green, comment and meta soft. Material's Pygments colours the Markdown
// code blocks at build time; this only colours the bundled files the panel fetches.
(() => {
  "use strict";

  const pythonKeywords = "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case self cls".replaceAll(" ", "|");

  const python = {
    comment: String.raw`#[^\n]*`,
    string: String.raw`[rRbBuUfF]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')`,
    meta: String.raw`^[ \t]*@[A-Za-z_][\w.]*`,
    number: String.raw`\b(?:0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?j?)\b`,
    keyword: String.raw`\b(?:${pythonKeywords})\b`,
    call: String.raw`\b[A-Za-z_]\w*(?=\()`,
    type: String.raw`\b[A-Z]\w*`,
  };

  // In a shell the program is the type and its switches are the keywords.
  const shell = {
    comment: String.raw`(?<!\S)#[^\n]*`,
    string: String.raw`"[^"\n]*"|'[^'\n]*'`,
    type: String.raw`^[ \t]*(?:[.&][ \t]+)?[A-Za-z_./\\][\w./\\:-]*(?![\w./\\:-]|[ \t]*=)`,
    keyword: String.raw`(?<=\s)--?[A-Za-z][\w-]*`,
    meta: String.raw`\$[A-Za-z_][\w:]*`,
    number: String.raw`(?<![\w.-])\d+(?:\.\d+)?\b`,
  };

  // Data files: keys as types, literals as strings and numbers.
  const data = {
    comment: String.raw`//[^\n]*|/\*[\s\S]*?\*/|(?<!\S)#[^\n]*`,
    type: String.raw`"(?:[^"\\\n]|\\.)*"(?=\s*:)|^[ \t]*[A-Za-z_][\w.-]*(?=[ \t]*[:=])`,
    string: String.raw`"(?:[^"\\\n]|\\.)*"|'[^'\n]*'`,
    number: String.raw`(?<![\w.-])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b`,
    keyword: String.raw`\b(?:true|false|null|True|False|None)\b`,
  };

  // Markdown: headings as types, code as strings, links as calls.
  const markdown = {
    meta: String.raw`^#{1,6}[ \t][^\n]*|^(?:-{3,}|\*{3,})[ \t]*$`,
    string: "```[\\s\\S]*?```|`[^`\\n]+`",
    call: String.raw`\[[^\]\n]*\]\([^)\n]*\)`,
    keyword: String.raw`^[ \t]*(?:[-*+]|\d+\.)(?=[ \t])`,
    type: String.raw`^>[ \t]?`,
  };

  const compile = (rules) => new RegExp(Object.entries(rules).map(([name, source]) => `(?<${name}>${source})`).join("|"), "gm");
  const patterns = {
    python: compile(python),
    powershell: compile(shell),
    shell: compile(shell),
    json: compile(data),
    yaml: compile(data),
    toml: compile(data),
    ini: compile(data),
    markdown: compile(markdown),
  };
  const languages = {
    py: "python", pyi: "python", ps1: "powershell", psd1: "powershell", sh: "shell", bash: "shell",
    json: "json", yml: "yaml", yaml: "yaml", toml: "toml", cfg: "ini", ini: "ini", md: "markdown",
  };

  function highlight(source, language) {
    const parts = document.createDocumentFragment();
    const pattern = patterns[language];
    if (!pattern) {
      parts.append(source);
      return parts;
    }
    let position = 0;
    for (const match of source.matchAll(pattern)) {
      parts.append(source.slice(position, match.index));
      const token = document.createElement("span");
      token.className = `tok-${Object.keys(match.groups).find((name) => match.groups[name] !== undefined)}`;
      token.textContent = match[0];
      parts.append(token);
      position = match.index + match[0].length;
    }
    parts.append(source.slice(position));
    return parts;
  }

  function languageFor(path) {
    const extension = String(path).toLowerCase().match(/\.([a-z0-9]+)$/)?.[1] ?? "";
    return languages[extension] ?? (extension || "text");
  }

  window.CodeColors = Object.freeze({ highlight, languageFor });
})();
