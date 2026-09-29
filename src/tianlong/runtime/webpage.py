"""
[INPUT]: 无（纯静态内容）
[OUTPUT]: 对外提供 PAGE（网页前端的整页 HTML：样式与脚本内联，无外部依赖）
[POS]: runtime/web 的页面：宣纸色调的聊天式界面。叙述按 SSE 逐句浮现，NPC 台词（“……”）单独着色，
       侧栏显示时辰与所在，提示/回顾/所知一键发出元指令，落幕后展示终章与真相揭晓；深色模式与手机宽度皆可用。
       页面只呈现服务端给的文字，不做任何判断
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>天龙 · 江湖主持</title>
<link rel="icon" href="data:,">
<style>
:root {
  --paper: #f5f0e6; --paper-2: #ece4d3; --ink: #2b2620; --ink-soft: #6b6153; --line: #d8ccb4;
  --cinnabar: #a8392b; --jade: #2f6b5f; --quote: #7a3b1d; --me: #34506b; --card: #fffaf0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #17150f; --paper-2: #211e16; --ink: #e9e1cf; --ink-soft: #a89d86; --line: #3a3427;
    --cinnabar: #e07a5f; --jade: #7fbfae; --quote: #e2b07a; --me: #9cc0e0; --card: #1d1a13;
  }
}
:root[data-theme="dark"] {
  --paper: #17150f; --paper-2: #211e16; --ink: #e9e1cf; --ink-soft: #a89d86; --line: #3a3427;
  --cinnabar: #e07a5f; --jade: #7fbfae; --quote: #e2b07a; --me: #9cc0e0; --card: #1d1a13;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC", "STSong", serif;
  font-size: 17px; line-height: 1.85; -webkit-font-smoothing: antialiased;
}
.app { display: grid; grid-template-columns: 1fr 240px; height: 100vh; max-width: 1100px; margin: 0 auto; }
header.top {
  grid-column: 1 / -1; display: flex; align-items: baseline; gap: 14px; padding: 14px 20px 10px;
  border-bottom: 1px solid var(--line);
}
header.top h1 { font-size: 20px; margin: 0; letter-spacing: 0.12em; font-weight: 600; }
header.top .seal {
  display: inline-block; color: var(--paper); background: var(--cinnabar); font-size: 12px;
  padding: 1px 6px; border-radius: 3px; letter-spacing: 0.1em;
}
header.top .mode { margin-left: auto; color: var(--ink-soft); font-size: 13px; }
main { display: flex; flex-direction: column; min-height: 0; }
#log { flex: 1; overflow-y: auto; padding: 18px 22px 8px; }
.entry { margin: 0 0 18px; animation: fade 0.35s ease; }
@keyframes fade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
.entry.me { color: var(--me); font-size: 15px; margin-bottom: 8px; }
.entry.me::before { content: "你 ▸ "; opacity: 0.7; }
.entry.gm p { margin: 0 0 6px; white-space: pre-wrap; }
.entry.gm .q { color: var(--quote); }
.entry.aside { color: var(--jade); font-size: 15px; border-left: 3px solid var(--jade); padding-left: 10px; }
.entry.opening { border-bottom: 1px dashed var(--line); padding-bottom: 14px; }
.entry.error { color: var(--cinnabar); font-size: 14px; }
.entry.ending {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
}
.entry.ending h2 { margin: 0 0 8px; font-size: 18px; color: var(--cinnabar); letter-spacing: 0.1em; }
.cursor::after { content: "▍"; color: var(--ink-soft); animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
form.say { display: flex; gap: 8px; padding: 12px 20px 16px; border-top: 1px solid var(--line); background: var(--paper); }
form.say input {
  flex: 1; font: inherit; font-size: 16px; padding: 10px 12px; color: var(--ink);
  background: var(--card); border: 1px solid var(--line); border-radius: 8px; outline: none;
}
form.say input:focus { border-color: var(--cinnabar); }
button {
  font: inherit; font-size: 15px; cursor: pointer; color: var(--ink); background: var(--paper-2);
  border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px;
}
button.primary { background: var(--cinnabar); color: #fff; border-color: var(--cinnabar); }
button:disabled, input:disabled { opacity: 0.55; cursor: default; }
aside.side { border-left: 1px solid var(--line); padding: 18px 16px; display: flex; flex-direction: column; gap: 16px; }
.stat .k { color: var(--ink-soft); font-size: 13px; }
.stat .v { font-size: 18px; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; }
.tip { color: var(--ink-soft); font-size: 13px; line-height: 1.7; }
@media (max-width: 760px) {
  body { font-size: 16px; }
  .app { grid-template-columns: 1fr; grid-template-rows: auto auto 1fr; }
  header.top { order: -2; padding: 10px 16px 8px; }
  aside.side { border-left: none; border-bottom: 1px solid var(--line); padding: 6px 16px 8px; flex-direction: row;
               flex-wrap: wrap; align-items: center; gap: 6px 12px; order: -1; }
  aside.side button { font-size: 13px; padding: 3px 9px; }
  .stat { display: flex; align-items: baseline; gap: 6px; }
  aside.side .tip { display: none; }
  .stat .v { font-size: 15px; }
  #log { padding: 14px 16px 6px; }
  form.say { padding: 10px 16px 14px; }
}
</style>
</head>
<body>
<div class="app">
  <header class="top"><span class="seal">江湖</span><h1 id="title">天龙</h1><span class="mode" id="mode"></span></header>
  <main>
    <div id="log" aria-live="polite"></div>
    <form class="say" id="form" autocomplete="off">
      <input id="input" placeholder="说你想做的事，或想说的话……" maxlength="300" autofocus>
      <button class="primary" id="send" type="submit">行</button>
    </form>
  </main>
  <aside class="side">
    <div class="stat"><div class="k">时辰</div><div class="v" id="clock">—</div></div>
    <div class="stat"><div class="k">所在</div><div class="v" id="place">—</div></div>
    <div class="tools">
      <button type="button" data-cmd="/hint">提示</button>
      <button type="button" data-cmd="/recap">回顾</button>
      <button type="button" data-cmd="/beliefs">所知</button>
    </div>
    <div class="tip" id="tip"></div>
    <div class="tools"><button type="button" id="theme">明暗</button><button type="button" id="restart">重开</button></div>
  </aside>
</div>
<script>
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send");
let busy = false, ended = false;

function esc(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
// 台词着色：“……”整段包起来；未收尾的引号在流式途中先照原样显示
function paint(text) { return esc(text).replace(/“[^”]*”/g, (m) => `<span class="q">${m}</span>`); }

function add(cls, html) {
  const div = document.createElement("div");
  div.className = "entry " + cls;
  if (html !== undefined) div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}
function paragraphs(text) { return text.split(/\n+/).filter(Boolean).map((p) => `<p>${paint(p)}</p>`).join(""); }
function setState(s) {
  if (s.clock) $("clock").textContent = s.clock;
  if (s.place) $("place").textContent = s.place;
}
function showEnding(title, epilogue) {
  ended = true;
  const card = add("ending", `<h2>${esc(title || "落幕")}</h2>` + paragraphs(epilogue || ""));
  card.scrollIntoView({ behavior: "smooth" });
  input.placeholder = "本幕已终。点“重开”再入江湖。";
  input.disabled = send.disabled = true;
}
function lock(on) { busy = on; input.disabled = send.disabled = on || ended; }

async function start() {
  const s = await (await fetch("/api/state")).json();
  $("title").textContent = s.title || "天龙";
  $("mode").textContent = s.voice ? "Gemini 叙述" : "模板叙述";
  $("tip").textContent = s.hints || "";
  log.innerHTML = "";
  add("gm opening", paragraphs(s.opening || ""));
  setState(s);
  ended = false; input.disabled = send.disabled = false; input.placeholder = "说你想做的事，或想说的话……";
  if (s.ended) showEnding(s.ending, s.epilogue);
  input.focus();
}

async function play(text) {
  if (busy || ended || !text.trim()) return;
  lock(true);
  add("me", esc(text));
  const aside = text.startsWith("/") || /^GM[:：]/i.test(text);
  const box = add(aside ? "aside cursor" : "gm cursor");
  let acc = "";
  try {
    const resp = await fetch("/api/turn", { method: "POST", headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ text }) });
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut); buf = buf.slice(cut + 2);
        const ev = (chunk.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((chunk.match(/^data: (.*)$/m) || [, "{}"])[1]);
        if (ev === "text") { acc += data.t; box.innerHTML = paragraphs(acc); log.scrollTop = log.scrollHeight; }
        else if (ev === "done") {
          if (!acc && data.narration) box.innerHTML = paragraphs(data.narration);
          if (data.kind === "ask_gm" || data.kind === "meta" || data.kind === "unclear") box.className = "entry aside";
          setState(data);
          if (data.ended) showEnding(data.ending, data.epilogue);
        } else if (ev === "error") { add("error", esc(data.message || "出错了")); }
      }
    }
  } catch (e) {
    add("error", "与主持人失去联系，请稍后再试。");
  } finally {
    box.classList.remove("cursor");
    lock(false);
    if (!ended) input.focus();
  }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; play(t); });
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => play(b.dataset.cmd)));
$("restart").addEventListener("click", async () => {
  if (busy) return;
  await fetch("/api/restart", { method: "POST" });
  start();
});
$("theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) {}
});
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
start();
</script>
</body>
</html>
"""
