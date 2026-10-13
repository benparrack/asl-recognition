// SignDesk client. The server owns recognition, the transcript and the
// recognition settings; this page renders its state (server-sent events) and
// sends commands. Display preferences and practice history live in
// localStorage, which is optional: everything works without it.
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

function load(key, defaults) {
  try { return { ...defaults, ...JSON.parse(localStorage.getItem(key) || "{}") }; }
  catch { return { ...defaults }; }
}
function store(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window */ }
}

const PREFS_KEY = "signdesk.prefs.v1";
const STATS_KEY = "signdesk.practice.v1";
const prefs = load(PREFS_KEY, {
  theme: "system", size: "m", case: "sentence", font: "sans", speakWords: false,
  layout: "studio", tab: "activity", helpSeen: false, practiceKind: "words", practicePool: "all", bestStreak: 0,
});
const SIZES = { s: 22, m: 30, l: 40, xl: 54 };
const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");
const RING = 2 * Math.PI * 54;

let S = null;              // latest server state
let lastEvent = null;      // highest event id already handled
let vocab = [];            // [{name, label, text, two_handed}]
let shownText = null;      // transcript text currently rendered
let shownCandidates = "";
let shownEvents = "";
let offline = false;
let streamStarted = false;

// ------------------------------------------------------------------ server
async function cmd(action, data = {}) {
  try {
    const r = await fetch("/api/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...data }),
    });
    const j = await r.json();
    if (!r.ok) { toast(j.error || "That didn't work"); return null; }
    apply(j);
    return j;
  } catch {
    toast("Can't reach SignDesk. Is it still running?");
    return null;
  }
}

function connect() {
  const es = new EventSource("/events");
  es.onopen = () => {
    S = null;                                  // a restarted server counts versions from 0
    setOffline(false);
    startStream();
  };
  es.onmessage = (e) => apply(JSON.parse(e.data));
  es.onerror = () => setOffline(true);         // EventSource reconnects by itself
}

function startStream() {
  const img = $("#video");
  img.src = "/stream.mjpg?t=" + Date.now();
  streamStarted = true;
}
$("#video").addEventListener("error", () => { if (!offline) setTimeout(startStream, 1000); });

function setOffline(v) {
  offline = v;
  $("#overlay-offline").hidden = !v;
  if (v) setStatus("error", "Disconnected");
}

let frame = 0;
function apply(st) {
  if (!st || typeof st.v !== "number") return;
  if (S && st.v < S.v) return;                 // an older snapshot arriving late
  const first = !S;
  S = st;
  if (first) onFirstState();
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
}

function onFirstState() {
  if (lastEvent === null) lastEvent = S.events.length ? S.events[S.events.length - 1].id : 0;
  // A reload mid-practice or mid-edit: the page lost that state, so end it.
  if (S.practice && !P.running) cmd("practice", { on: false });
  if (S.editing && !editing) cmd("editing", { on: false });
  if (!vocab.length && S.words.available) loadVocab();
  fillCameraSelect();
}

// ------------------------------------------------------------------ render
function render() {
  if (!S) return;
  renderStatus();
  renderMode();
  if (S.mode === "letters") renderLetters(); else renderWords();
  renderTranscript();
  renderEvents();
  renderSettings();
  handleNewEvents();
}

function setStatus(kind, text) {
  const pill = $("#status-pill");
  pill.className = "status-pill " + kind;
  $("#status-text").textContent = text;
}

function renderStatus() {
  const cam = S.camera;
  if (offline) return;
  if (S.paused) setStatus("paused", "Paused");
  else if (!cam.ok && cam.error) setStatus("error", "Camera unavailable");
  else if (!cam.ok) setStatus("", "Starting…");
  else setStatus("live", `Live · ${Math.round(cam.fps)} fps`);

  $("#overlay-paused").hidden = !S.paused;
  $("#overlay-error").hidden = S.paused || cam.ok || !cam.error;
  $("#camera-error-text").textContent = cam.error || "";
  $("#overlay-connecting").hidden = S.paused || !!cam.error || (cam.ok && streamStarted && $("#video").naturalWidth > 0);

  const pause = $("#btn-pause");
  pause.querySelector("use").setAttribute("href", S.paused ? "#i-play" : "#i-pause");
  pause.title = S.paused ? "Resume camera (P)" : "Pause camera (P)";
  pause.setAttribute("aria-label", S.paused ? "Resume camera" : "Pause camera");

  $("#practice-banner").hidden = !S.practice;
  const m = S.models;
  $("#model-info").textContent = [
    `Letters: ${m.letters}`, m.motion ? `J/Z: ${m.motion}` : null,
    m.words ? `Words: ${m.words}` : "Words: not loaded", m.device,
  ].filter(Boolean).join("  ·  ");
}

function renderMode() {
  $$("#mode-switch button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.mode === S.mode));
    if (b.dataset.mode === "words") {
      b.disabled = !S.words.available;
      if (!S.words.available) b.title = "No word model loaded (--words-checkpoint)";
    }
  });
  $("#predict-letters").hidden = S.mode !== "letters";
  $("#predict-words").hidden = S.mode !== "words";
}

function renderLetters() {
  const L = S.letters;
  const s = S.settings;
  const tile = $("#letter-tile");
  // The smoother keeps its last letter after the hand leaves; don't show it.
  const letter = S.hand || L.motion ? L.letter : null;
  const held = L.held && !!letter;
  $("#letter-glyph").textContent = letter || "–";
  tile.classList.toggle("empty", !letter);
  tile.classList.toggle("added", held);
  const progress = L.motion ? 0 : (s.letters_auto_add ? L.hold : 0);
  $("#hold-progress").style.strokeDashoffset = String(RING * (1 - (held ? 1 : progress)));

  let title, sub;
  if (L.motion) { title = "Tracing a motion letter…"; sub = "Finish the J or Z, then hold still."; }
  else if (!S.hand) { title = "Show your hand"; sub = "Hold your signing hand up, palm toward the camera."; }
  else if (!letter) { title = "Reading your hand…"; sub = "Hold the handshape steady."; }
  else if (held && !S.practice && !S.editing) { title = `Added “${letter}”`; sub = "Change letter, or lower your hand to repeat it."; }
  else if (held) { title = `Recognised “${letter}”`; sub = S.practice ? "Scored in practice." : "Not added while editing."; }
  else if (s.letters_auto_add) { title = `Letter ${letter}`; sub = "Keep holding to add it."; }
  else { title = `Letter ${letter}`; sub = "Press Enter to add it."; }
  $("#letter-title").textContent = title;
  $("#letter-sub").textContent = sub;

  const conf = letter ? L.confidence : 0;
  const meter = $("#letter-meter");
  meter.style.width = `${Math.round(conf * 100)}%`;
  meter.classList.toggle("low", conf < s.letters_min_confidence);
  $("#letter-conf").textContent = letter ? `Confidence ${Math.round(conf * 100)}%` : "Confidence —";
  $("#motion-chip").hidden = !L.motion;
  $("#btn-add-letter").disabled = !letter;

  const vf = $("#video-frame");
  vf.classList.toggle("recording", L.motion);
  vf.classList.toggle("holding", !L.motion && !!letter && !held && progress > 0.05);
  vf.classList.toggle("added", held && !!letter);
  setChip(L.motion ? "rec" : S.hand ? "ok" : "",
    L.motion ? "Tracing J/Z" : !S.hand ? "No hand in view" : letter ? `Signing ${letter}` : "Hand detected");
}

function setChip(kind, text) {
  $("#video-chip").className = "video-chip " + kind;
  $("#video-chip-text").textContent = text;
}

function renderWords() {
  const W = S.words;
  const rec = W.state !== "idle";
  const auto = S.settings.words_auto_segment;
  const box = $("#rec-state");
  box.className = "rec-state " + (rec ? "recording" : W.raised ? "ready" : "");
  if (rec) {
    $("#words-title").textContent = "Recording sign…";
    $("#words-sub").textContent = W.state === "manual"
      ? "Press Enter to stop and recognise it."
      : `${W.buffered} frames. Lower your hands or hold still to finish.`;
  } else {
    $("#words-title").textContent = "Ready";
    $("#words-sub").textContent = auto
      ? (S.hand ? "Raise your hands into signing space to start." : "Raise your hands to start a sign.")
      : "Press Enter or Record to start a sign.";
  }
  const btn = $("#btn-sign");
  btn.classList.toggle("btn-primary", !rec);
  btn.classList.toggle("recording", rec);
  btn.querySelector("use").setAttribute("href", rec ? "#i-stop" : "#i-record");
  btn.querySelector("span").textContent = rec ? "Stop & recognise" : "Record sign";
  $("#btn-cancel-sign").hidden = !rec;

  const vf = $("#video-frame");
  vf.classList.toggle("recording", rec);
  vf.classList.remove("holding", "added");
  setChip(rec ? "rec" : S.hand ? "ok" : "", rec ? "Recording sign" : S.hand ? "Hands in view" : "Ready");

  const key = `${W.event}:${W.chosen}:${W.candidates.length}`;
  if (key === shownCandidates) return;
  shownCandidates = key;
  const ev = S.events.find((e) => e.id === W.event);
  const unsure = ev && !ev.committed;
  const head = $("#candidates-head");
  head.hidden = !W.candidates.length;
  head.classList.toggle("unsure", !!unsure);
  $("#candidates-label").textContent = unsure
    ? (ev.outcome === "low_confidence" ? "Not sure. Pick the right sign" : "Pick a sign to add")
    : "Top matches";
  $("#candidates-hint").textContent = unsure ? "Press 1–3 or click to add it" : "Press 1–3 or click to swap the word";
  const list = $("#candidates");
  list.replaceChildren(...W.candidates.map((c, i) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "candidate" + (i === W.chosen ? " chosen" : "");
    b.dataset.action = "choose";
    b.dataset.index = String(i);
    b.setAttribute("aria-label", `${c.label}, ${Math.round(c.p * 100)} percent`);
    b.innerHTML = `<span class="idx">${i + 1}</span><span class="name"></span>`
      + `<span class="bar"><span style="width:${Math.max(2, Math.round(c.p * 100))}%"></span></span>`
      + `<span class="pct">${Math.round(c.p * 100)}%</span>`;
    b.querySelector(".name").textContent = c.label;
    li.append(b);
    return li;
  }));
}

// ------------------------------------------------------------------ transcript
function displayText(t) {
  switch (prefs.case) {
    case "upper": return t.toUpperCase();
    case "lower": return t.toLowerCase();
    case "sentence": {
      let s = t.toLowerCase();
      s = s.replace(/(^\s*|[.!?]\s+|\n\s*)(\p{L})/gu, (_, a, b) => a + b.toUpperCase());
      return s.replace(/\bi\b/g, "I").replace(/\btv\b/g, "TV");
    }
    default: return t;
  }
}

function renderTranscript() {
  const text = S.transcript.text;
  $("#btn-undo").disabled = !S.transcript.can_undo;
  $("#btn-redo").disabled = !S.transcript.can_redo;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  $("#transcript-count").textContent = `${words} word${words === 1 ? "" : "s"} · ${text.length} characters`;
  if (editing) return;

  const box = $("#transcript-text");
  const cursor = S.transcript.cursor;      // null = at the end
  const key = text + "\u0000" + prefs.case + "\u0000" + cursor;
  if (key !== shownText) {
    const prev = shownText === null ? null : shownText.split("\u0000")[0];
    shownText = key;
    const shown = displayText(text);
    box.replaceChildren();
    if (!text) {
      const ph = document.createElement("span");
      ph.className = "placeholder";
      ph.innerHTML = S.mode === "words"
        ? "<strong>Sign a word to start.</strong><br>Raise your hands, sign, then lower them. Recognised words appear here."
        : "<strong>Fingerspell to start.</strong><br>Hold each letter until its ring fills. Lower your hand to end a word.";
      box.append(ph);
    } else {
      const grew = cursor === null && prev !== null && text.length > prev.length && text.startsWith(prev);
      const cut = grew ? prev.length : cursor === null ? shown.length : cursor;
      box.append(document.createTextNode(shown.slice(0, cut)));
      if (grew) {
        const s = document.createElement("span");
        s.className = "flash";
        s.textContent = shown.slice(cut);
        box.append(s);
      }
    }
    const ghost = document.createElement("span");
    ghost.className = "ghost";
    ghost.id = "ghost";
    const caret = document.createElement("span");
    caret.className = "caret";
    if (cursor === null || !text) box.append(ghost, caret);
    else box.append(caret, document.createTextNode(shown.slice(cursor)), ghost);
    if (box.scrollHeight - box.scrollTop - box.clientHeight < 200) box.scrollTop = box.scrollHeight;
  }
  // The letter being held, previewed at the caret.
  const L = S.letters;
  const g = $("#ghost");
  const ghostText = S.mode === "letters" && !S.practice && L.letter && !L.held && L.hold > 0.05 && S.settings.letters_auto_add
    ? displayText((text.length ? text : "") + L.letter).slice(-1) : "";
  if (g && g.textContent !== ghostText) g.textContent = ghostText;
}

// Click in the text to put the cursor there; signing puts it back at the end.
function moveCursor(pos) {
  const n = S.transcript.text.length;
  cmd("cursor", { pos: pos === null || pos >= n ? null : Math.max(0, pos) });
}
function cursorPos() { return S.transcript.cursor === null ? S.transcript.text.length : S.transcript.cursor; }
function wordJump(pos, dir) {
  const t = S.transcript.text;
  if (dir < 0) { while (pos > 0 && /\s/.test(t[pos - 1])) pos--; while (pos > 0 && !/\s/.test(t[pos - 1])) pos--; }
  else { while (pos < t.length && !/\s/.test(t[pos])) pos++; while (pos < t.length && /\s/.test(t[pos])) pos++; }
  return pos;
}
$("#transcript-text").addEventListener("click", (e) => {
  if (!S || !S.transcript.text) return;
  const box = e.currentTarget;
  const hit = document.caretPositionFromPoint ? document.caretPositionFromPoint(e.clientX, e.clientY)
    : document.caretRangeFromPoint && document.caretRangeFromPoint(e.clientX, e.clientY);
  if (!hit) return;
  const node = hit.offsetNode || hit.startContainer, off = hit.offset ?? hit.startOffset;
  if (!box.contains(node)) return;
  if (node.nodeType !== Node.TEXT_NODE || node.parentElement.id === "ghost") { moveCursor(null); return; }
  let pos = 0;
  const walk = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
  for (let n = walk.nextNode(); n && n !== node; n = walk.nextNode()) {
    if (n.parentElement.id !== "ghost") pos += n.length;
  }
  moveCursor(pos + off);
});

let editing = false;
function startEdit() {
  if (!S || editing) return;
  editing = true;
  cmd("editing", { on: true });
  const ta = $("#transcript-edit");
  ta.value = displayText(S.transcript.text);
  ta.className = "transcript-edit font-" + prefs.font;
  $("#transcript-text").hidden = true;
  ta.hidden = false;
  $("#edit-banner").hidden = false;
  $("#btn-edit").setAttribute("aria-pressed", "true");
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
}
async function finishEdit() {
  if (!editing) return;
  const value = $("#transcript-edit").value;
  editing = false;
  $("#transcript-edit").hidden = true;
  $("#transcript-text").hidden = false;
  $("#edit-banner").hidden = true;
  $("#btn-edit").setAttribute("aria-pressed", "false");
  shownText = null;
  await cmd("set_text", { text: value });
  await cmd("editing", { on: false });
}

function plainText() { return S ? displayText(S.transcript.text).trim() : ""; }

async function copyText() {
  const t = plainText();
  if (!t) return toast("Nothing to copy yet");
  try { await navigator.clipboard.writeText(t); }
  catch {
    const ta = document.createElement("textarea");
    ta.value = t; document.body.append(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  toast("Copied to clipboard");
}

function downloadText() {
  const t = plainText();
  if (!t) return toast("Nothing to download yet");
  const a = document.createElement("a");
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  a.download = `signdesk-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.txt`;
  a.href = URL.createObjectURL(new Blob([t + "\n"], { type: "text/plain" }));
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function speak(text, quiet = false) {
  if (!("speechSynthesis" in window)) { if (!quiet) toast("Speech isn't available in this browser"); return; }
  if (!quiet && speechSynthesis.speaking) { speechSynthesis.cancel(); return; }
  if (!text) { if (!quiet) toast("Nothing to read yet"); return; }
  if (!quiet) speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

// ------------------------------------------------------------------ activity
function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 5) return "now";
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function visibleEvent(e) {
  return e.kind === "letter" || (e.kind === "word" && (e.outcome !== "too_short" || e.reason === "manual_stop"));
}

function renderEvents(force = false) {
  const evs = S.events.filter(visibleEvent);
  const key = evs.length ? `${evs[evs.length - 1].id}:${evs.length}` : "";
  if (key === shownEvents && !force) return;
  shownEvents = key;
  $("#activity-empty").hidden = evs.length > 0;
  $("#activity").replaceChildren(...evs.slice().reverse().map((e) => {
    const li = document.createElement("li");
    const badge = document.createElement("span");
    const what = document.createElement("div");
    const detail = document.createElement("div");
    const when = document.createElement("span");
    const body = document.createElement("div");
    when.className = "when"; when.textContent = ago(e.t);
    what.className = "what"; detail.className = "detail";
    const added = e.committed ? "added" : S.practice ? "practice" : "not added";
    if (e.kind === "letter") {
      badge.className = "badge"; badge.textContent = e.letter;
      what.textContent = `Letter ${e.letter}`;
      detail.textContent = `${Math.round(e.confidence * 100)}% · ${{ hold: "held", motion: "traced", manual: "added by hand" }[e.source] || e.source} · ${added}`;
    } else if (!e.top.length) {
      badge.className = "badge miss"; badge.textContent = "–";
      what.textContent = "Sign too short";
      detail.textContent = "Too few frames with a hand in view";
    } else {
      const top = e.top[0];
      badge.className = "badge word" + (e.outcome === "accepted" ? "" : " miss");
      badge.textContent = e.outcome === "accepted" ? "W" : "?";
      what.textContent = top.label;
      const others = e.top.slice(1).map((c) => `${c.label} ${Math.round(c.p * 100)}%`).join(", ");
      detail.textContent = `${Math.round(top.p * 100)}% · ${e.outcome === "low_confidence" ? "unsure" : added}${others ? ` · also ${others}` : ""}`;
    }
    body.append(what, detail);
    li.append(badge, body, when);
    return li;
  }));
}
setInterval(() => { if (S) renderEvents(true); }, 10000);

function handleNewEvents() {
  const fresh = S.events.filter((e) => e.id > lastEvent);
  if (!fresh.length) return;
  lastEvent = fresh[fresh.length - 1].id;
  for (const e of fresh) {
    if (e.kind === "word" && e.committed && prefs.speakWords && e.top.length) speak(e.top[0].label.split(" / ")[0], true);
    if (e.kind === "word" && e.outcome === "too_short" && e.reason === "manual_stop") toast("That sign was too short. Try again with your hands in view.");
    if (P.running) practiceEvent(e);
  }
}

// ------------------------------------------------------------------ settings
function fmt(input) {
  const v = Number(input.value);
  switch (input.dataset.unit) {
    case "s": return `${v.toFixed(1)} s`;
    case "%": return `${Math.round(v * 100)}%`;
    case "frames": return `${v} frames`;
    case "lull": return v === 0 ? "Off" : v < 0.3 ? "Only when very still" : v < 0.6 ? "Normal" : "Quickly";
    default: return String(v);
  }
}

function renderSettings() {
  if ($("#settings").hidden) return;
  for (const el of $$("[data-setting]")) {
    const key = el.dataset.setting;
    const v = S.settings[key];
    if (el === document.activeElement && el.type === "range") continue;
    if (el.type === "checkbox") el.checked = !!v;
    else if (String(el.value) !== String(v)) el.value = v;
    const out = $(`output[data-for="${key}"]`);
    if (out) out.textContent = fmt(el);
  }
  const motion = $("#row-motion");
  motion.classList.toggle("disabled", !S.letters.motion_available);
  motion.querySelector("input").disabled = !S.letters.motion_available;
  for (const el of $$("#settings-words input")) el.disabled = !S.words.available;
  $("#s-camera").disabled = S.camera.source.startsWith("camera") === false;
  const m = S.models;
  const dl = $("#models-list");
  const rows = [["Letters", m.letters], ["J / Z", m.motion || "off"], ["Words", m.words || "not loaded"], ["Device", m.device]];
  const html = rows.map(([k, v]) => `<dt>${k}</dt><dd></dd>`).join("");
  if (dl.dataset.k !== JSON.stringify(rows)) {
    dl.innerHTML = html;
    $$("dd", dl).forEach((dd, i) => { dd.textContent = rows[i][1]; });
    dl.dataset.k = JSON.stringify(rows);
  }
}

function fillCameraSelect() {
  const sel = $("#s-camera");
  if (!S.camera.source.startsWith("camera")) {
    sel.replaceChildren(new Option(`Video file: ${S.camera.source.split("/").pop()}`, S.settings.camera));
    return;
  }
  sel.replaceChildren(...S.cameras.map((i) => new Option(`Camera ${i}  (/dev/video${i})`, i)));
  sel.value = S.settings.camera;
}

let settingTimer = 0;
function sendSetting(el) {
  const key = el.dataset.setting;
  let v = el.type === "checkbox" ? el.checked : Number(el.value);
  if (key === "letters_window" || key === "camera") v = Math.round(v);
  clearTimeout(settingTimer);
  settingTimer = setTimeout(() => cmd("settings", { values: { [key]: v } }), el.type === "range" ? 120 : 0);
}
$$("[data-setting]").forEach((el) => {
  el.addEventListener(el.type === "range" ? "input" : "change", () => {
    const out = $(`output[data-for="${el.dataset.setting}"]`);
    if (out) out.textContent = fmt(el);
    sendSetting(el);
  });
});

let lastFocus = null;
function openSettings() {
  lastFocus = document.activeElement;
  $("#settings").hidden = false;
  $("#scrim").hidden = false;
  if (S) { fillCameraSelect(); renderSettings(); }
  $("#settings .drawer-head .icon-btn").focus();
}
function closeSettings() {
  $("#settings").hidden = true;
  $("#scrim").hidden = true;
  if (lastFocus) lastFocus.focus();
}
$("#scrim").addEventListener("click", closeSettings);

function openHelp() { lastFocus = document.activeElement; $("#help").hidden = false; $("#help .drawer-head .icon-btn").focus(); }
function closeHelp() {
  $("#help").hidden = true;
  if (!prefs.helpSeen) { prefs.helpSeen = true; store(PREFS_KEY, prefs); }
  if (lastFocus && lastFocus !== document.body) lastFocus.focus();
}
$("#help").addEventListener("click", (e) => { if (e.target.id === "help") closeHelp(); });

// ------------------------------------------------------------------ prefs
const systemDark = matchMedia("(prefers-color-scheme: dark)");
function applyPrefs() {
  const theme = prefs.theme === "system" ? (systemDark.matches ? "dark" : "light") : prefs.theme;
  document.documentElement.dataset.theme = theme;
  $("#btn-theme use").setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
  document.documentElement.style.setProperty("--t-size", `${SIZES[prefs.size] || 30}px`);
  $("#transcript-text").className = "transcript-text font-" + prefs.font;
  $("#transcript-edit").className = "transcript-edit font-" + prefs.font;
  document.body.className = "layout-" + prefs.layout;
  const cap = prefs.layout === "captions";
  const lb = $("#btn-layout");
  lb.setAttribute("aria-pressed", String(cap));
  lb.querySelector("use").setAttribute("href", cap ? "#i-layout" : "#i-captions");
  lb.title = cap ? "Studio view (V)" : "Captions view (V)";
  lb.setAttribute("aria-label", cap ? "Switch to studio view" : "Switch to captions view");
  for (const group of $$("[data-pref]")) {
    for (const b of $$("button", group)) b.setAttribute("aria-pressed", String(b.dataset.value === prefs[group.dataset.pref]));
  }
  for (const c of $$("[data-pref-check]")) c.checked = !!prefs[c.dataset.prefCheck];
  shownText = null;
  if (S) render();
}
function setPref(key, value) { prefs[key] = value; store(PREFS_KEY, prefs); applyPrefs(); }
systemDark.addEventListener("change", applyPrefs);
for (const group of $$("[data-pref]")) {
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setPref(group.dataset.pref, b.dataset.value);
  });
}
for (const c of $$("[data-pref-check]")) c.addEventListener("change", () => setPref(c.dataset.prefCheck, c.checked));

// ------------------------------------------------------------------ tabs
const TABS = ["activity", "signs", "practice"];
function selectTab(name, focus = false) {
  for (const t of TABS) {
    const tab = $(`#tab-${t}`);
    tab.setAttribute("aria-selected", String(t === name));
    tab.tabIndex = t === name ? 0 : -1;
    $(`#panel-${t}`).hidden = t !== name;
  }
  if (focus) $(`#tab-${name}`).focus();
  prefs.tab = name; store(PREFS_KEY, prefs);
  if (name === "practice") renderPracticeSetup();
}
for (const t of TABS) $(`#tab-${t}`).addEventListener("click", () => selectTab(t));
$(".tabs").addEventListener("keydown", (e) => {
  const i = TABS.indexOf(prefs.tab);
  if (e.key === "ArrowRight") selectTab(TABS[(i + 1) % TABS.length], true);
  else if (e.key === "ArrowLeft") selectTab(TABS[(i + TABS.length - 1) % TABS.length], true);
  else return;
  e.preventDefault();
});

// ------------------------------------------------------------------ signs
const lookupUrl = (text) => `https://www.signasl.org/sign/${encodeURIComponent(text.toLowerCase().replace(/\s+/g, "-"))}`;

async function loadVocab() {
  try { vocab = await (await fetch("/api/vocab")).json(); }
  catch { vocab = []; }
  renderSigns();
}
let signFilter = "all";
function renderSigns() {
  const q = $("#sign-search").value.trim().toLowerCase();
  const list = vocab.filter((v) => (!q || v.label.toLowerCase().includes(q) || v.name.includes(q))
    && (signFilter === "all" || (signFilter === "two") === v.two_handed));
  $("#sign-count").textContent = vocab.length ? `${list.length} of ${vocab.length}` : "";
  const grid = $("#sign-grid");
  if (!vocab.length) {
    grid.innerHTML = `<li class="empty"><p>No word model is loaded, so there are no signs to list.</p></li>`;
    return;
  }
  grid.replaceChildren(...list.map((v) => {
    const li = document.createElement("li");
    li.className = "sign";
    const b = document.createElement("button");
    b.className = "insert";
    b.dataset.action = "insert-word";
    b.dataset.name = v.name;
    b.title = `Type “${v.text}”`;
    const name = document.createElement("span");
    name.textContent = v.label;
    b.append(name);
    if (v.two_handed) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "2H"; t.title = "Two-handed"; b.append(t); }
    const a = document.createElement("a");
    a.href = lookupUrl(v.text);
    a.target = "_blank"; a.rel = "noopener noreferrer";
    a.title = `How to sign “${v.label}” (SignASL.org)`;
    a.setAttribute("aria-label", a.title);
    a.innerHTML = `<svg class="ic"><use href="#i-book"/></svg>`;
    li.append(b, a);
    return li;
  }));
  if (!list.length) grid.innerHTML = `<li class="empty"><p>No signs match “${q.replace(/[<>&"]/g, "")}”.</p></li>`;
}
$("#sign-search").addEventListener("input", renderSigns);
$("#sign-filter").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  signFilter = b.dataset.value;
  $$("#sign-filter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  renderSigns();
});

// ------------------------------------------------------------------ practice
const P = { running: false, kind: prefs.practiceKind, pool: prefs.practicePool, target: null, tries: 0, correct: 0, streak: 0, wait: 0 };
const stats = load(STATS_KEY, {});
const statKey = (kind, item) => `${kind[0]}:${item}`;

function practiceItems(kind, pool) {
  let items = kind === "letters"
    ? LETTERS.concat(S && S.letters.motion_available && S.settings.motion_letters ? ["J", "Z"] : [])
    : vocab.filter((v) => pool !== "two" || v.two_handed).map((v) => v.name);
  if (pool === "weak") {
    const weak = items.filter((i) => { const s = stats[statKey(kind, i)]; return s && s.a >= 1 && s.c / s.a < 0.6; });
    if (weak.length >= 3) items = weak;
    else toast("Not enough misses yet, so practising everything");
  }
  return items;
}

function practiceLabel(kind, item) {
  if (kind === "letters") return item;
  const v = vocab.find((x) => x.name === item);
  return v ? v.label : item;
}

function nextTarget() {
  const items = practiceItems(P.kind, P.pool === "weak" ? "all" : P.pool);
  const weak = P.pool === "weak" ? P.items : null;
  const pool = weak && weak.length ? weak : items;
  let t;
  do { t = pool[Math.floor(Math.random() * pool.length)]; } while (pool.length > 1 && t === P.target);
  P.target = t;
  P.since = lastEvent;
  clearTimeout(P.wait);
  const card = $("#practice-target-card");
  card.classList.remove("correct", "wrong");
  $("#practice-target").textContent = practiceLabel(P.kind, t);
  $("#practice-result").textContent = P.kind === "letters" ? "Hold the letter until it's recognised." : "Raise your hands, sign, then lower them.";
  const lookup = $("#practice-lookup");
  const text = P.kind === "letters" ? `${t.toLowerCase()}` : (vocab.find((x) => x.name === t) || { text: t }).text;
  lookup.href = P.kind === "letters" ? `https://www.signasl.org/sign/letter-${text}` : lookupUrl(text);
  renderScore();
}

function renderScore() {
  $("#practice-correct").textContent = P.correct;
  $("#practice-tries").textContent = P.tries;
  $("#practice-streak").textContent = P.streak;
  $("#practice-best").textContent = prefs.bestStreak || 0;
}

async function startPractice() {
  if (!S) return;
  if (P.kind === "words" && !S.words.available) return toast("No word model is loaded");
  if (P.kind === "words" && !vocab.length) await loadVocab();
  P.items = P.pool === "weak" ? practiceItems(P.kind, "weak") : null;
  if (S.mode !== P.kind) await cmd("mode", { mode: P.kind });
  await cmd("practice", { on: true });
  P.running = true; P.tries = P.correct = P.streak = 0;
  $("#practice-setup").hidden = true;
  $("#practice-run").hidden = false;
  selectTab("practice");
  nextTarget();
}

async function stopPractice() {
  P.running = false;
  clearTimeout(P.wait);
  await cmd("practice", { on: false });
  $("#practice-setup").hidden = false;
  $("#practice-run").hidden = true;
  renderPracticeSetup();
  if (P.tries) toast(`Practice done: ${P.correct} of ${P.tries} correct`);
}

function practiceEvent(e) {
  if (e.id <= P.since || !P.target) return;
  let seen, top;
  if (P.kind === "letters" && e.kind === "letter") { seen = e.letter; top = [{ name: e.letter, label: e.letter, p: e.confidence }]; }
  else if (P.kind === "words" && e.kind === "word" && e.top.length) { seen = e.top[0].name; top = e.top; }
  else return;
  const k = statKey(P.kind, P.target);
  const s = stats[k] || (stats[k] = { a: 0, c: 0 });
  s.a += 1; P.tries += 1;
  const card = $("#practice-target-card");
  if (seen === P.target) {
    s.c += 1; P.correct += 1; P.streak += 1;
    if (P.streak > (prefs.bestStreak || 0)) { prefs.bestStreak = P.streak; store(PREFS_KEY, prefs); }
    card.classList.remove("wrong"); card.classList.add("correct");
    $("#practice-result").textContent = `Correct, ${Math.round(top[0].p * 100)}% sure`;
    P.since = Infinity;                       // ignore stragglers until the next prompt
    P.wait = setTimeout(nextTarget, 1300);
  } else {
    P.streak = 0;
    card.classList.remove("correct"); card.classList.add("wrong");
    const rank = top.findIndex((c) => c.name === P.target);
    $("#practice-result").textContent = `It saw “${top[0].label}” (${Math.round(top[0].p * 100)}%)`
      + (rank > 0 ? `. Yours was its #${rank + 1} guess, so you're close.` : ". Try again.");
  }
  store(STATS_KEY, stats);
  renderScore();
}

function renderPracticeSetup() {
  $$("#practice-kind button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.value === P.kind));
    if (b.dataset.value === "words") b.disabled = !!S && !S.words.available;
  });
  $$("#practice-pool button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.value === P.pool));
    if (b.dataset.value === "two") b.disabled = P.kind === "letters";
  });
  const entries = Object.entries(stats).filter(([k]) => k.startsWith(P.kind[0] + ":"));
  const box = $("#practice-stats");
  if (!entries.length) { box.innerHTML = `<p class="muted small">Your results will show up here.</p>`; return; }
  const a = entries.reduce((n, [, s]) => n + s.a, 0);
  const c = entries.reduce((n, [, s]) => n + s.c, 0);
  const weak = entries.filter(([, s]) => s.a >= 1 && s.c / s.a < 0.6)
    .sort((x, y) => x[1].c / x[1].a - y[1].c / y[1].a || y[1].a - x[1].a).slice(0, 12);
  box.innerHTML = `<h4>Your ${P.kind}</h4><p class="small"><strong></strong> correct over <strong></strong> tries on <strong></strong> ${P.kind}.`
    + ` <button class="btn btn-ghost btn-sm" data-action="practice-reset">Reset history</button></p>`
    + (weak.length ? `<h4>Needs work</h4><div class="weak-list"></div>` : "");
  const strongs = $$("strong", box);
  strongs[0].textContent = `${Math.round((100 * c) / a)}%`;
  strongs[1].textContent = a;
  strongs[2].textContent = entries.length;
  const wl = $(".weak-list", box);
  if (wl) wl.replaceChildren(...weak.map(([k, s]) => {
    const chip = document.createElement("span");
    chip.className = "chip chip-warn";
    chip.textContent = `${practiceLabel(P.kind, k.slice(2))} ${s.c}/${s.a}`;
    return chip;
  }));
}
$("#practice-kind").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || b.disabled) return;
  P.kind = prefs.practiceKind = b.dataset.value;
  if (P.kind === "letters" && P.pool === "two") P.pool = prefs.practicePool = "all";
  store(PREFS_KEY, prefs);
  renderPracticeSetup();
});
$("#practice-pool").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || b.disabled) return;
  P.pool = prefs.practicePool = b.dataset.value;
  store(PREFS_KEY, prefs);
  renderPracticeSetup();
});

// ------------------------------------------------------------------ toasts
function toast(message, action) {
  const t = document.createElement("div");
  t.className = "toast";
  const span = document.createElement("span");
  span.textContent = message;
  t.append(span);
  if (action) {
    const b = document.createElement("button");
    b.textContent = action.label;
    b.addEventListener("click", () => { action.fn(); dismiss(); });
    t.append(b);
  }
  const box = $("#toasts");
  box.append(t);
  while (box.children.length > 3) box.firstChild.remove();
  const dismiss = () => { t.classList.add("out"); setTimeout(() => t.remove(), 200); };
  setTimeout(dismiss, action ? 5000 : 2600);
}

// ------------------------------------------------------------------ actions
async function setMode(mode) {
  if (!S || S.mode === mode) return;
  if (mode === "words" && !S.words.available) return toast("No word model is loaded");
  shownText = null;
  await cmd("mode", { mode });
  if (P.running && P.kind !== mode) { P.kind = mode; P.items = null; nextTarget(); }
}

function chooseCandidate(i) {
  if (!S || S.mode !== "words" || i >= S.words.candidates.length) return;
  if (S.practice) return toast("Practice mode: nothing is added to your text");
  cmd("choose", { index: i });
}

async function clearText() {
  if (!S || !S.transcript.text) return;
  await cmd("clear");
  toast("Text cleared", { label: "Undo", fn: () => cmd("undo") });
}

const ACTIONS = {
  undo: () => cmd("undo"), redo: () => cmd("redo"),
  space: () => cmd("space"), newline: () => cmd("newline"),
  backspace: () => cmd("backspace"), delete_word: () => cmd("delete_word"),
  punct: (el) => cmd("punct", { char: el.dataset.char }),
  copy: copyText, download: downloadText, clear: clearText,
  speak: () => speak(plainText()),
  edit: () => (editing ? finishEdit() : startEdit()), "edit-done": finishEdit,
  "add-letter": () => cmd("add_letter"),
  sign: () => cmd("sign"), "cancel-sign": () => cmd("cancel_sign"),
  choose: (el) => chooseCandidate(Number(el.dataset.index)),
  "insert-word": (el) => cmd("insert_word", { name: el.dataset.name }),
  resume: () => cmd("pause", { paused: false }),
  "open-settings": openSettings, "close-settings": closeSettings, "close-help": closeHelp,
  "reset-settings": async () => { await cmd("settings", { reset: true }); toast("Recognition settings reset"); },
  "practice-start": startPractice, "practice-stop": stopPractice, "practice-skip": () => nextTarget(),
  "practice-reset": () => {
    for (const k of Object.keys(stats)) if (k.startsWith(P.kind[0] + ":")) delete stats[k];
    store(STATS_KEY, stats);
    renderPracticeSetup();
  },
};

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (el && ACTIONS[el.dataset.action]) {
    ACTIONS[el.dataset.action](el);
    if (e.detail > 0 && el.tagName === "BUTTON") el.blur();   // so Space/Enter keep meaning "sign" after a mouse click
  }
  const pop = $("#display-popover");
  if (!pop.hidden && !e.target.closest(".popover-wrap")) togglePopover(false);
});

$$("#mode-switch button").forEach((b) => b.addEventListener("click", (e) => { setMode(b.dataset.mode); if (e.detail > 0) b.blur(); }));
$("#btn-pause").addEventListener("click", (e) => { cmd("pause", { paused: !(S && S.paused) }); if (e.detail > 0) e.currentTarget.blur(); });
$("#btn-theme").addEventListener("click", () => setPref("theme", document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
$("#btn-layout").addEventListener("click", () => setPref("layout", prefs.layout === "captions" ? "studio" : "captions"));
$("#btn-help").addEventListener("click", openHelp);
$("#btn-settings").addEventListener("click", openSettings);
function togglePopover(open) {
  const pop = $("#display-popover");
  pop.hidden = !open;
  $("#btn-display").setAttribute("aria-expanded", String(open));
}
$("#btn-display").addEventListener("click", () => togglePopover($("#display-popover").hidden));

function pressKey(selector) {
  const el = $(selector);
  if (!el) return;
  el.classList.add("pressed");
  setTimeout(() => el.classList.remove("pressed"), 120);
}

document.addEventListener("keydown", (e) => {
  if (e.defaultPrevented) return;                 // e.g. arrows already used by the tabs
  const t = e.target;
  const typing = t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable;
  const ctrl = e.ctrlKey || e.metaKey;

  if (e.key === "Escape") {
    if (!$("#help").hidden) closeHelp();
    else if (!$("#settings").hidden) closeSettings();
    else if (!$("#display-popover").hidden) togglePopover(false);
    else if (editing) finishEdit();
    else if (S && S.mode === "words" && S.words.state !== "idle") cmd("cancel_sign");
    else return;
    e.preventDefault();
    return;
  }
  if (editing && ctrl && e.key === "Enter") { e.preventDefault(); finishEdit(); return; }
  if (ctrl && e.key === ",") { e.preventDefault(); openSettings(); return; }
  if (typing || !$("#help").hidden || !$("#settings").hidden || !S) return;
  if ((e.key === "Enter" || e.key === " ") && (t.tagName === "BUTTON" || t.tagName === "A")) return;

  if (ctrl) {
    const k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) cmd("undo");
    else if ((k === "z" && e.shiftKey) || k === "y") cmd("redo");
    else if (e.key === "Delete") clearText();
    else if (e.key === "ArrowLeft") moveCursor(wordJump(cursorPos(), -1));
    else if (e.key === "ArrowRight") moveCursor(wordJump(cursorPos(), 1));
    else return;
    e.preventDefault();
    return;
  }
  if (e.altKey) return;

  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  switch (k) {
    case "l": setMode("letters"); break;
    case "w": setMode("words"); break;
    case "Enter":
      if (e.shiftKey) { cmd("newline"); pressKey('[data-action="newline"]'); }
      else if (S.mode === "letters") cmd("add_letter");
      else cmd("sign");
      break;
    case " ": cmd("space"); pressKey('[data-action="space"]'); break;
    case "Backspace":
      if (e.shiftKey) { cmd("delete_word"); pressKey('[data-action="delete_word"]'); }
      else { cmd("backspace"); pressKey('[data-action="backspace"]'); }
      break;
    case ".": case ",": case "?": case "!":
      cmd("punct", { char: k }); pressKey(`[data-char="${k}"]`); break;
    case "1": case "2": case "3": chooseCandidate(Number(k) - 1); break;
    case "c": copyText(); break;
    case "s": speak(plainText()); break;
    case "e": startEdit(); break;
    case "p": cmd("pause", { paused: !S.paused }); break;
    case "v": setPref("layout", prefs.layout === "captions" ? "studio" : "captions"); break;
    case "h": openHelp(); break;
    case "Tab": if (!P.running) return; nextTarget(); break;
    case "ArrowLeft": moveCursor(cursorPos() - 1); break;
    case "ArrowRight": moveCursor(cursorPos() + 1); break;
    case "Home": moveCursor(0); break;
    case "End": moveCursor(null); break;
    default: return;
  }
  e.preventDefault();
});

// Keep keyboard focus inside an open dialog.
document.addEventListener("keydown", (e) => {
  if (e.key !== "Tab") return;
  const dlg = !$("#help").hidden ? $("#help .modal") : !$("#settings").hidden ? $("#settings") : null;
  if (!dlg) return;
  const f = $$("button, input, select, a[href]", dlg).filter((x) => !x.disabled && x.offsetParent);
  if (!f.length) return;
  if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
  else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
});

// ------------------------------------------------------------------ boot
applyPrefs();
selectTab(TABS.includes(prefs.tab) ? prefs.tab : "activity");
renderScore();
if (!prefs.helpSeen) openHelp();
connect();
