import { FaceLandmarker, FilesetResolver } from "./vendor/mediapipe/vision_bundle.mjs";
import { QUESTIONS } from "./questions.js";
import * as A from "./analysis.js";
import * as DB from "./db.js";

const $ = id => document.getElementById(id);
const video = $("video"), cam = $("cam"), badge = $("faceBadge");
const CAM_SCREENS = new Set(["setup", "record"]);
const MAX_ANSWER_MS = 45000;

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
const personKey = s => s.trim().toLowerCase().replace(/\s+/g, " ");
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = x => Math.round(x * 100) + "%";
const local = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} }
};

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

let currentScreen = "home";
function show(name) {
  currentScreen = name;
  document.querySelectorAll("[data-screen]").forEach(s => { s.hidden = s.dataset.screen !== name; });
  cam.hidden = !(CAM_SCREENS.has(name) && stream);
  if (!cam.hidden) video.play().catch(() => {});
  window.scrollTo(0, 0);
}

/* ---------------- Camera and face tracking ---------------- */

let landmarker = null, stream = null, running = false, lastVideoTime = -1;
let names = null, IDX = {};
let face = null, faceSeen = null;
let rec = null;   // active recording: { start, last, frames }

const g = (b, k) => (IDX[k] === undefined ? 0 : b[IDX[k]]);

async function initTracker() {
  if (landmarker) return landmarker;
  const files = await FilesetResolver.forVisionTasks(new URL("./vendor/mediapipe/wasm", location.href).href);
  const opts = delegate => ({
    baseOptions: { modelAssetPath: new URL("./models/face_landmarker.task", location.href).href, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true
  });
  try { landmarker = await FaceLandmarker.createFromOptions(files, opts("GPU")); }
  catch { landmarker = await FaceLandmarker.createFromOptions(files, opts("CPU")); }
  return landmarker;
}

async function startCamera() {
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play().catch(() => {});
  }
  await initTracker();
  if (!running) { running = true; requestAnimationFrame(loop); }
}

// Head angles in degrees from the face's 4x4 transform (column-major).
function pose(m) {
  if (!m || !m.data) return [0, 0, 0];
  const d = m.data, r = (i, j) => d[j * 4 + i];
  const deg = x => Math.round(x * 1800 / Math.PI) / 10;
  return [
    deg(Math.asin(Math.max(-1, Math.min(1, -r(2, 0))))),   // yaw: turning left/right
    deg(Math.atan2(r(2, 1), r(2, 2))),                      // pitch: nodding
    deg(Math.atan2(r(1, 0), r(0, 0)))                       // roll: tilting
  ];
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    try { handle(landmarker.detectForVideo(video, now), now); } catch (e) { console.warn(e); }
  }
  requestAnimationFrame(loop);
}

function handle(res, now) {
  const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
  if (cats && cats.length) {
    if (!names) { names = cats.map(c => c.categoryName); names.forEach((n, i) => { IDX[n] = i; }); }
    face = {
      b: cats.map(c => Math.round(c.score * 1000) / 1000),
      p: pose(res.facialTransformationMatrixes && res.facialTransformationMatrixes[0])
    };
  } else {
    face = null;
  }
  if (faceSeen !== !!face) {
    faceSeen = !!face;
    badge.textContent = face ? "Face found" : "No face in view";
    badge.classList.toggle("ok", !!face);
  }
  if (rec && now - rec.last >= 30) {
    rec.last = now;
    const t = Math.round(now - rec.start);
    rec.frames.push(face ? { t, b: face.b, p: face.p } : { t, miss: 1 });
    live(now);
  }
}

/* ---------------- Live readings during an answer ---------------- */

const METERS = [
  ["Blink", b => (g(b, "eyeBlinkLeft") + g(b, "eyeBlinkRight")) / 2],
  ["Eyes away", b => Math.max(
    (g(b, "eyeLookOutLeft") + g(b, "eyeLookInRight")) / 2,
    (g(b, "eyeLookInLeft") + g(b, "eyeLookOutRight")) / 2,
    (g(b, "eyeLookUpLeft") + g(b, "eyeLookUpRight")) / 2,
    (g(b, "eyeLookDownLeft") + g(b, "eyeLookDownRight")) / 2)],
  ["Lip press", b => Math.min(1, (g(b, "mouthPressLeft") + g(b, "mouthPressRight")) / 2 + g(b, "mouthRollLower") * 0.5)],
  ["Brow raise", b => g(b, "browInnerUp")],
  ["Smile", b => (g(b, "mouthSmileLeft") + g(b, "mouthSmileRight")) / 2]
];
const meterFills = [];
(function buildMeters() {
  const box = $("meters");
  METERS.forEach(([label]) => {
    const row = document.createElement("div");
    row.className = "meter";
    row.innerHTML = `<span>${label}</span><div class="track"><div class="fill"></div></div>`;
    box.appendChild(row);
    meterFills.push(row.querySelector(".fill"));
  });
})();

const trace = $("trace"), tctx = trace.getContext("2d");
let traceVals = [], prevB = null, lastPaint = 0;

function live(now) {
  if (face) {
    let energy = 0;
    if (prevB) for (let i = 1; i < face.b.length; i++) energy += Math.abs(face.b[i] - prevB[i]);
    prevB = face.b;
    traceVals.push(Math.min(1, energy / 1.5));
  } else {
    prevB = null;
    traceVals.push(0);
  }
  if (traceVals.length > 240) traceVals.shift();
  if (now - lastPaint < 60) return;
  lastPaint = now;
  if (face) METERS.forEach(([, fn], i) => { meterFills[i].style.width = Math.round(Math.min(1, fn(face.b)) * 100) + "%"; });
  drawTrace();
}

function drawTrace() {
  const dpr = window.devicePixelRatio || 1;
  const w = trace.clientWidth, h = trace.clientHeight;
  if (trace.width !== Math.round(w * dpr)) { trace.width = Math.round(w * dpr); trace.height = Math.round(h * dpr); }
  const css = getComputedStyle(document.documentElement);
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tctx.clearRect(0, 0, w, h);
  tctx.strokeStyle = css.getPropertyValue("--rule");
  tctx.lineWidth = 1;
  for (let y = h / 4; y < h; y += h / 4) { tctx.beginPath(); tctx.moveTo(0, y); tctx.lineTo(w, y); tctx.stroke(); }
  tctx.strokeStyle = css.getPropertyValue("--lie");
  tctx.lineWidth = 2;
  tctx.lineJoin = "round";
  tctx.beginPath();
  const step = w / 239;
  traceVals.forEach((v, i) => {
    const x = w - (traceVals.length - 1 - i) * step;
    const y = h - 8 - v * (h - 16);
    i ? tctx.lineTo(x, y) : tctx.moveTo(x, y);
  });
  tctx.stroke();
}

/* ---------------- Home ---------------- */

async function renderHome() {
  let rounds = [];
  try { rounds = await DB.all(); } catch {}
  const people = new Set(rounds.map(r => r.person)).size;
  const usable = A.dataset(rounds).rows.length;
  $("homeStats").innerHTML =
    `<div class="stat"><b>${rounds.length}</b><span>answers recorded</span></div>` +
    `<div class="stat"><b>${people}</b><span>people tested</span></div>` +
    `<div class="stat"><b>${usable}</b><span>ready for analysis</span></div>`;
  show("home");
}

$("goHome").onclick = () => { if (!rec) renderHome(); };
$("startBtn").onclick = () => {
  $("answerer").value = local.get("tell.answerer", "");
  $("guesser").value = local.get("tell.guesser", "");
  updateBegin();
  show("setup");
};
$("labBtn").onclick = renderLab;

$("exportBtn").onclick = async () => {
  const rounds = await DB.all();
  if (!rounds.length) { $("dataMsg").textContent = "Nothing to export yet. Play a game first."; return; }
  const blob = new Blob([JSON.stringify({ app: "tell", version: 1, exportedAt: new Date().toISOString(), rounds })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "tell-data-" + new Date().toISOString().slice(0, 10) + ".json";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  $("dataMsg").textContent = `Exported ${rounds.length} answers.`;
};

$("importFile").onchange = async e => {
  let added = 0, bad = 0;
  for (const file of e.target.files) {
    try {
      const data = JSON.parse(await file.text());
      const rounds = (data && Array.isArray(data.rounds) ? data.rounds : []).filter(r => r && r.id && Array.isArray(r.frames) && r.person);
      if (!rounds.length) { bad++; continue; }
      await DB.putMany(rounds);
      added += rounds.length;
    } catch { bad++; }
  }
  e.target.value = "";
  $("dataMsg").textContent = (added ? `Imported ${added} answers.` : "No answers were imported.") + (bad ? ` ${bad} file(s) weren't Tell data.` : "");
  renderHome();
  $("dataBox").open = true;
};

let wipeArmed = null;
$("wipeBtn").onclick = async () => {
  const btn = $("wipeBtn");
  if (!wipeArmed) {
    btn.textContent = "Tap again to delete everything";
    wipeArmed = setTimeout(() => { btn.textContent = "Delete all data"; wipeArmed = null; }, 4000);
    return;
  }
  clearTimeout(wipeArmed); wipeArmed = null;
  btn.textContent = "Delete all data";
  await DB.clear();
  $("dataMsg").textContent = "All data deleted from this phone.";
  renderHome();
  $("dataBox").open = true;
};

/* ---------------- Setup ---------------- */

let roundsChoice = 10, cameraReady = false;
$("roundsSeg").querySelectorAll("button").forEach(b => {
  b.onclick = () => {
    roundsChoice = Number(b.dataset.n);
    $("roundsSeg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  };
});

function updateBegin() {
  const a = $("answerer").value.trim(), gg = $("guesser").value.trim();
  const ok = a && gg && personKey(a) !== personKey(gg) && $("consent").checked && cameraReady;
  $("beginBtn").disabled = !ok;
  if (a && gg && personKey(a) === personKey(gg)) $("setupMsg").textContent = "The answerer and guesser need different names.";
}
["answerer", "guesser"].forEach(id => $(id).addEventListener("input", updateBegin));
$("consent").addEventListener("change", updateBegin);

$("camBtn").onclick = async () => {
  const btn = $("camBtn"), msg = $("setupMsg");
  btn.disabled = true;
  msg.textContent = "Starting the camera and loading face tracking. The first time takes a few seconds…";
  try {
    await startCamera();
    cameraReady = true;
    cam.hidden = false;
    btn.textContent = "Camera on";
    msg.textContent = "Make sure the answerer's face shows \"Face found\" before you begin.";
  } catch (e) {
    console.error(e);
    btn.disabled = false;
    msg.textContent = e && e.name === "NotAllowedError"
      ? "Camera access was blocked. In iPhone Settings, go to Safari, then Camera, and choose Allow. Then tap Turn on camera again."
      : e && (e.name === "NotFoundError" || e.name === "OverconstrainedError")
        ? "No front camera was found on this device."
        : "The camera or face tracking couldn't start. Close other apps using the camera, reload the page, and try again.";
  }
  updateBegin();
};

$("beginBtn").onclick = () => newGame($("answerer").value.trim(), $("guesser").value.trim(), roundsChoice);

/* ---------------- Game ---------------- */

let game = null;

async function newGame(answerer, guesser, total) {
  local.set("tell.answerer", answerer);
  local.set("tell.guesser", guesser);
  const all = await DB.all().catch(() => []);
  const person = personKey(answerer);
  const asked = new Set(all.filter(r => r.person === person).map(r => r.questionId));
  let qs = shuffle(QUESTIONS.filter(q => !asked.has(q.id)));
  if (qs.length < total) qs = qs.concat(shuffle(QUESTIONS.filter(q => asked.has(q.id))));
  const instructions = shuffle(Array.from({ length: total }, (_, i) => (i % 2 ? "lie" : "truth")));
  game = {
    id: uid(), answerer, guesser, person, total, i: 0,
    plan: qs.slice(0, total).map((q, i) => ({ q, instruction: instructions[i] })),
    human: { right: 0, n: 0 }, ai: { right: 0, n: 0 },
    model: A.trainForLive(all.filter(r => r.person !== person)),
    current: null
  };
  goHandoff();
}

function goHandoff() {
  const { answerer, guesser, i, total } = game;
  $("handoffCount").textContent = `Question ${i + 1} of ${total}`;
  $("handoffTitle").textContent = `Hand the phone to ${answerer}`;
  $("handoffText").textContent = `${guesser}, don't look at the screen. You'll get the phone back to make your guess.`;
  $("revealBtn").textContent = `I'm ${answerer}. Show my instruction`;
  show("handoff");
}

$("revealBtn").onclick = () => {
  const step = game.plan[game.i];
  const w = $("instrWord");
  w.textContent = step.instruction === "lie" ? "LIE" : "TRUTH";
  w.className = "verdict " + step.instruction;
  $("instrHow").textContent = step.instruction === "lie"
    ? "Make up a believable answer that isn't true. Keep a straight face."
    : "Answer honestly, the way you normally would.";
  $("instrQ").textContent = step.q.text;
  show("instruction");
};

let recTimer = null;
$("recBtn").onclick = () => {
  const step = game.plan[game.i];
  $("recQ").textContent = step.q.text;
  traceVals = []; prevB = null;
  meterFills.forEach(f => { f.style.width = "0"; });
  rec = { start: performance.now(), last: 0, frames: [] };
  show("record");
  drawTrace();
  recTimer = setInterval(() => {
    const ms = performance.now() - rec.start;
    $("recTime").textContent = (ms / 1000).toFixed(1) + " s";
    if (ms >= MAX_ANSWER_MS) finishAnswer();
  }, 100);
};
$("doneBtn").onclick = () => finishAnswer();

function finishAnswer() {
  if (!rec) return;
  clearInterval(recTimer);
  const r = rec; rec = null;
  const step = game.plan[game.i];
  game.current = {
    id: uid(), app: "tell", v: 1,
    sessionId: game.id, createdAt: new Date().toISOString(),
    person: game.person, answerer: game.answerer, guesser: personKey(game.guesser),
    questionId: step.q.id, question: step.q.text,
    instruction: step.instruction, guess: null,
    durationMs: Math.round(performance.now() - r.start),
    names, frames: r.frames,
    device: navigator.userAgent
  };
  const seen = r.frames.filter(f => f.b).length;
  const warn = $("guessWarn");
  warn.hidden = seen >= 10;
  warn.textContent = "Tell couldn't see the face well during that answer, so it won't count toward the experiment.";
  $("guessTitle").textContent = `${game.guesser}, your call`;
  $("guessText").textContent = `Take the phone back. Was ${game.answerer} telling the truth or lying?`;
  show("guess");
}

$("gTruth").onclick = () => onGuess("truth");
$("gLie").onclick = () => onGuess("lie");
$("gSkip").onclick = () => onGuess(null);

async function onGuess(guess) {
  const r = game.current;
  r.guess = guess;
  try { await DB.put(r); } catch (e) { console.error(e); }

  const truthIs = r.instruction;
  const w = $("revWord");
  w.textContent = truthIs === "lie" ? "A LIE" : "THE TRUTH";
  w.className = "verdict " + truthIs;

  $("revHumanWho").textContent = game.guesser;
  const hb = $("revHuman");
  if (guess) {
    game.human.n++;
    const ok = guess === truthIs;
    if (ok) game.human.right++;
    hb.innerHTML = `Guessed ${guess} · <span class="${ok ? "right" : "wrong"}">${ok ? "right" : "wrong"}</span>`;
  } else {
    hb.textContent = "No guess";
  }

  const ab = $("revAI");
  if (!game.model) {
    ab.textContent = "Still learning. It needs games from at least 2 other people first.";
  } else {
    const personRounds = (await DB.all().catch(() => [])).filter(x => x.person === game.person);
    const p = A.predictOne(game.model, personRounds, r);
    if (p === null) {
      ab.textContent = `Warming up. It needs ${A.MIN_PER_PERSON} clear answers from ${game.answerer} first.`;
    } else {
      const call = p >= 0.5 ? "lie" : "truth";
      const ok = call === truthIs;
      game.ai.n++;
      if (ok) game.ai.right++;
      const sure = Math.round(Math.max(p, 1 - p) * 100);
      ab.innerHTML = `Guessed ${call} (${sure}% sure) · <span class="${ok ? "right" : "wrong"}">${ok ? "right" : "wrong"}</span>`;
    }
  }

  const parts = [];
  if (game.human.n) parts.push(`${esc(game.guesser)} ${game.human.right}/${game.human.n}`);
  if (game.ai.n) parts.push(`Tell ${game.ai.right}/${game.ai.n}`);
  $("revScore").innerHTML = parts.join(" &nbsp;·&nbsp; ");
  $("nextBtn").textContent = game.i + 1 < game.total ? "Next question" : "See results";
  show("reveal");
}

$("nextBtn").onclick = () => {
  game.i++;
  if (game.i < game.total) goHandoff();
  else showSummary();
};

function showSummary() {
  const h = game.human, a = game.ai;
  $("sumCards").innerHTML =
    `<div class="card"><span class="who">${esc(game.guesser)}</span><span class="big">${h.n ? pct(h.right / h.n) : "–"}</span><span class="hint">${h.right} of ${h.n} right</span></div>` +
    `<div class="card"><span class="who">Tell</span><span class="big">${a.n ? pct(a.right / a.n) : "–"}</span><span class="hint">${a.n ? `${a.right} of ${a.n} right` : "Not guessing yet"}</span></div>`;
  $("sumText").textContent = "One game is far too small to prove anything either way. Every answer you play adds to the experiment, so swap roles and keep going.";
  $("swapBtn").textContent = `Swap: ${game.guesser} answers, ${game.answerer} guesses`;
  show("summary");
}
$("swapBtn").onclick = () => newGame(game.guesser, game.answerer, game.total);
$("sumLab").onclick = renderLab;
$("sumHome").onclick = renderHome;
$("labHome").onclick = renderHome;

/* ---------------- Lab results ---------------- */

function verdictText(m) {
  if (m.n < 40) return "Too few answers to tell skill from luck. Keep playing.";
  if (m.acc > 0.5 && m.p < 0.01) return "Tell is beating a coin flip on people it never trained on. That's a real signal. Confirm it with more new people before calling it a discovery.";
  if (m.acc > 0.5 && m.p < 0.05) return "Promising, but this could still be luck. Keep collecting.";
  return "So far Tell is no better than a coin flip on new people.";
}

async function renderLab() {
  const body = $("labBody");
  body.innerHTML = `<p class="hint">Crunching the numbers…</p>`;
  show("lab");
  await new Promise(r => setTimeout(r, 30));
  const rounds = await DB.all().catch(() => []);
  const rep = A.report(rounds);
  const h = rep.human, m = rep.model;

  let html = `<div class="panel"><h3>The data</h3>
    <div class="kv"><span>Answers recorded</span><b>${rep.total}</b></div>
    <div class="kv"><span>Usable answers</span><b>${rep.usable}</b></div>
    <div class="kv"><span>Lies / truths</span><b>${rep.lies} / ${rep.truths}</b></div>
    <div class="kv"><span>People tested</span><b>${rep.people}</b></div>
    <div class="kv"><span>People with enough answers</span><b>${rep.countedPeople}</b></div></div>`;

  html += `<div class="panel"><h3>People guessing</h3>`;
  html += h.n
    ? `<div class="bigline">${pct(h.acc)}</div><p>${h.right} of ${h.n} guesses right. A coin flip gets 50%. Chance of doing this well by luck: ${h.p < 0.001 ? "under 0.1%" : pct(h.p)}.</p>`
    : `<p>No guesses yet.</p>`;
  html += `</div>`;

  html += `<div class="panel"><h3>Tell, tested on people it never saw</h3>`;
  if (m.status === "need-data") {
    html += `<p>Tell needs at least 3 people with ${A.MIN_PER_PERSON} or more clear answers each, including both truths and lies, before it can be tested fairly. So far: ${m.people}.</p>`;
  } else {
    html += `<div class="bigline">${pct(m.acc)}</div><p>${m.right} of ${m.n} right across ${m.people} people. Chance of doing this well by luck: ${m.p < 0.001 ? "under 0.1%" : pct(m.p)}.</p><p><b>${verdictText(m)}</b></p>`;
  }
  html += `</div>`;

  if (rep.signals.length) {
    html += `<div class="panel"><h3>Biggest differences so far</h3><div class="sig">`;
    rep.signals.forEach(s => {
      html += `<span>${esc(s.label)}</span><span class="dir">${s.d > 0 ? "higher" : "lower"} when lying · ${Math.abs(s.d).toFixed(2)}</span>`;
    });
    html += `</div><p class="hint">Tell checks more than 100 measurements, so a few will look different by luck alone. Treat these as leads to test, not findings.</p></div>`;
  }

  html += `<div class="panel"><h3>How to read this</h3>
    <p>Each person is compared against their own normal face, so someone who always blinks a lot doesn't look like a liar. Tell is graded only on people it didn't learn from, which is the test that has tripped up earlier lie-detection research.</p>
    <p>To trust a result you'll want a few hundred answers from at least 10 to 20 different people.</p></div>`;
  body.innerHTML = html;
}

renderHome();
