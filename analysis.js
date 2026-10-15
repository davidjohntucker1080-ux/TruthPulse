// Turns recorded face movements into numbers, trains a simple model, and tests it fairly.
//
// The rules that keep this honest:
// 1. Each person's measurements are compared against that same person's own average
//    (some people just blink more). This uses no truth/lie labels, so it can't cheat.
// 2. The model is always tested on people it was NOT trained on (leave-one-person-out).
// 3. Results are compared against a coin flip (50%) with a binomial test.

export const MIN_PER_PERSON = 4;   // answers a person needs before they count
const MIN_FACE_FRAMES = 10;        // frames with a visible face for an answer to count

const sig = s => 1 / (1 + Math.exp(-s));
const clamp = v => Math.max(-5, Math.min(5, v));

function sd(values) {
  if (!values.length) return 0;
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, v) => a + (v - m) ** 2, 0) / values.length);
}

function pretty(name) {
  const s = name.replace(/([A-Z])/g, " $1").toLowerCase().trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function featureLabels(names) {
  const labels = [];
  for (let i = 1; i < names.length; i++) labels.push(pretty(names[i]) + ", average");
  for (let i = 1; i < names.length; i++) labels.push(pretty(names[i]) + ", variability");
  labels.push("Blink rate", "Head turning", "Head nodding", "Head tilting",
    "Head movement speed", "Answer length", "Time face was out of view");
  return labels;
}

// One answer -> one row of numbers. Order must match featureLabels().
export function features(r) {
  if (!r || !Array.isArray(r.frames) || !Array.isArray(r.names)) return null;
  const K = r.names.length;
  const valid = r.frames.filter(f => Array.isArray(f.b) && f.b.length === K);
  if (valid.length < MIN_FACE_FRAMES) return null;
  const n = valid.length;

  const sum = new Array(K).fill(0), sq = new Array(K).fill(0);
  for (const f of valid) for (let i = 0; i < K; i++) { sum[i] += f.b[i]; sq[i] += f.b[i] * f.b[i]; }
  const means = [], sds = [];
  for (let i = 1; i < K; i++) {         // index 0 is "_neutral", skip it
    const m = sum[i] / n;
    means.push(m);
    sds.push(Math.sqrt(Math.max(0, sq[i] / n - m * m)));
  }
  const x = means.concat(sds);

  const secs = Math.max(0.5, (r.durationMs || 0) / 1000);
  const iL = r.names.indexOf("eyeBlinkLeft"), iR = r.names.indexOf("eyeBlinkRight");
  let blinks = 0, closed = false;
  for (const f of valid) {
    const e = ((f.b[iL] || 0) + (f.b[iR] || 0)) / 2;
    if (!closed && e > 0.5) { blinks++; closed = true; }
    else if (closed && e < 0.3) closed = false;
  }
  x.push(blinks / secs);

  for (let a = 0; a < 3; a++) x.push(sd(valid.map(f => (f.p ? f.p[a] : 0))));

  let speed = 0;
  for (let k = 1; k < n; k++) {
    const dt = Math.max(10, valid[k].t - valid[k - 1].t) / 1000;
    const p0 = valid[k - 1].p || [0, 0, 0], p1 = valid[k].p || [0, 0, 0];
    speed += (Math.abs(p1[0] - p0[0]) + Math.abs(p1[1] - p0[1])) / dt;
  }
  x.push(speed / Math.max(1, n - 1));
  x.push(secs);
  x.push(1 - n / r.frames.length);
  return x;
}

// Group usable answers by person and z-score each person against themselves.
export function dataset(rounds) {
  const groups = new Map();
  for (const r of rounds) {
    if (r.instruction !== "lie" && r.instruction !== "truth") continue;
    const x = features(r);
    if (!x) continue;
    if (!groups.has(r.person)) groups.set(r.person, []);
    groups.get(r.person).push({ round: r, person: r.person, x, y: r.instruction === "lie" ? 1 : 0 });
  }
  const rows = [], people = [];
  let dim = null, names = null;
  for (const [person, g] of groups) {
    if (g.length < MIN_PER_PERSON) continue;
    if (dim === null) { dim = g[0].x.length; names = g[0].round.names; }
    const same = g.filter(r => r.x.length === dim);
    if (same.length < MIN_PER_PERSON) continue;
    normalize(same);
    rows.push(...same);
    people.push(person);
  }
  return { rows, people, dim, names };
}

function normalize(g) {
  const p = g[0].x.length, n = g.length;
  g.forEach(r => { r.z = new Array(p); });
  for (let j = 0; j < p; j++) {
    let m = 0;
    for (const r of g) m += r.x[j];
    m /= n;
    let v = 0;
    for (const r of g) v += (r.x[j] - m) ** 2;
    const s = Math.sqrt(v / n);
    for (const r of g) r.z[j] = s > 1e-6 ? clamp((r.x[j] - m) / s) : 0;
  }
}

// Regularized logistic regression. Strong regularization because we'll have
// far fewer answers than measurements for a long time.
export function train(rows, lambda = 0.1, iters = 250, lr = 0.3) {
  const p = rows[0].z.length, n = rows.length;
  const w = new Float64Array(p);
  let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = new Float64Array(p);
    let gb = 0;
    for (const r of rows) {
      let s = b;
      for (let j = 0; j < p; j++) s += w[j] * r.z[j];
      const e = sig(s) - r.y;
      gb += e;
      for (let j = 0; j < p; j++) gw[j] += e * r.z[j];
    }
    for (let j = 0; j < p; j++) w[j] -= lr * (gw[j] / n + lambda * w[j]);
    b -= lr * gb / n;
  }
  return { w: Array.from(w), b, dim: p };
}

export function prob(model, z) {
  let s = model.b;
  for (let j = 0; j < model.dim; j++) s += model.w[j] * z[j];
  return sig(s);
}

// One-sided binomial test: chance of getting k or more right out of n by coin flips.
export function binomP(k, n) {
  if (n === 0) return 1;
  const lf = [0];
  for (let i = 1; i <= n; i++) lf[i] = lf[i - 1] + Math.log(i);
  let s = 0;
  for (let i = k; i <= n; i++) s += Math.exp(lf[n] - lf[i] - lf[n - i] - n * Math.LN2);
  return Math.min(1, s);
}

function hasBoth(rows) { return rows.some(r => r.y === 1) && rows.some(r => r.y === 0); }

// Leave-one-person-out: train on everyone else, test on the person left out.
export function evaluate(ds) {
  const eligible = ds.people.filter(p => hasBoth(ds.rows.filter(r => r.person === p)));
  if (eligible.length < 3) return { status: "need-data", people: eligible.length };
  let right = 0, n = 0;
  const perPerson = [];
  for (const p of eligible) {
    const trainRows = ds.rows.filter(r => r.person !== p);
    if (!hasBoth(trainRows)) continue;
    const m = train(trainRows);
    let pr = 0, pn = 0;
    for (const r of ds.rows.filter(r => r.person === p)) {
      pn++;
      if ((prob(m, r.z) >= 0.5 ? 1 : 0) === r.y) pr++;
    }
    right += pr; n += pn;
    perPerson.push({ person: p, right: pr, n: pn });
  }
  return { status: "ok", right, n, acc: n ? right / n : 0, p: binomP(right, n), people: eligible.length, perPerson };
}

// Model for live guesses during a game, trained on everyone except the current answerer.
export function trainForLive(otherRounds) {
  const ds = dataset(otherRounds);
  if (ds.people.length < 2 || ds.rows.length < 12 || !hasBoth(ds.rows)) return null;
  return train(ds.rows);
}

// Probability (0-1) that one answer is a lie, or null if there isn't enough to go on yet.
export function predictOne(model, personRounds, round) {
  if (!model) return null;
  const ds = dataset(personRounds);
  if (ds.dim !== model.dim) return null;
  const row = ds.rows.find(r => r.round.id === round.id);
  return row ? prob(model, row.z) : null;
}

// Which single measurements differ most between lies and truths (Cohen's d).
export function signals(ds, top = 6) {
  if (ds.rows.length < 20 || !hasBoth(ds.rows)) return [];
  const labels = featureLabels(ds.names);
  const lies = ds.rows.filter(r => r.y === 1), truths = ds.rows.filter(r => r.y === 0);
  const out = [];
  for (let j = 0; j < ds.dim; j++) {
    const a = lies.map(r => r.z[j]), b = truths.map(r => r.z[j]);
    const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
    const pooled = Math.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2);
    if (pooled < 1e-6) continue;
    out.push({ label: labels[j], d: (ma - mb) / pooled });
  }
  return out.sort((p, q) => Math.abs(q.d) - Math.abs(p.d)).slice(0, top);
}

export function humanStats(rounds) {
  const guessed = rounds.filter(r => r.guess === "lie" || r.guess === "truth");
  const right = guessed.filter(r => r.guess === r.instruction).length;
  return { right, n: guessed.length, acc: guessed.length ? right / guessed.length : 0, p: binomP(right, guessed.length) };
}

export function report(rounds) {
  const ds = dataset(rounds);
  const people = new Set(rounds.map(r => r.person));
  return {
    total: rounds.length,
    usable: ds.rows.length,
    people: people.size,
    countedPeople: ds.people.length,
    lies: ds.rows.filter(r => r.y === 1).length,
    truths: ds.rows.filter(r => r.y === 0).length,
    human: humanStats(rounds),
    model: evaluate(ds),
    signals: signals(ds)
  };
}
