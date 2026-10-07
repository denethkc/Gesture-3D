import './style.css';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { OneEuro, damp, clamp, clamp01 } from './smooth.js';

/* =====================================================================
   CONFIG: tweak these to change the feel
   ===================================================================== */
const CONFIG = {
  modelSize: 2.4, // model is scaled so its biggest side equals this
  camZ: 6,
  filter: { minCutoff: 1.4, beta: 7 }, // One Euro: lower minCutoff = smoother, higher beta = less lag
  follow: 12, // how fast the model follows the hand (higher = snappier)
  rotate: 9,
  zoom: 9,
  explode: 8,
  yawGain: 1.2, // flip the sign if twisting your hand turns the model the wrong way
  pitchGain: 1.0,
  pinchOn: 0.3,
  pinchOff: 0.45,
  swipe: { dist: 0.32, windowMs: 450, cooldownMs: 1300 },
  maxLabels: 8,
};

/* Put your .glb files in /public/models/ and list them here */
const MODELS = [
  { name: 'Ferrari', group: 'Machines', file: '/models/ferrari.glb', blurb: 'Body, chassis, drivetrain and interior.' },
  { name: 'Time Machine', group: 'Machines', file: '/models/time_machine.glb', blurb: 'Dials, frame and mechanism of a time machine.' },
  { name: 'Mech Design', group: 'Machines', file: '/models/zbrush_for_concept_-_mech_design_d_ver.glb', blurb: 'Concept mech sculpt, split into its parts.' },
];

const HAND_LINKS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

/* =====================================================================
   DOM
   ===================================================================== */
const $ = (s) => document.querySelector(s);
const stage = $('#stage');
const video = $('#cam');
const hud = $('#hud');
const hctx = hud.getContext('2d');
const linesSvg = $('#lines');
const tagsEl = $('#tags');
const panel = $('#panel');
const toast = $('#toast');
const ui = {
  track: $('#stTrack'), hands: $('#stHands'),
  exRange: $('#exRange'), exVal: $('#exVal'), inRange: $('#inRange'), inVal: $('#inVal'),
  title: $('#mTitle'), blurb: $('#mBlurb'), group: $('#chipGroup'), parts: $('#chipParts'),
  info: $('#info'), infoTitle: $('#infoTitle'), infoText: $('#infoText'),
  gestures: [...document.querySelectorAll('#gestures li')],
  models: $('#models'), spin: $('#btnSpin'),
};

let toastTimer;
function say(msg, ms = 2200) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  if (ms) toastTimer = setTimeout(() => toast.classList.remove('show'), ms);
}

/* =====================================================================
   THREE SCENE
   ===================================================================== */
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.insertBefore(renderer.domElement, linesSvg);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.z = CONFIG.camZ;

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(3, 4, 5);
scene.add(key);
scene.add(new THREE.AmbientLight(0xffffff, 0.25));

const rig = new THREE.Group(); // moved / rotated / scaled by the hand
rig.rotation.order = 'YXZ';
scene.add(rig);

function resize() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const dpr = Math.min(window.devicePixelRatio, 2);
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  hud.width = w * dpr;
  hud.height = h * dpr;
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);
resize();

const halfH = () => Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * CONFIG.camZ;

/* =====================================================================
   STATE
   ===================================================================== */
const S = {
  t: { px: 0, py: 0, rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 }, // targets (from the hand)
  cur: { px: 0, py: 0, rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 }, // what is drawn (eased)
  intensity: 1,
  spin: 0,
  autoSpin: false,
  hands: [],
  handCount: 0,
  lastSeen: 0,
  gesture: '',
  gestureUntil: 0,
  pinching: false,
  drag: null,
  twoStart: null,
  swipeHist: [],
  swipeCooldown: 0,
  openPrev: 0,
  current: null,
  index: -1,
  selected: null,
  loadToken: 0,
  sliderActive: false,
};

const filters = [];
const openFilter = new OneEuro(1.2, 2);

/* =====================================================================
   MODELS
   ===================================================================== */
const loader = new GLTFLoader();
const cache = new Map();
const tmpBox = new THREE.Box3();

const cleanName = (n) => n.replace(/[_.\-]+/g, ' ').replace(/\s*\d+$/, '').trim() || 'Part';

function prepare(gltf) {
  const root = gltf.scene;
  const wrap = new THREE.Group();
  wrap.add(root);
  wrap.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  root.position.sub(center);
  wrap.updateMatrixWorld(true);

  const parts = [];
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.material = Array.isArray(o.material) ? o.material.map((m) => m.clone()) : o.material.clone();
    o.geometry.computeBoundingBox();
    tmpBox.setFromObject(o);
    const wc = tmpBox.getCenter(new THREE.Vector3());
    const sz = tmpBox.getSize(new THREE.Vector3());

    const len = wc.length();
    const dir = len > 1e-6 ? wc.clone().divideScalar(len) : new THREE.Vector3(0, 1, 0);
    const dist = maxDim * 0.9 * Math.min(1, 0.35 + len / (maxDim * 0.5));
    const parent = o.parent;
    const full = parent.worldToLocal(wc.clone().addScaledVector(dir, dist)).sub(parent.worldToLocal(wc.clone()));

    const mats = (Array.isArray(o.material) ? o.material : [o.material])
      .filter((m) => m.emissive)
      .map((m) => ({ m, c: m.emissive.clone(), i: m.emissiveIntensity }));

    parts.push({
      mesh: o,
      name: cleanName(o.name || parent.name || ''),
      base: o.position.clone(),
      full,
      localCenter: o.geometry.boundingBox.getCenter(new THREE.Vector3()),
      vol: sz.x * sz.y * sz.z,
      drag: new THREE.Vector3(),
      dragTarget: new THREE.Vector3(),
      mats,
    });
  });

  wrap.scale.setScalar(CONFIG.modelSize / maxDim);
  const byMesh = new Map(parts.map((p) => [p.mesh, p]));
  const top = [...parts].sort((a, b) => b.vol - a.vol).slice(0, CONFIG.maxLabels);
  return { wrap, parts, byMesh, top };
}

async function loadModel(i) {
  const token = ++S.loadToken;
  const def = MODELS[i];
  say(`Loading ${def.name}…`, 0);
  try {
    let entry = cache.get(i);
    if (!entry) {
      const gltf = await loader.loadAsync(def.file);
      entry = prepare(gltf);
      cache.set(i, entry);
    }
    if (token !== S.loadToken) return;

    if (S.current) rig.remove(S.current.wrap);
    select(null);
    S.current = entry;
    S.index = i;
    rig.add(entry.wrap);
    S.t.explode = 0;
    S.cur.explode = 0;
    buildLabels(entry);

    ui.title.textContent = def.name;
    ui.blurb.textContent = def.blurb;
    ui.group.textContent = def.group;
    ui.parts.textContent = `${entry.parts.length} parts`;
    [...ui.models.querySelectorAll('button')].forEach((b) => b.classList.toggle('on', Number(b.dataset.i) === i));
    toast.classList.remove('show');
  } catch (err) {
    console.error(err);
    say(`Could not load ${def.file}. Check public/models/`, 4000);
  }
}

function buildModelBar() {
  const groups = new Map();
  MODELS.forEach((m, i) => {
    if (!groups.has(m.group)) groups.set(m.group, []);
    groups.get(m.group).push(i);
  });
  groups.forEach((idx, name) => {
    const g = document.createElement('div');
    g.className = 'grp';
    g.innerHTML = `<span>${name}</span>`;
    idx.forEach((i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.i = i;
      b.textContent = MODELS[i].name;
      b.addEventListener('click', () => loadModel(i));
      g.appendChild(b);
    });
    ui.models.appendChild(g);
  });
}

const nextModel = () => loadModel((S.index + 1) % MODELS.length);

/* =====================================================================
   PARTS: explode, select, highlight, pick
   ===================================================================== */
const inv = new THREE.Matrix4();
const v0 = new THREE.Vector3();
const v1 = new THREE.Vector3();

function dragToLocal(p) {
  inv.copy(p.mesh.parent.matrixWorld).invert();
  v1.copy(p.drag).applyMatrix4(inv);
  v0.set(0, 0, 0).applyMatrix4(inv);
  return v1.sub(v0);
}

function applyExplode(dt) {
  if (!S.current) return;
  const k = S.cur.explode * S.intensity;
  for (const p of S.current.parts) {
    p.mesh.position.copy(p.base).addScaledVector(p.full, k);
    const moving = p.dragTarget.lengthSq() > 0 || p.drag.lengthSq() > 1e-8;
    if (moving) {
      p.drag.x = damp(p.drag.x, p.dragTarget.x, 14, dt);
      p.drag.y = damp(p.drag.y, p.dragTarget.y, 14, dt);
      p.drag.z = damp(p.drag.z, p.dragTarget.z, 14, dt);
      if (p.drag.lengthSq() < 1e-8 && p.dragTarget.lengthSq() === 0) p.drag.set(0, 0, 0);
      else p.mesh.position.add(dragToLocal(p));
    }
  }
}

function setGlow(p, on) {
  if (!p) return;
  p.mats.forEach(({ m, c, i }) => {
    if (on) {
      m.emissive.set(0xffb800);
      m.emissiveIntensity = 0.55;
    } else {
      m.emissive.copy(c);
      m.emissiveIntensity = i;
    }
  });
}

function select(p) {
  if (S.selected === p) return;
  setGlow(S.selected, false);
  S.selected = p;
  setGlow(p, true);
  ui.info.classList.toggle('has', !!p);
  if (p) {
    const n = S.current.parts.indexOf(p) + 1;
    ui.infoTitle.textContent = p.name;
    ui.infoText.textContent = `Part ${n} of ${S.current.parts.length}. Pinch it to pull it out.`;
  } else {
    ui.infoTitle.textContent = 'No part selected';
    ui.infoText.textContent = 'Point at a part, or click one, to see its name.';
  }
}

const tmpV = new THREE.Vector3();
function projectPart(p, out) {
  tmpV.copy(p.localCenter);
  p.mesh.localToWorld(tmpV);
  tmpV.project(camera);
  out.x = (tmpV.x + 1) / 2;
  out.y = (1 - tmpV.y) / 2;
  return out;
}

function nearestPart(sx, sy, maxDist) {
  if (!S.current) return null;
  rig.updateMatrixWorld(true);
  const aspect = stage.clientWidth / stage.clientHeight;
  const out = { x: 0, y: 0 };
  let best = null;
  let bestD = maxDist;
  for (const p of S.current.parts) {
    projectPart(p, out);
    const d = Math.hypot((out.x - sx) * aspect, out.y - sy);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

const ray = new THREE.Raycaster();
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!S.current) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera({ x: ((e.clientX - r.left) / r.width) * 2 - 1, y: -((e.clientY - r.top) / r.height) * 2 + 1 }, camera);
  const hit = ray.intersectObjects(S.current.parts.map((p) => p.mesh), false)[0];
  select(hit ? S.current.byMesh.get(hit.object) : null);
});

/* =====================================================================
   LABELS with leader lines
   ===================================================================== */
let labels = [];
function buildLabels(entry) {
  tagsEl.innerHTML = '';
  linesSvg.innerHTML = '';
  labels = entry.top.map((p) => {
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = p.name;
    tagsEl.appendChild(tag);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('r', '2.5');
    linesSvg.append(line, dot);
    return { p, tag, line, dot, w: tag.offsetWidth, h: tag.offsetHeight };
  });
}

function updateLabels() {
  const show = S.cur.explode > 0.12 && labels.length > 0;
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  labels.forEach((l) => {
    l.tag.style.opacity = show ? 1 : 0;
    l.line.style.display = l.dot.style.display = show ? '' : 'none';
  });
  if (!show) return;

  rig.updateMatrixWorld(true);
  const pt = { x: 0, y: 0 };
  const items = labels.map((l) => {
    projectPart(l.p, pt);
    return { l, x: pt.x * sw, y: pt.y * sh };
  });
  items.sort((a, b) => a.x - b.x);
  const half = Math.ceil(items.length / 2);
  const left = items.slice(0, half).sort((a, b) => a.y - b.y);
  const right = items.slice(half).sort((a, b) => a.y - b.y);

  const leftX = panel.getBoundingClientRect().right + 28;
  const rightX = sw - 28;
  const y0 = sh * 0.2;
  const y1 = sh * 0.78;
  const place = (arr, side) =>
    arr.forEach((it, k) => {
      const y = arr.length === 1 ? (y0 + y1) / 2 : y0 + ((y1 - y0) * k) / (arr.length - 1);
      const { l } = it;
      const tx = side === 'L' ? leftX : rightX - l.w;
      l.tag.style.transform = `translate(${tx}px, ${y - l.h / 2}px)`;
      const ex = side === 'L' ? tx + l.w + 4 : tx - 4;
      const sel = S.selected === l.p;
      l.tag.classList.toggle('sel', sel);
      l.line.classList.toggle('sel', sel);
      l.line.setAttribute('x1', ex);
      l.line.setAttribute('y1', y);
      l.line.setAttribute('x2', it.x);
      l.line.setAttribute('y2', it.y);
      l.dot.setAttribute('cx', it.x);
      l.dot.setAttribute('cy', it.y);
    });
  place(left, 'L');
  place(right, 'R');
}

/* =====================================================================
   HAND TRACKING
   ===================================================================== */
let landmarker = null;

async function startTracking() {
  ui.track.textContent = 'Starting…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    ui.track.textContent = 'No camera';
    ui.track.className = 'off';
    say('Camera blocked. Allow camera access and reload.', 5000);
    return;
  }

  const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm');
  const make = (delegate) =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate,
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
  try {
    landmarker = await make('GPU');
  } catch {
    landmarker = await make('CPU');
  }
  ui.track.textContent = 'Show your hand';
  ui.track.className = 'off';
}

/* video pixel -> normalised stage coords (handles object-fit: cover and the mirror) */
function toScreen(p) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  const s = Math.max(sw / vw, sh / vh);
  const ox = (sw - vw * s) / 2;
  const oy = (sh - vh * s) / 2;
  return { x: 1 - (p.x * vw * s + ox) / sw, y: (p.y * vh * s + oy) / sh };
}

function worldFromScreen(sx, sy) {
  const hh = halfH();
  const hw = hh * camera.aspect;
  return { x: (sx - 0.5) * 2 * hw, y: -(sy - 0.5) * 2 * hh };
}

function getFilters(h) {
  if (!filters[h]) {
    filters[h] = Array.from({ length: 21 }, () =>
      [0, 1, 2].map(() => new OneEuro(CONFIG.filter.minCutoff, CONFIG.filter.beta)),
    );
  }
  return filters[h];
}

function smoothHand(lm, h, t) {
  const f = getFilters(h);
  return lm.map((p, j) => ({
    x: f[j][0].filter(p.x, t),
    y: f[j][1].filter(p.y, t),
    z: f[j][2].filter(p.z, t),
  }));
}

const dd = (a, b) => {
  const ar = (video.videoWidth || 16) / (video.videoHeight || 9);
  return Math.hypot((a.x - b.x) * ar, a.y - b.y);
};

function palmCenter(lm) {
  let x = 0;
  let y = 0;
  [0, 5, 9, 13, 17].forEach((i) => {
    const s = toScreen(lm[i]);
    x += s.x;
    y += s.y;
  });
  return { x: x / 5, y: y / 5 };
}

function releasePinch() {
  if (S.drag) S.drag.part.dragTarget.set(0, 0, 0);
  S.drag = null;
  S.pinching = false;
}

function flash(g, ms = 700) {
  S.gesture = g;
  S.gestureUntil = performance.now() + ms;
}

function processHands(result, now) {
  const raw = (result.landmarks || []).slice(0, 2);
  raw.sort((a, b) => a[0].x - b[0].x);

  if (raw.length !== S.handCount) {
    filters.length = 0;
    S.handCount = raw.length;
    S.swipeHist.length = 0;
    S.twoStart = null;
    releasePinch();
  }
  S.hands = raw.map((l, i) => smoothHand(l, i, now));
  if (S.hands.length) S.lastSeen = now;

  ui.hands.textContent = S.hands.length;
  ui.track.textContent = S.hands.length ? 'Hands on' : 'Show your hand';
  ui.track.className = S.hands.length ? 'on' : 'off';

  const T = S.t;
  const H = S.hands;

  if (H.length === 2) {
    /* ---- two hands: zoom + move ---- */
    const a = palmCenter(H[0]);
    const b = palmCenter(H[1]);
    const aspect = stage.clientWidth / stage.clientHeight;
    const d = Math.hypot((a.x - b.x) * aspect, a.y - b.y);
    if (!S.twoStart) S.twoStart = { d, scale: T.scale };
    T.scale = clamp((S.twoStart.scale * d) / Math.max(S.twoStart.d, 0.05), 0.35, 3.2);
    const w = worldFromScreen((a.x + b.x) / 2, (a.y + b.y) / 2);
    T.px = w.x;
    T.py = w.y;
    if (!S.gestureUntil || performance.now() > S.gestureUntil) S.gesture = 'two';
    releasePinch();
    return;
  }

  S.twoStart = null;
  if (H.length !== 1) return;

  /* ---- one hand ---- */
  const lm = H[0];
  const palm = Math.max(dd(lm[0], lm[9]), 0.01);
  const c = palmCenter(lm);
  const world = worldFromScreen(c.x, c.y);

  // pinch (with hysteresis)
  const pinchD = dd(lm[4], lm[8]) / palm;
  const pinchPt = toScreen({ x: (lm[4].x + lm[8].x) / 2, y: (lm[4].y + lm[8].y) / 2 });
  if (!S.pinching && pinchD < CONFIG.pinchOn) {
    S.pinching = true;
    const part = nearestPart(pinchPt.x, pinchPt.y, 0.16);
    if (part) {
      select(part);
      S.drag = { part, start: worldFromScreen(pinchPt.x, pinchPt.y) };
    }
  } else if (S.pinching && pinchD > CONFIG.pinchOff) {
    releasePinch();
  }

  if (S.pinching) {
    if (S.drag) {
      const w = worldFromScreen(pinchPt.x, pinchPt.y);
      S.drag.part.dragTarget.set(w.x - S.drag.start.x, w.y - S.drag.start.y, 0);
    }
    S.gesture = 'pinch';
    return; // model stays still while a part is being pulled
  }

  // pointing: index out, other fingers curled
  const ext = (tip, pip) => dd(lm[tip], lm[0]) > dd(lm[pip], lm[0]) * 1.08;
  const pointing = ext(8, 6) && !ext(12, 10) && !ext(16, 14) && !ext(20, 18);
  if (pointing) {
    const tip = toScreen(lm[8]);
    const part = nearestPart(tip.x, tip.y, 0.13);
    if (part) select(part);
    S.gesture = 'point';
    T.px = world.x;
    T.py = world.y;
    return;
  }

  // move
  T.px = world.x;
  T.py = world.y;

  // twist / tilt from the palm orientation
  const sgn = lm[5].x < lm[17].x ? 1 : -1;
  const wx = Math.abs(lm[17].x - lm[5].x) + 0.001;
  T.rz = clamp(Math.atan2((lm[17].y - lm[5].y) * sgn, wx), -1.4, 1.4);
  T.ry = clamp(Math.atan2((lm[17].z - lm[5].z) * sgn, wx) * CONFIG.yawGain, -1.5, 1.5);
  T.rx = clamp(Math.atan2(lm[9].z - lm[0].z, dd(lm[0], lm[9]) + 0.001) * CONFIG.pitchGain, -1.1, 1.1);

  // explode from how open the hand is
  const tips = [8, 12, 16, 20];
  const avg = tips.reduce((s, i) => s + dd(lm[i], lm[0]), 0) / 4;
  const open = clamp01(openFilter.filter(clamp01((avg / palm - 1.15) / 0.8), now));
  T.explode = open;
  const rate = (open - S.openPrev) / 0.033;
  S.openPrev = open;
  if (performance.now() > S.gestureUntil) {
    S.gesture = rate > 0.7 ? 'open' : rate < -0.7 ? 'close' : 'move';
  }

  // swipe right-to-left = next model
  S.swipeHist.push({ t: now, x: c.x });
  while (S.swipeHist.length && now - S.swipeHist[0].t > CONFIG.swipe.windowMs) S.swipeHist.shift();
  if (S.swipeHist.length > 3 && now > S.swipeCooldown && open > 0.35) {
    const dx = c.x - S.swipeHist[0].x;
    if (dx < -CONFIG.swipe.dist) {
      S.swipeCooldown = now + CONFIG.swipe.cooldownMs;
      S.swipeHist.length = 0;
      flash('swipe');
      nextModel();
    }
  }
}

/* =====================================================================
   HUD: hand skeleton + openness ring
   ===================================================================== */
function drawHud() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  hctx.clearRect(0, 0, w, h);
  for (const lm of S.hands) {
    const pts = lm.map((p) => {
      const s = toScreen(p);
      return { x: s.x * w, y: s.y * h };
    });
    hctx.lineWidth = 1.6;
    hctx.strokeStyle = 'rgba(127, 227, 255, 0.85)';
    hctx.beginPath();
    HAND_LINKS.forEach(([a, b]) => {
      hctx.moveTo(pts[a].x, pts[a].y);
      hctx.lineTo(pts[b].x, pts[b].y);
    });
    hctx.stroke();
    hctx.fillStyle = '#fff';
    pts.forEach((p) => {
      hctx.beginPath();
      hctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
      hctx.fill();
    });

    const cx = (pts[0].x + pts[9].x) / 2;
    const cy = (pts[0].y + pts[9].y) / 2;
    const r = Math.hypot(pts[0].x - pts[9].x, pts[0].y - pts[9].y) * 1.6;
    hctx.lineWidth = 2;
    hctx.strokeStyle = 'rgba(255,255,255,0.18)';
    hctx.beginPath();
    hctx.arc(cx, cy, r, 0, Math.PI * 2);
    hctx.stroke();
    hctx.strokeStyle = '#ffc83a';
    hctx.lineWidth = 3;
    hctx.beginPath();
    hctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + S.cur.explode * Math.PI * 2);
    hctx.stroke();
  }
}

/* =====================================================================
   MAIN LOOP
   ===================================================================== */
let last = performance.now();
let lastVideoTime = -1;

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;

  // run detection only when the camera produced a new frame
  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    processHands(landmarker.detectForVideo(video, now), now);
  }

  const T = S.t;
  const C = S.cur;
  if (!S.hands.length && now - S.lastSeen > 1500) {
    T.px = 0;
    T.py = 0;
    T.rx = 0;
    T.rz = 0;
  }
  C.px = damp(C.px, T.px, CONFIG.follow, dt);
  C.py = damp(C.py, T.py, CONFIG.follow, dt);
  C.rx = damp(C.rx, T.rx, CONFIG.rotate, dt);
  C.ry = damp(C.ry, T.ry, CONFIG.rotate, dt);
  C.rz = damp(C.rz, T.rz, CONFIG.rotate, dt);
  C.scale = damp(C.scale, T.scale, CONFIG.zoom, dt);
  C.explode = damp(C.explode, T.explode, CONFIG.explode, dt);

  if (S.autoSpin && !S.hands.length) S.spin += dt * 0.5;

  rig.position.set(C.px, C.py, 0);
  rig.rotation.set(C.rx, C.ry + S.spin, C.rz);
  rig.scale.setScalar(C.scale);
  applyExplode(dt);

  if (!S.sliderActive) {
    ui.exRange.value = C.explode;
    ui.exVal.textContent = `${Math.round(C.explode * 100)}%`;
  }

  ui.gestures.forEach((li) => li.classList.toggle('active', li.dataset.g === S.gesture && S.hands.length > 0));

  renderer.render(scene, camera);
  updateLabels();
  drawHud();
}

/* =====================================================================
   UI WIRING
   ===================================================================== */
ui.exRange.addEventListener('pointerdown', () => (S.sliderActive = true));
window.addEventListener('pointerup', () => (S.sliderActive = false));
ui.exRange.addEventListener('input', () => {
  S.t.explode = Number(ui.exRange.value);
  ui.exVal.textContent = `${Math.round(S.t.explode * 100)}%`;
});
ui.inRange.addEventListener('input', () => {
  S.intensity = Number(ui.inRange.value);
  ui.inVal.textContent = `${Math.round(S.intensity * 100)}%`;
});
$('#btnAssemble').addEventListener('click', () => (S.t.explode = 0));
ui.spin.addEventListener('click', () => {
  S.autoSpin = !S.autoSpin;
  ui.spin.classList.toggle('on', S.autoSpin);
});
$('#btnFull').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
});
$('#btnSnap').addEventListener('click', () => {
  const w = renderer.domElement.width;
  const h = renderer.domElement.height;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');
  const vw = video.videoWidth || w;
  const vh = video.videoHeight || h;
  const s = Math.max(w / vw, h / vh);
  g.save();
  g.translate(w, 0);
  g.scale(-1, 1); // mirror like the live view
  g.drawImage(video, (w - vw * s) / 2, (h - vh * s) / 2, vw * s, vh * s);
  g.restore();
  renderer.render(scene, camera);
  g.drawImage(renderer.domElement, 0, 0);
  const a = document.createElement('a');
  a.download = `gesture-3d-${Date.now()}.png`;
  a.href = c.toDataURL('image/png');
  a.click();
  say('Snapshot saved');
});

/* keyboard fallback for testing without a camera */
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') nextModel();
  if (e.key === 'e') S.t.explode = S.t.explode > 0.5 ? 0 : 1;
});

buildModelBar();
loadModel(0);
requestAnimationFrame(loop);
startTracking();
