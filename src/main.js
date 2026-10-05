import './style.css';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/* GLB files live in /public/models. Add your own here (they load in this order). */
const GLB_MODELS = [
  { url: '/models/engine.glb', name: 'Engine' },
  { url: '/models/ferrari.glb', name: 'Ferrari' },
  { url: '/models/ReciprocatingSaw.glb', name: 'Reciprocating Saw' },
  { url: '/models/GearboxAssy.glb', name: 'Gearbox' },
  { url: '/models/zbrush_for_concept_-_mech_design_d_ver.glb', name: 'Mech Design' },
  { url: '/models/time_machine.glb', name: 'Time Machine' },
];

/* ---- tuning knobs ---- */
const FOLLOW_MOVE = 0.7; // how far the model moves with your hand (0 = never, 1 = to the screen edge)
const FOLLOW_ROTATE = 1.0; // how much it turns with your hand (0 = never)
const SWIPE_DIST = 0.22; // swipe: hand travel as a fraction of camera width...
const SWIPE_WINDOW = 0.4; // ...within this many seconds
const SWIPE_COOLDOWN = 1.0; // seconds before the next swipe is accepted
const SWIPE_PREVIOUS = false; // true = left -> right swipe goes to the previous model
const ZOOM_MIN = 5;
const ZOOM_MAX = 22;
const ZOOM_START = 11;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const hud = document.getElementById('hud');
const labelsEl = document.getElementById('labels');
const cursorEl = document.getElementById('cursor');
const infoEl = document.getElementById('info');
const infoName = document.getElementById('infoName');
const infoDesc = document.getElementById('infoDesc');
const infoMeta = document.getElementById('infoMeta');

/* ---------------- Three.js scene (solid, lit, with reflections) ---------------- */
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0, ZOOM_START);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
document.getElementById('stage').appendChild(renderer.domElement);

// a built-in "studio" environment gives metals and paint realistic reflections
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(5, 8, 6);
scene.add(key);
const rim = new THREE.DirectionalLight(0x8fb4ff, 0.8);
rim.position.set(-6, 2, -5);
scene.add(rim);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const group = new THREE.Group(); // moved / rotated by your hand
scene.add(group);

/* ---------------- Names and descriptions ---------------- */
// "Piston_123-844_0_Parts_1" -> "Piston", "rim_fl" -> "Rim fl"
function cleanName(raw) {
  const n = (raw || '')
    .replace(/_+\d.*$/, '')
    .replace(/[_\-.]+/g, ' ')
    .replace(/\s*instance\s*\d*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!n || /^(mesh|node|object|group)\s*\d*$/i.test(n)) return '';
  return n.charAt(0).toUpperCase() + n.slice(1);
}

const DESCRIPTIONS = [
  [/piston/i, 'Slides up and down inside a cylinder, turning pressure into motion.'],
  [/^rod|conrod|connecting/i, 'Links the piston to the crankshaft and carries the force between them.'],
  [/spring/i, 'Stores energy and pushes a part back to its starting position.'],
  [/lifter|tappet/i, 'Follows the cam and opens the valve at exactly the right moment.'],
  [/valve/i, 'Opens and closes to let air and fuel in, or exhaust out.'],
  [/crank/i, 'Turns the up-and-down push of the pistons into rotation.'],
  [/cam/i, 'A shaped lobe that controls when valves open and close.'],
  [/gear|cog/i, 'Toothed wheel that passes on torque and changes speed.'],
  [/shaft|axle/i, 'A rotating bar that carries power from one part to another.'],
  [/bearing/i, 'Lets a shaft spin smoothly with very little friction.'],
  [/bolt|screw|nut|washer/i, 'A fastener that holds neighbouring parts together.'],
  [/tire|tyre/i, 'The rubber that grips the road and absorbs bumps.'],
  [/rim|wheel/i, 'Carries the tire and transfers drive power to the road.'],
  [/brake/i, 'Slows the wheel by squeezing it with friction pads.'],
  [/glass|window|windshield/i, 'Transparent panel that lets the driver see out.'],
  [/light|led/i, 'Lamp used for seeing, signalling or decoration.'],
  [/wiper/i, 'Sweeps rain off the windscreen.'],
  [/grill/i, 'Lets air flow in to cool the engine and brakes.'],
  [/leather|seat/i, 'Soft upholstery covering the seats and trim.'],
  [/steering/i, 'Part of the steering wheel and column that turns the front wheels.'],
  [/carpet/i, 'Floor covering that reduces noise and wear.'],
  [/chrome|metal/i, 'Polished metal trim.'],
  [/carbon/i, 'Light, very strong carbon-fibre panel.'],
  [/body|casing|housing|block/i, 'The main housing that holds the other parts in place.'],
];
const describe = (name, model) =>
  (DESCRIPTIONS.find(([re]) => re.test(name)) || [null, `A component of the ${model}.`])[1];

/* ---------------- Turning a .glb into explodable, selectable parts ---------------- */
const tmpM = new THREE.Matrix4();

function modelFromGLTF(modelName, gltf) {
  const root = gltf.scene;
  root.updateMatrixWorld(true);

  // centre the model and scale it to a fixed size
  const bounds = new THREE.Box3().setFromObject(root);
  const centre = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const s = 4.5 / Math.max(size.x, size.y, size.z);
  const norm = new THREE.Matrix4()
    .makeScale(s, s, s)
    .multiply(new THREE.Matrix4().makeTranslation(-centre.x, -centre.y, -centre.z));

  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && !o.isSkinnedMesh && o.geometry?.attributes?.position) meshes.push(o);
  });

  // every mesh becomes one part, moved on its own when exploding
  const modelGroup = new THREE.Group();
  const parts = [];
  const box = new THREE.Box3();
  const partOfMesh = new Map();

  meshes.forEach((mesh, i) => {
    const base = norm.clone().multiply(mesh.matrixWorld);
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    box.copy(mesh.geometry.boundingBox).applyMatrix4(base);
    const c = box.getCenter(new THREE.Vector3());
    const partSize = box.getSize(new THREE.Vector3()).length();

    const dir = c.clone();
    if (dir.length() < 0.05) dir.set(Math.cos(i * 2.1), Math.sin(i * 1.3), Math.sin(i * 2.1));
    dir.normalize();

    // own copy of the materials so we can highlight / dim a single part
    const wasArray = Array.isArray(mesh.material);
    const mats = (wasArray ? mesh.material : [mesh.material]).map((m) => m.clone());
    mesh.material = wasArray ? mats : mats[0];
    const orig = mats.map((m) => ({
      transparent: m.transparent,
      opacity: m.opacity,
      depthWrite: m.depthWrite,
      emissive: m.emissive ? m.emissive.clone() : null,
      emissiveIntensity: m.emissiveIntensity,
    }));

    const idx = mesh.geometry.index;
    const tris = Math.round((idx ? idx.count : mesh.geometry.attributes.position.count) / 3);

    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(base);
    mesh.matrixWorldNeedsUpdate = true;
    modelGroup.add(mesh);

    const part = { rawName: mesh.name, center: c, dir, mesh, base, mats, orig, tris, size: partSize };
    parts.push(part);
    partOfMesh.set(mesh, i);
  });

  // readable unique names ("Wheel 1", "Wheel 2"...) and descriptions
  const baseNames = parts.map((p, i) => cleanName(p.rawName) || `Part ${i + 1}`);
  const seen = {};
  baseNames.forEach((n) => (seen[n] = (seen[n] || 0) + 1));
  const used = {};
  const totalTris = parts.reduce((a, p) => a + p.tris, 0) || 1;
  parts.forEach((p, i) => {
    const n = baseNames[i];
    used[n] = (used[n] || 0) + 1;
    p.name = seen[n] === 1 ? n : `${n} ${used[n]}`;
    p.desc = describe(n, modelName);
    p.share = ((p.tris / totalTris) * 100).toFixed(1);
  });

  // labels: the biggest parts are shown by default, any part when selected
  const ranked = parts.map((p, i) => i).sort((a, b) => parts[b].size - parts[a].size);
  const eligible = new Set(ranked.slice(0, 12));
  const labels = parts.map((p, i) => {
    const el = document.createElement('div');
    el.className = 'label';
    el.textContent = p.name;
    el.style.display = 'none';
    labelsEl.appendChild(el);
    return { el, part: p, index: i, eligible: eligible.has(i) };
  });

  modelGroup.visible = false;
  group.add(modelGroup);
  return { name: modelName, group: modelGroup, parts, labels, partOfMesh, appliedExplode: null };
}

function applyExplode(model, v) {
  if (model.appliedExplode !== null && Math.abs(model.appliedExplode - v) < 1e-4) return;
  model.appliedExplode = v;
  for (const p of model.parts) {
    tmpM.makeTranslation(p.dir.x * v, p.dir.y * v, p.dir.z * v);
    p.mesh.matrix.multiplyMatrices(tmpM, p.base);
    p.mesh.matrixWorldNeedsUpdate = true;
  }
}

/* highlight one part and fade the others (idx < 0 restores everything) */
function applySel(model, idx) {
  model.parts.forEach((p, i) => {
    p.mats.forEach((m, k) => {
      const o = p.orig[k];
      const wasTransparent = m.transparent;
      m.transparent = o.transparent;
      m.opacity = o.opacity;
      m.depthWrite = o.depthWrite;
      if (m.emissive) {
        m.emissive.copy(o.emissive);
        m.emissiveIntensity = o.emissiveIntensity;
      }
      if (idx >= 0 && i === idx) {
        if (m.emissive) {
          m.emissive.set(0xffa733);
          m.emissiveIntensity = 0.55;
        }
      } else if (idx >= 0) {
        m.transparent = true;
        m.opacity = Math.min(o.opacity, 0.1);
        m.depthWrite = false;
      }
      if (m.transparent !== wasTransparent) m.needsUpdate = true;
    });
  });
}

/* ---------------- Model list, switching, selection ---------------- */
const models = [];
let current = 0;
let selected = -1;

function addModel(m) {
  models.push(m);
  if (models.length === 1) showModel(0);
}

function showModel(i) {
  if (models[current]) applySel(models[current], -1);
  selected = -1;
  infoEl.style.display = 'none';
  models.forEach((m, k) => {
    m.group.visible = k === i;
    m.labels.forEach((l) => (l.el.style.display = k === i ? '' : 'none'));
  });
  current = i;
  models[i].appliedExplode = null; // force re-apply the current explosion
}

let lastSwitch = 0;
function stepModel(dir = 1) {
  const now = performance.now();
  if (now - lastSwitch < 700 || models.length < 2) return;
  lastSwitch = now;
  showModel((current + dir + models.length) % models.length);
}

function select(i) {
  const m = models[current];
  if (!m) return;
  selected = i;
  applySel(m, i);
  if (i < 0) {
    infoEl.style.display = 'none';
    return;
  }
  const p = m.parts[i];
  infoName.textContent = p.name;
  infoDesc.textContent = p.desc;
  infoMeta.textContent = `${p.tris.toLocaleString()} triangles · ${p.share}% of the model`;
  infoEl.style.display = 'block';
}

/* where is a part on screen right now? */
const tmpV = new THREE.Vector3();
function projectPart(p) {
  tmpV.copy(p.center).addScaledVector(p.dir, state.explode);
  group.localToWorld(tmpV);
  tmpV.project(camera);
  return {
    x: (tmpV.x * 0.5 + 0.5) * innerWidth,
    y: (-tmpV.y * 0.5 + 0.5) * innerHeight,
    visible: tmpV.z < 1,
  };
}

function nearestPart(sx, sy, maxDist) {
  const m = models[current];
  if (!m) return -1;
  group.updateMatrixWorld();
  let best = -1;
  let bestD = maxDist;
  m.parts.forEach((p, i) => {
    const s = projectPart(p);
    if (!s.visible) return;
    const d = Math.hypot(s.x - sx, s.y - sy);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/* shoot a ray from the cursor into the model; fall back to the closest part centre */
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function pickPart(sx, sy) {
  const m = models[current];
  if (!m) return -1;
  ndc.set((sx / innerWidth) * 2 - 1, -(sy / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  scene.updateMatrixWorld(true);
  const hit = raycaster.intersectObjects(m.group.children, false)[0];
  if (hit && m.partOfMesh.has(hit.object)) return m.partOfMesh.get(hit.object);
  return nearestPart(sx, sy, 60);
}

function updateLabels() {
  const m = models[current];
  if (!m) return;
  group.updateMatrixWorld();
  const vis = THREE.MathUtils.smoothstep(state.explode, 0.25, 1.0);
  for (const l of m.labels) {
    const isSel = l.index === selected;
    if (!l.eligible && !isSel) {
      l.el.style.opacity = 0;
      continue;
    }
    const s = projectPart(l.part);
    const a = isSel ? 1 : selected >= 0 ? Math.min(vis, 0.3) : vis;
    l.el.style.opacity = s.visible ? a : 0;
    l.el.style.transform = `translate(${s.x}px, ${s.y}px) translate(-50%, -140%)`;
    l.el.classList.toggle('sel', isSel);
  }
}

/* mouse / keyboard fallbacks (handy for testing without a camera) */
addEventListener('click', (e) => select(pickPart(e.clientX, e.clientY)));
addEventListener('keydown', (e) => {
  if (e.key === 'Escape') select(-1);
  if (e.key === 'n' || e.key === 'N') stepModel(1);
  if (e.key === 'p' || e.key === 'P') stepModel(-1);
});

/* ---------------- Hand tracking ---------------- */
let landmarker = null;

async function initHands() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch {
    return await HandLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = THREE.MathUtils.clamp;

// 0 = fist, 1 = fully open hand
function openness(h) {
  const scale = dist(h[0], h[9]) || 0.001;
  const tips = [8, 12, 16, 20];
  const r = tips.reduce((s, i) => s + dist(h[i], h[0]), 0) / tips.length / scale;
  return clamp((r - 1.1) / 0.6, 0, 1); // tweak 1.1 / 0.6 if it feels off
}

// centre of the palm (more stable than a single landmark)
function palmCenter(h) {
  let x = 0;
  let y = 0;
  for (const i of [0, 5, 9, 13, 17]) {
    x += h[i].x;
    y += h[i].y;
  }
  return { x: x / 5, y: y / 5 };
}

/* One Euro filter: very smooth when the hand is slow, still responsive when it is fast */
class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.t = null;
    this.x = 0;
    this.dx = 0;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, t) {
    if (this.t === null || t - this.t > 0.4) {
      this.t = t;
      this.x = x;
      this.dx = 0;
      return x;
    }
    const dt = Math.max(t - this.t, 1e-3);
    this.dx += OneEuro.alpha(this.dCutoff, dt) * ((x - this.x) / dt - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    this.t = t;
    return this.x;
  }
  reset() {
    this.t = null;
  }
}

// lower first number = smoother (but laggier); higher beta = snappier when moving fast
const fPalmX = new OneEuro(0.5, 3);
const fPalmY = new OneEuro(0.5, 3);
const fOpen = new OneEuro(0.8, 1.5);
const fZoom = new OneEuro(0.6, 3);
const fCurX = new OneEuro(1.2, 8);
const fCurY = new OneEuro(1.2, 8);
const fPinch = new OneEuro(1.2, 4);
const allFilters = [fPalmX, fPalmY, fOpen, fZoom, fCurX, fCurY, fPinch];
const resetFilters = () => allFilters.forEach((f) => f.reset());

const state = {
  explode: 0, explodeT: 0,
  rotY: 0, rotYT: 0,
  rotX: 0, rotXT: 0,
  posX: 0, posXT: 0,
  posY: 0, posYT: 0,
  zoom: ZOOM_START, zoomT: ZOOM_START,
  zoomRef: null,
  hands: 0,
  openness: 0,
  pinching: false,
  wasPinching: false,
  palmHist: [],
  swipeLockUntil: 0,
};

function drawHands(list) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.fillStyle = '#4cc9f0';
  for (const h of list) {
    for (const p of h) {
      octx.beginPath();
      octx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
      octx.fill();
    }
  }
}

function handleHands(list, now) {
  state.hands = list.length;
  drawHands(list);

  /* ---- no hands: settle back to the centre ---- */
  if (!list.length) {
    cursorEl.style.display = 'none';
    state.wasPinching = state.pinching = false;
    state.palmHist.length = 0;
    state.zoomRef = null;
    resetFilters();
    if (selected < 0) {
      state.explodeT = 0;
      state.posXT = 0;
      state.posYT = 0;
    }
    return;
  }

  /* ---- two hands: zoom (spread apart = in, together = out) ---- */
  if (list.length === 2) {
    cursorEl.style.display = 'none';
    state.wasPinching = state.pinching = false;
    state.palmHist.length = 0;
    const a = palmCenter(list[0]);
    const b = palmCenter(list[1]);
    const d = Math.max(fZoom.filter(dist(a, b), now), 0.04);
    if (!state.zoomRef) state.zoomRef = { d, zoom: state.zoomT }; // remember the start
    state.zoomT = clamp((state.zoomRef.zoom * state.zoomRef.d) / d, ZOOM_MIN, ZOOM_MAX);
    return;
  }
  state.zoomRef = null;

  /* ---- one hand ---- */
  const h = list[0];
  const scale = dist(h[0], h[9]) || 0.001;
  const ratio = fPinch.filter(dist(h[4], h[8]) / scale, now); // thumb tip <-> index tip

  if (state.pinching) {
    if (ratio > 0.45) state.pinching = false;
  } else if (ratio < 0.3) {
    state.pinching = true;
  }
  const near = ratio < 0.6; // about to pinch: freeze the view so aiming is easy

  // cursor = midpoint between thumb and index (image is mirrored on screen)
  const mx = fCurX.filter((h[4].x + h[8].x) / 2, now);
  const my = fCurY.filter((h[4].y + h[8].y) / 2, now);
  const sx = (1 - mx) * innerWidth;
  const sy = my * innerHeight;
  cursorEl.style.display = 'block';
  cursorEl.style.transform = `translate(${sx}px, ${sy}px) translate(-50%, -50%)`;
  cursorEl.classList.toggle('pinch', state.pinching);

  // pinch started -> select the part under the cursor (or clear if there is none)
  if (state.pinching && !state.wasPinching) select(pickPart(sx, sy));
  state.wasPinching = state.pinching;

  if (near) {
    state.palmHist.length = 0;
    return;
  }

  const palm = palmCenter(h);

  /* swipe: raw palm path over the last ~0.4 s, in screen space (so mirrored) */
  state.palmHist.push({ t: now, x: 1 - palm.x, y: palm.y });
  while (state.palmHist.length && now - state.palmHist[0].t > SWIPE_WINDOW) state.palmHist.shift();
  const first = state.palmHist[0];
  const last = state.palmHist[state.palmHist.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;
  const swiped =
    state.palmHist.length >= 4 && Math.abs(dx) >= SWIPE_DIST && Math.abs(dy) < Math.abs(dx) * 0.7;

  if (swiped && now > state.swipeLockUntil) {
    state.swipeLockUntil = now + SWIPE_COOLDOWN;
    state.palmHist.length = 0;
    if (dx < 0) stepModel(1); // right -> left = next
    else if (SWIPE_PREVIOUS) stepModel(-1); // left -> right = previous (optional)
    return;
  }

  if (selected >= 0) return; // inspecting a part: keep the view still

  /* explode / assemble follows how open the hand is */
  state.openness = fOpen.filter(openness(h), now);
  state.explodeT = state.openness * 3;

  /* the model follows the hand (and turns a little with it) */
  if (now < state.swipeLockUntil) {
    // just swiped: let the new model glide back to the centre
    state.posXT = 0;
    state.posYT = 0;
    return;
  }
  const px = 1 - fPalmX.filter(palm.x, now); // mirrored, 0..1 left to right
  const py = fPalmY.filter(palm.y, now); // 0..1 top to bottom
  const visH = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * state.zoom;
  const visW = visH * camera.aspect;
  state.posXT = (px - 0.5) * visW * FOLLOW_MOVE;
  state.posYT = -(py - 0.5) * visH * FOLLOW_MOVE;
  state.rotYT = (px - 0.5) * 1.6 * FOLLOW_ROTATE;
  state.rotXT = (py - 0.5) * 0.9 * FOLLOW_ROTATE;
}

/* ---------------- Main loop ---------------- */
let lastVideoTime = -1;
let frames = 0;
let fpsTime = performance.now();
let lastFrame = performance.now();
let statusMsg = '';

function loop() {
  requestAnimationFrame(loop);

  const frameStart = performance.now();
  const dt = Math.min((frameStart - lastFrame) / 1000, 0.1);
  lastFrame = frameStart;

  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    handleHands(res.landmarks, performance.now() / 1000);
  }

  if (state.hands === 0 && selected < 0) state.rotYT += 0.25 * dt; // idle spin

  // keep the spin angle small so it never unwinds a long way when a hand appears
  if (state.rotYT > Math.PI) { state.rotYT -= Math.PI * 2; state.rotY -= Math.PI * 2; }
  if (state.rotYT < -Math.PI) { state.rotYT += Math.PI * 2; state.rotY += Math.PI * 2; }

  // frame-rate independent easing: smaller number = smoother and floatier
  const ease = (rate) => 1 - Math.exp(-rate * dt);
  state.explode += (state.explodeT - state.explode) * ease(5);
  state.rotY += (state.rotYT - state.rotY) * ease(5);
  state.rotX += (state.rotXT - state.rotX) * ease(5);
  state.posX += (state.posXT - state.posX) * ease(5);
  state.posY += (state.posYT - state.posY) * ease(5);
  state.zoom += (state.zoomT - state.zoom) * ease(4);

  if (models[current]) applyExplode(models[current], state.explode);
  group.position.set(state.posX, state.posY, 0);
  group.rotation.set(state.rotX, state.rotY, 0);
  camera.position.z = state.zoom;

  renderer.render(scene, camera);
  updateLabels();

  frames++;
  const now = performance.now();
  if (now - fpsTime > 500) {
    const fps = Math.round((frames * 1000) / (now - fpsTime));
    frames = 0;
    fpsTime = now;
    if (statusMsg) {
      hud.textContent = statusMsg;
    } else if (!models.length) {
      hud.textContent = 'Loading 3D models…';
    } else if (!landmarker) {
      hud.textContent = 'Loading hand tracking…';
    } else {
      const sel = selected >= 0 ? models[current].parts[selected].name : '-';
      hud.textContent =
        `Model: ${models[current].name} (${current + 1}/${models.length})\n` +
        `Hands: ${state.hands}   Pinch: ${state.pinching ? 'yes' : 'no'}\n` +
        `Openness: ${(state.openness * 100).toFixed(0)}%   Zoom: ${(ZOOM_START / state.zoom).toFixed(2)}x\n` +
        `Selected: ${sel}\n` +
        `FPS: ${fps}`;
    }
  }
}

/* ---------------- Loading ---------------- */
const draco = new DRACOLoader();
draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(draco);

async function loadModels() {
  for (const { url, name } of GLB_MODELS) {
    try {
      const gltf = await gltfLoader.loadAsync(url);
      addModel(modelFromGLTF(name, gltf));
    } catch (err) {
      console.warn(`Could not load ${url}`, err);
    }
  }
  if (!models.length) statusMsg = 'No models loaded. Check public/models and the console (F12).';
}

async function start() {
  loadModels(); // runs in the background; the first model appears as soon as it is ready

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    video.srcObject = stream;
    await video.play();
    landmarker = await initHands();
  } catch (err) {
    statusMsg = 'Camera/hand error: ' + err.message + '\n(mouse + N key still work)';
    console.error(err);
  }
}

loop();
start();