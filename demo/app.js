// Silhouette AR lab: camera frame -> segmentation -> tracked silhouette meshes
// (depth-only occluders) -> selection -> avatar interactions on A* paths.
import * as THREE from '/static/vendor/three.module.js';

const $ = s => document.querySelector(s);
const SESSION = 'web-' + Math.random().toString(36).slice(2, 10);
const SAMPLE_CLASSES = ['background', 'bear', 'rabbit'];
const SAMPLE_TOYS = [
  {cls: 1, kind: 'bear', x: -.3, z: 1.0, h: .26, color: '#9a6a3c', move: true},
  {cls: 2, kind: 'rabbit', x: .24, z: 1.3, h: .30, color: '#e9e4da'},
  {cls: 1, kind: 'bear', x: .05, z: 1.75, h: .24, color: '#c99a5b'}
];
const VERBS = [[/\b(follow|chase|track)\b/, 'follow'], [/\b(pet|pat|stroke|touch)\b/, 'pet'], [/\bpush\b/, 'push'],
  [/\b(ride|mount|jump on|sit on)\b/, 'ride'], [/\b(point|show)\b/, 'point'], [/\b(approach|go|walk|come|move)\b/, 'approach']];

export function start(Avatar) {
  const view = $('#view'), details = $('#details'), statusLine = $('#status');
  const stage = Avatar.createStage(view, {background: '#000000'});
  const {scene, camera} = stage, actor = stage.avatar;
  const hidden = [];
  scene.traverse(o => { if (o.type === 'GridHelper' || o.isGridHelper || (o.isMesh && o.geometry?.type === 'PlaneGeometry')) { o.visible = false; hidden.push(o); } });
  const frame = document.createElement('canvas'); frame.width = 640; frame.height = 480;
  const fctx = frame.getContext('2d', {willReadFrequently: true});
  const texture = new THREE.CanvasTexture(frame); texture.colorSpace = THREE.SRGBColorSpace;
  scene.background = texture;
  const meshGroup = new THREE.Group(), pathLine = new THREE.Group(); scene.add(meshGroup, pathLine);
  let W = 640, H = 480, source = 'sample', video = null, still = null, maskImage = null;
  let current = null, selected = null, following = null, walkToken = 0, timers = [], inflight = false, lastUpdate = 0, riding = null;
  const motion = new Map();  // instance id -> {seat, time, velocity}: ride points per silhouette update
  let calibrating = false, pendingReset = true, ort = null, ortSession = null, ortMeta = null, serverStatus = {};

  const say = text => { statusLine.textContent = text; };
  const num = (id, fallback) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : fallback; };
  const w2t = p => new THREE.Vector3(p[0], p[1], -p[2]);
  const scaleOf = () => num('#avatar-height', .36) / (actor.rig?.metrics?.height || 1.75);
  const avatarXZ = () => [actor.root.position.x, -actor.root.position.z];
  const later = (fn, ms) => { const id = setTimeout(fn, ms); timers.push(id); return id; };
  async function post(url, body) {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const v = await r.json(); if (!r.ok) throw Error(v.error || r.statusText); return v;
  }
  fetch('/api/status').then(r => r.json()).then(v => { serverStatus = v; }).catch(() => {});

  // ------------------------------------------------------------ calibration / camera
  function calibration() {
    return {hfov: num('#hfov', 60), camera_height: num('#height', .55), pitch: num('#pitch', 0), clearance: num('#clearance', .05), contact: 'bbox'};
  }
  const focal = () => (W / 2) / Math.tan(THREE.MathUtils.degToRad(num('#hfov', 60)) / 2);
  let orbit = {on: false, angle: .6, drag: null};
  function applyCamera() {
    view.style.aspectRatio = `${W} / ${H}`;
    if (orbit.on) {
      const c = current?.meshes?.length ? current.meshes.flatMap(m => m.vertices_world) : [[0, 0, 1.2]];
      const cx = c.reduce((a, p) => a + p[0], 0) / c.length, cz = c.reduce((a, p) => a + p[2], 0) / c.length;
      camera.fov = 45; camera.position.set(cx + 1.5 * Math.sin(orbit.angle), .9, -(cz + 1.5 * Math.cos(orbit.angle)));
      camera.up.set(0, 1, 0); camera.lookAt(cx, .15, -cz); camera.updateProjectionMatrix(); return;
    }
    const fy = focal(), pitch = THREE.MathUtils.degToRad(num('#pitch', 0)), h = num('#height', .55);
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(H / 2 / fy)); camera.near = .01; camera.far = 50;
    camera.position.set(0, h, 0); camera.up.set(0, 1, 0); camera.lookAt(0, h - Math.sin(pitch), -Math.cos(pitch));
    camera.updateProjectionMatrix();
  }
  function setOrbit(on) {
    orbit.on = on; scene.background = on ? new THREE.Color('#101827') : texture; hidden.forEach(o => { o.visible = on; });
    meshGroup.children.forEach(o => { if (o.userData.fill) o.visible = on || $('#outlines').checked; }); applyCamera();
  }

  // ------------------------------------------------------------ sources
  function resize(w, h) { const s = Math.min(1, 640 / w); W = Math.round(w * s); H = Math.round(h * s); frame.width = W; frame.height = H; applyCamera(); }
  function project(p) {
    const pitch = THREE.MathUtils.degToRad(num('#pitch', 0)), h = num('#height', .55), f = focal();
    const dy = p[1] - h, dz = p[2], xc = p[0], yc = -Math.cos(pitch) * dy - Math.sin(pitch) * dz, zc = -Math.sin(pitch) * dy + Math.cos(pitch) * dz;
    return [W / 2 + f * xc / zc, H / 2 + f * yc / zc, zc];
  }
  function toyShapes(kind) {
    return kind === 'bear'
      ? [['e', 0, .11, .075, .095], ['e', 0, .235, .058, .055], ['e', -.045, .285, .022, .022], ['e', .045, .285, .022, .022],
        ['e', -.077, .13, .024, .05], ['e', .077, .13, .024, .05], ['e', -.04, .03, .032, .03], ['e', .04, .03, .032, .03], ['a', 0, .22, .022, .018]]
      : [['e', 0, .11, .07, .095], ['e', 0, .23, .055, .052], ['e', -.022, .33, .016, .07], ['e', .022, .33, .016, .07],
        ['e', -.035, .015, .03, .015], ['e', .035, .015, .03, .015], ['a', 0, .215, .018, .012]];
  }
  function toyPose(toy, t) { return toy.move ? {...toy, x: toy.x + .15 * Math.sin(t * .3), z: toy.z + .08 * Math.sin(t * .2)} : toy; }
  function drawToy(ctx, toy, maskOnly) {
    const [u, v, zc] = project([toy.x, 0, toy.z]), s = focal() / zc * toy.h / .3;
    for (const [type, x, y, rx, ry] of toyShapes(toy.kind)) {
      if (maskOnly && type === 'a') continue;
      const cx = u + x * s, cy = v - y * s;
      ctx.beginPath(); ctx.ellipse(cx, cy, rx * s, ry * s, 0, 0, Math.PI * 2);
      if (maskOnly) ctx.fillStyle = '#fff';
      else if (type === 'a') ctx.fillStyle = '#3a2a20';
      else { const g = ctx.createRadialGradient(cx - rx * s * .4, cy - ry * s * .4, 1, cx, cy, Math.max(rx, ry) * s * 1.1); g.addColorStop(0, '#ffffff55'); g.addColorStop(.35, toy.color); g.addColorStop(1, '#00000066'); ctx.fillStyle = toy.color; ctx.fill(); ctx.fillStyle = g; }
      ctx.fill();
    }
  }
  const sampleToys = t => SAMPLE_TOYS.map(toy => toyPose(toy, t)).sort((a, b) => b.z - a.z);
  function drawSample(t) {
    const ctx = fctx; ctx.fillStyle = '#c9c2b5'; ctx.fillRect(0, 0, W, H);
    const corners = [[-4, 0, .25], [4, 0, .25], [4, 0, 8], [-4, 0, 8]].map(project).filter(p => p[2] > 0);
    ctx.beginPath(); corners.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, '#8a6a4c'); g.addColorStop(1, '#b48a5e'); ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = '#00000022'; ctx.lineWidth = 1;
    for (let x = -4; x <= 4; x += .25) { const a = project([x, 0, .25]), b = project([x, 0, 8]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    for (let z = .5; z <= 8; z += .5) { const a = project([-4, 0, z]), b = project([4, 0, z]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    for (const toy of sampleToys(t)) {
      const [u, v, zc] = project([toy.x, 0, toy.z]), s = focal() / zc * toy.h / .3;
      ctx.fillStyle = '#00000040'; ctx.beginPath(); ctx.ellipse(u, v, .09 * s, .025 * s, 0, 0, Math.PI * 2); ctx.fill();
      drawToy(ctx, toy, false);
    }
  }
  const scratch = document.createElement('canvas'), maskCanvas = document.createElement('canvas');
  function classMaskURL(fill) {
    scratch.width = maskCanvas.width = W; scratch.height = maskCanvas.height = H;
    const out = new ImageData(W, H); fill(out.data); maskCanvas.getContext('2d').putImageData(out, 0, 0);
    return maskCanvas.toDataURL('image/png');
  }
  function sampleMask(t) {
    const sctx = scratch.getContext('2d', {willReadFrequently: true});
    return classMaskURL(data => {
      for (let i = 3; i < data.length; i += 4) data[i] = 255;
      for (const toy of sampleToys(t)) {
        sctx.clearRect(0, 0, W, H); drawToy(sctx, toy, true);
        const a = sctx.getImageData(0, 0, W, H).data;
        for (let i = 0; i < a.length; i += 4) if (a[i + 3] > 127) data[i] = data[i + 1] = data[i + 2] = toy.cls;
      }
    });
  }
  async function startWebcam() {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('This browser offers no camera access (getUserMedia); use a file instead');
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    video = Object.assign(document.createElement('video'), {srcObject: stream, muted: true, playsInline: true}); await video.play();
    resize(video.videoWidth || 640, video.videoHeight || 480);
  }
  function stopVideo() { if (video?.srcObject) video.srcObject.getTracks().forEach(t => t.stop()); if (video) video.pause(); video = null; }
  async function setSource(next) {
    stopVideo(); still = null; source = next; resetScene();
    if (next === 'sample') { resize(640, 480); $('#seg').value = 'authored'; $('#height').value = .55; $('#hfov').value = 60; $('#pitch').value = 24; applyCamera(); return update(true); }
    if (next === 'webcam') {
      try { await startWebcam(); } catch (e) { say('Webcam unavailable: ' + e.message); $('#source').value = 'sample'; return setSource('sample'); }
      chooseModelSegmentation(); say('Webcam running. Calibrate the floor: press "Tap floor to calibrate", then tap a floor point at the set distance.');
    }
    if (next === 'file') $('#file').click();
  }
  function chooseModelSegmentation() {
    if ($('#seg').value !== 'authored') return;
    $('#seg').value = serverStatus.segmenter ? 'server' : serverStatus.onnx ? 'onnx' : 'maskfile';
    $('#mask-row').hidden = $('#seg').value !== 'maskfile';
  }
  $('#file').onchange = async e => {
    const f = e.target.files[0]; if (!f) return; const url = URL.createObjectURL(f); stopVideo(); still = null; resetScene();
    if (f.type.startsWith('video/')) {
      video = Object.assign(document.createElement('video'), {src: url, muted: true, loop: true, playsInline: true}); await video.play();
      resize(video.videoWidth, video.videoHeight);
    } else {
      still = await new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = url; });
      resize(still.naturalWidth, still.naturalHeight);
    }
    source = 'file'; chooseModelSegmentation(); drawFrame(0); say(`Loaded ${f.name}; calibrate the floor, then Update silhouettes.`); update(true);
  };
  $('#mask-file').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    maskImage = await new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = URL.createObjectURL(f); });
    if (source === 'sample' && !still) { still = null; } update(true);
  };
  function drawFrame(t) {
    if (source === 'sample') drawSample(t);
    else if (video && video.readyState >= 2) fctx.drawImage(video, 0, 0, W, H);
    else if (still) fctx.drawImage(still, 0, 0, W, H);
    texture.needsUpdate = true;
  }

  // ------------------------------------------------------------ segmentation routes
  async function ensureOnnx() {
    if (ortSession) return;
    serverStatus = await (await fetch('/api/status')).json();
    if (!serverStatus.onnx) throw Error('Start the demo with --onnx <model.onnx> to segment in the browser');
    if (!serverStatus.onnx_runtime_web) throw Error('Run python scripts/prepare_onnx_web.py to vendor onnxruntime-web locally');
    ort = await import('/static/vendor/onnxruntime/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = '/static/vendor/onnxruntime/'; ort.env.wasm.numThreads = 1;
    ortMeta = await (await fetch('/api/model.json')).json();
    ortSession = await ort.InferenceSession.create('/api/model.onnx', {executionProviders: ['wasm']});
  }
  async function onnxMask() {
    await ensureOnnx();
    const S = ortMeta.input_size, small = document.createElement('canvas'); small.width = small.height = S;
    const sctx = small.getContext('2d', {willReadFrequently: true}); sctx.drawImage(frame, 0, 0, S, S);
    const px = sctx.getImageData(0, 0, S, S).data, input = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) input[c * S * S + i] = px[i * 4 + c] / 255;
    const started = performance.now();
    const out = await ortSession.run({image: new ort.Tensor('float32', input, [1, 3, S, S])});
    const logits = out.logits.data, C = ortMeta.classes.length, classes = new Uint8Array(S * S);
    for (let i = 0; i < S * S; i++) { let best = 0; for (let c = 1; c < C; c++) if (logits[c * S * S + i] > logits[best * S * S + i]) best = c; classes[i] = best; }
    const ms = performance.now() - started, img = sctx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) { img.data.set([classes[i], classes[i], classes[i], 255], i * 4); }
    sctx.putImageData(img, 0, 0);
    const big = document.createElement('canvas'); big.width = W; big.height = H; const bctx = big.getContext('2d');
    bctx.imageSmoothingEnabled = false; bctx.drawImage(small, 0, 0, W, H);
    return {mask: big.toDataURL('image/png'), ms};
  }
  function uploadedMask() {
    if (!maskImage) throw Error('Choose a binary mask image (white object on black) for the uploaded-mask route');
    const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
    x.imageSmoothingEnabled = false; x.drawImage(maskImage, 0, 0, W, H); return c.toDataURL('image/png');
  }
  async function payload(t) {
    const seg = $('#seg').value, base = {session: SESSION, calibration: calibration()};
    if (seg === 'authored') {
      if (source !== 'sample') throw Error('The authored mask exists only for the sample scene; choose a model or mask route');
      return {...base, segmentation: 'mask', mask_kind: 'indexed', classes: SAMPLE_CLASSES, mask: sampleMask(t), provenance: 'authored sample mask (drawn with the scene)'};
    }
    if (seg === 'server') return {...base, segmentation: 'server', image: frame.toDataURL('image/jpeg', .85)};
    if (seg === 'onnx') { const r = await onnxMask(); return {...base, segmentation: 'mask', mask_kind: 'indexed', classes: ortMeta.classes, mask: r.mask, provenance: `in-browser ONNX U-Net (${r.ms.toFixed(0)} ms)`}; }
    return {...base, segmentation: 'mask', mask_kind: 'binary', label: $('#mask-label').value || 'object', mask: uploadedMask(), provenance: 'uploaded binary mask'};
  }
  const live = () => !$('#pause').checked && (source === 'sample' || (video && !video.paused));
  async function update(force = false, t = performance.now() / 1000) {
    if (inflight || (!force && !live())) return;
    inflight = true; lastUpdate = performance.now();
    try {
      drawFrame(t); const body = await payload(t);
      if (pendingReset) { body.reset = true; pendingReset = false; motion.clear(); }
      applyFrame(await post('/api/frame', body), t);
    }
    catch (e) { say(e.message); details.textContent = 'Error: ' + e.message; }
    finally { inflight = false; }
  }

  // ------------------------------------------------------------ meshes
  function disposeGroup(group) { while (group.children.length) { const o = group.children.pop(); o.geometry?.dispose(); (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m?.dispose()); } }
  function occluder(mesh) {
    if (typeof stage.makeOccluder === 'function') return stage.makeOccluder(mesh);
    if (typeof Avatar.makeOccluder === 'function') return Avatar.makeOccluder(mesh);
    mesh.material = new THREE.MeshBasicMaterial({colorWrite: false, depthWrite: true, side: THREE.DoubleSide}); mesh.renderOrder = -10; return mesh;
  }
  function buildMeshes() {
    disposeGroup(meshGroup);
    for (const m of current?.meshes || []) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices_world.flatMap(p => [p[0], p[1], -p[2]]), 3));
      geometry.setIndex(m.triangles.flat());
      const hide = occluder(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({side: THREE.DoubleSide})));
      hide.userData.id = m.instance_id; meshGroup.add(hide);
      const chosen = m.instance_id === selected;
      const fill = new THREE.Mesh(geometry.clone(), new THREE.MeshBasicMaterial({color: chosen ? 0xffcf6e : 0x5cd2b7, transparent: true, opacity: chosen ? .35 : .18, depthWrite: false, side: THREE.DoubleSide}));
      fill.renderOrder = 5; fill.userData.fill = true; fill.visible = orbit.on || $('#outlines').checked; meshGroup.add(fill);
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(m.vertices_world.map(w2t)), new THREE.LineBasicMaterial({color: chosen ? 0xffcf6e : 0xffffff, depthTest: false, transparent: true, opacity: .9}));
      line.renderOrder = 6; line.userData.fill = true; line.visible = fill.visible; meshGroup.add(line);
    }
  }
  function drawPath(path) {
    disposeGroup(pathLine);
    if (path?.length > 1) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path.map(p => new THREE.Vector3(p[0], .004, -p[2]))), new THREE.LineBasicMaterial({color: 0xffe06e, depthTest: false})); l.renderOrder = 7; pathLine.add(l); }
  }
  let placed = false;
  function trackMotion(data, t) {
    // Ride points per tracked id, with a velocity estimate between silhouette
    // updates, so a rider stays on a moving object between (and across) updates.
    for (const m of data.meshes) {
      const seat = m.interaction_targets.ride, prev = motion.get(m.instance_id);
      let velocity = [0, 0, 0];
      if (prev && t - prev.time > .05 && t - prev.time < 3) velocity = seat.map((v, i) => .5 * prev.velocity[i] + .5 * (v - prev.seat[i]) / (t - prev.time));
      motion.set(m.instance_id, {seat, time: t, velocity});
    }
  }
  function applyFrame(data, t = performance.now() / 1000) {
    current = data; trackMotion(data, t);
    if (!placed && data.spawn) { actor.root.position.set(data.spawn[0], 0, -data.spawn[2]); placed = true; face([0, 0, 2]); }
    if (selected && !data.meshes.some(m => m.instance_id === selected)) say(`${selected} is not visible in this frame; keeping its last position.`);
    buildMeshes(); applyCamera();
    if (following) replanFollow();
    details.textContent = JSON.stringify({provenance: data.provenance, tracked_ids: data.tracked_ids, selected, following,
      labels: data.meshes.map(m => m.label), triangles: data.meshes.map(m => m.triangles.length), path_points: data.path.length,
      timing_ms: data.timing_ms, calibration: data.calibration, targets: data.meshes.find(m => m.instance_id === selected)?.interaction_targets}, null, 2);
  }
  function resetScene() { stopAll(); selected = null; placed = false; current = null; pendingReset = true; motion.clear(); disposeGroup(meshGroup); disposeGroup(pathLine); }

  // ------------------------------------------------------------ selection
  async function select(body, label) {
    const r = await post('/api/select', {session: SESSION, ...body});
    if (!r.ok) { say(`${label}: no silhouette hit${r.labels?.length ? ' (visible: ' + r.labels.join(', ') + ')' : ''}.`); return null; }
    selected = r.target_id; buildMeshes(); say(`Selected ${r.target_id} (${r.label}) by ${r.method}.`);
    if (stage.lookAt) { const m = meshById(selected); if (m) stage.lookAt(w2t(m.interaction_targets.point)); }
    return r.target_id;
  }
  const meshById = id => current?.meshes?.find(m => m.instance_id === id);
  let down = null;
  view.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; if (orbit.on) orbit.drag = e.clientX; });
  view.addEventListener('pointermove', e => { if (orbit.on && orbit.drag !== null) { orbit.angle += (e.clientX - orbit.drag) * .006; orbit.drag = e.clientX; applyCamera(); } });
  view.addEventListener('pointerup', e => {
    orbit.drag = null; if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6 || e.target !== stage.renderer.domElement || orbit.on) return;
    const rect = view.getBoundingClientRect(), u = (e.clientX - rect.left) / rect.width * W, v = (e.clientY - rect.top) / rect.height * H;
    if (calibrating) {
      const h = num('#height', .55), d = num('#tap-distance', 1), pitch = THREE.MathUtils.radToDeg(Math.atan2(h, d) - Math.atan2(v - H / 2, focal()));
      $('#pitch').value = pitch.toFixed(1); calibrating = false; view.classList.remove('calibrating'); pendingReset = true; selected = null; following = null;
      say(`Floor calibrated: pitch ${pitch.toFixed(1)}° from a tap ${d} m away with the camera ${h} m high.`); applyCamera(); update(true); return;
    }
    select({pixel: [u, v]}, 'Tap').catch(err => say(err.message));
  });
  $('#calibrate').onclick = () => { calibrating = true; view.classList.add('calibrating'); say(`Tap a floor point ${num('#tap-distance', 1)} m in front of the camera.`); };
  $('#select-centre').onclick = () => select({pixel: [W / 2, H / 2]}, 'Centre ray').catch(e => say(e.message));
  async function command(text) {
    const id = await select({keyword: text}, `Keyword "${text}"`).catch(e => { say(e.message); return null; });
    if (!id) return;
    const verb = VERBS.find(([re]) => re.test(text.toLowerCase()));
    if (verb) act(verb[1]);
  }
  $('#select-keyword').onclick = () => command($('#keyword').value);
  $('#keyword').addEventListener('keydown', e => { if (e.key === 'Enter') command($('#keyword').value); });
  $('#mic').onclick = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { say('Speech recognition is not available in this browser; type the keyword instead.'); return; }
    const rec = new SR(); rec.lang = 'en-US'; rec.interimResults = false; rec.maxAlternatives = 1;
    rec.onresult = e => { const text = e.results[0][0].transcript; $('#keyword').value = text; command(text); };
    rec.onerror = e => say('Speech recognition: ' + e.error); rec.start(); say('Listening… say e.g. "follow the rabbit".');
  };

  // ------------------------------------------------------------ avatar behaviour
  function face(p) { if (typeof stage.setFacing === 'function') stage.setFacing([p[0], 0, -p[2]]); else actor.root.rotation.y = Math.atan2(p[0] - actor.root.position.x, -p[2] - actor.root.position.z); }
  function release() { if (typeof stage.reachTo === 'function') stage.reachTo(null); }
  function standUp() { if (riding && typeof stage.stand === 'function') stage.stand(); riding = null; }
  function dismount() {
    // Step off toward the camera-side stand point instead of standing up inside the object.
    const m = riding && meshById(riding.id); standUp();
    if (m) { const s = m.interaction_targets.stand; stage.moveTo(s[0], -s[2], .35); }
  }
  function stopAll() { walkToken++; timers.forEach(clearTimeout); timers = []; following = null; followGoal = null; $('#follow').setAttribute('aria-pressed', 'false'); release(); dismount(); stage.gesture('idle'); }
  function walk(path, done) {
    const token = ++walkToken; standUp(); drawPath(path);
    const points = path.slice(1), speed = Math.max(.12, num('#avatar-height', .36) * .75);
    const step = () => {
      if (token !== walkToken) return;
      if (!points.length) { stage.gesture('idle'); done?.(); return; }
      const p = points.shift(), dist = Math.hypot(p[0] - actor.root.position.x, -p[2] - actor.root.position.z), dur = Math.max(.08, dist / speed);
      if (dist < .005) return step();
      face(p); stage.gesture('walk'); stage.moveTo(p[0], -p[2], dur); later(step, dur * 1000);
    };
    step();
  }
  async function goTo(id, kind, done) {
    const r = await post('/api/plan', {session: SESSION, target_id: id, from: avatarXZ(), interaction: kind});
    if (!r.ok) { say(`${kind}: ${r.reason}`); return null; }
    walk(r.path.length ? r.path : [[...avatarXZ()].flatMap((v, i) => i ? [0, v] : [v])], () => { face(r.targets.point); if (stage.lookAt) stage.lookAt(w2t(r.targets.point)); done?.(r.targets); });
    return r;
  }
  // Follow: re-planned on every silhouette update toward a stand-off point beside
  // the object. The planner dilates every footprint by the clearance plus the
  // avatar's body radius, so neither the path nor the stance enters the mesh.
  let planning = false, followGoal = null;
  const bodyRadius = () => .22 * num('#avatar-height', .36);
  async function replanFollow() {
    if (planning || !following) return; planning = true;
    const id = following;
    try {
      const r = await post('/api/plan', {session: SESSION, target_id: id, from: avatarXZ(), interaction: 'follow', body_radius: bodyRadius()});
      if (following !== id) return;
      if (!r.ok) { say(`Follow: ${r.reason}`); return; }
      const goal = r.targets.follow;
      if (followGoal && Math.hypot(goal[0] - followGoal[0], goal[2] - followGoal[2]) < .02) return;  // keep walking/standing; no restart jitter
      followGoal = goal; const here = avatarXZ();
      walk(r.path.length ? r.path : [[here[0], 0, here[1]]], () => { face(r.targets.point); if (stage.lookAt) stage.lookAt(w2t(r.targets.point)); });
      say(`Following ${id}: ${r.path.length} waypoint(s) to a point ${r.targets.follow_standoff.toFixed(2)} m beside it.`);
    }
    catch (e) { say(e.message); } finally { planning = false; }
  }
  // Ride: the seat is the tracked silhouette's body top, re-read from every
  // silhouette update (extrapolated between updates) and applied every frame.
  const RIDE_TURN = THREE.MathUtils.degToRad(65);
  function seatOf(id, t) {
    const m = motion.get(id); if (!m) return null;
    const ahead = live() ? Math.min(Math.max(t - m.time, 0), .8) : 0;
    return m.seat.map((v, i) => v + m.velocity[i] * ahead);
  }
  function mount(id) {
    const seat = seatOf(id, performance.now() / 1000); if (!seat) { say(`${id} is not visible.`); return; }
    const cam = current?.camera?.origin || [0, 0, 0];
    actor.move = null;
    riding = {id, start: performance.now() / 1000, from: actor.root.position.clone(), turn: seat[0] < cam[0] ? -RIDE_TURN : RIDE_TURN};
    stage.gesture('idle'); disposeGroup(pathLine); rideTick(performance.now() / 1000);
    say(`Riding ${id} (seated on the silhouette's body top at ${seat[1].toFixed(2)} m, tracking it).`);
  }
  function rideTick(t) {
    if (!riding) return;
    const seat = seatOf(riding.id, t); if (!seat) return;  // object briefly lost: keep the last seat
    const k = Math.min(1, (t - riding.start) / .45), ease = k * k * (3 - 2 * k), target = new THREE.Vector3(seat[0], 0, -seat[2]);
    if (k < 1) actor.root.position.lerpVectors(riding.from, target, ease); else actor.root.position.copy(target);
    // Turned 65° from the camera, away from the image centre: the bent legs read in
    // profile and hang just in front of the object's depth-only occluder.
    const cam = current?.camera?.origin || [0, 0, 0];
    actor.root.rotation.y = Math.atan2(cam[0] - seat[0], seat[2] - cam[2]) + riding.turn;
    // sit() takes the seat in the root's unscaled units; the feet target adds the
    // rig's unscaled ankle height, so convert the wanted world foot height.
    const s = scaleOf(), ankle = actor.rig?.metrics?.ankleY || 0;
    if (typeof stage.sit === 'function') stage.sit({seatHeight: seat[1] / s, floorHeight: seat[1] * .35 + ankle * s - ankle, handsOnLap: true});
    else actor.root.position.y = seat[1];
  }
  function reach(point, side = 'auto') {
    if (typeof stage.reachTo === 'function') return stage.reachTo(side, point);
    stage.pointAt([point.x, point.y, point.z]); stage.gesture('touch');
  }
  function act(kind) {
    if (!selected) { say('Select an object first: tap it, use the centre ray, or say its class.'); return; }
    const id = selected, m = meshById(id); if (!m) { say(`${id} is not visible.`); return; }
    if (kind !== 'follow') { walkToken++; timers.forEach(clearTimeout); timers = []; following = null; $('#follow').setAttribute('aria-pressed', 'false'); }
    release();
    if (kind === 'point') { dismount(); face(m.interaction_targets.point); stage.pointAt(m.interaction_targets.point.map((v, i) => i === 2 ? -v : v)); if (stage.lookAt) stage.lookAt(w2t(m.interaction_targets.point)); later(() => stage.gesture('idle'), 3500); say(`Pointing at ${id}.`); return; }
    if (kind === 'follow') {
      following = following === id ? null : id; followGoal = null; $('#follow').setAttribute('aria-pressed', String(Boolean(following)));
      if (following) replanFollow(); else stopAll(); return;
    }
    goTo(id, kind, targets => {
      if (kind === 'approach') { say(`Standing next to ${id}.`); return; }
      if (kind === 'pet') {
        const base = w2t(targets.pet); let k = 0;
        const iv = setInterval(() => { const p = base.clone(); p.y += .015 * Math.sin(k++ * .9); reach(p); }, 110); timers.push(iv);
        later(() => { clearInterval(iv); release(); stage.gesture('idle'); }, 3200); say(`Petting ${id}.`); return;
      }
      if (kind === 'push') {
        const p = w2t(targets.push), right = new THREE.Vector3(1, 0, 0).applyQuaternion(actor.root.quaternion).multiplyScalar(.035 * num('#avatar-height', .36) / .42);
        if (typeof stage.reachTo === 'function') { stage.reachTo('left', p.clone().add(right)); stage.reachTo('right', p.clone().sub(right)); } else reach(p);
        later(() => { release(); stage.gesture('idle'); }, 2600); say(`Pushing ${id}.`); return;
      }
      if (kind === 'ride') mount(id);  // the live seat, not the plan-time target: the object may have moved meanwhile
    }).catch(e => say(e.message));
  }
  for (const kind of ['point', 'approach', 'follow', 'pet', 'push', 'ride']) $('#' + kind).onclick = () => act(kind);
  $('#stop').onclick = () => { stopAll(); say('Stopped.'); };

  // ------------------------------------------------------------ controls and loop
  $('#source').onchange = e => setSource(e.target.value);
  $('#seg').onchange = e => { $('#mask-row').hidden = e.target.value !== 'maskfile'; update(true); };
  $('#build').onclick = () => update(true);
  $('#outlines').onchange = () => meshGroup.children.forEach(o => { if (o.userData.fill) o.visible = orbit.on || $('#outlines').checked; });
  $('#orbit').onchange = e => setOrbit(e.target.checked);
  for (const id of ['#height', '#hfov', '#pitch', '#clearance']) $(id).addEventListener('change', () => { pendingReset = true; selected = null; following = null; applyCamera(); update(true); });
  function tick(now) {
    const t = now / 1000;
    if (live()) drawFrame(t);
    actor.root.scale.setScalar(scaleOf());
    rideTick(t);
    if (now - lastUpdate > 650) update(false, t);
    requestAnimationFrame(tick);
  }
  setSource('sample'); requestAnimationFrame(tick);
  window.silhouetteDemo = {stage, select, act, command, get current() { return current; }, get selected() { return selected; }, get following() { return following; }, get riding() { return riding?.id || null; }};
}
