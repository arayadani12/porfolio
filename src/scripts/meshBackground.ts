// Unstructured adaptive triangular mesh background.
//
// Architecture:
//   • BASE_N jittered points fill the canvas (fixed per resize).
//   • REFINE_N "cloud" points live in a Poisson-disk template scaled to
//     REFINE_R world-units. They track the cursor when the mouse is active
//     and drift back to rest positions (same density as base) when it leaves.
//   • Every frame the combined point set is triangulated with Bowyer-Watson
//     Delaunay, and the resulting edges + fills are pushed to GPU buffers.
//   • The scalar-field backdrop shader is unchanged.

import * as THREE from 'three';

// ─── Visual constants ─────────────────────────────────────────────────────────
const BG_COLOR   = '#0a0a0f';
const LINE_COLOR = '#2c4b80';

// ─── Mesh parameters ──────────────────────────────────────────────────────────
const BASE_N    = 400;  // interior base points
const BOUND_DIV = 14;   // boundary anchor subdivisions per edge
const REFINE_N  = 2;   // mouse-cloud points
const REFINE_R  = 0.1; // active cluster radius (world units)

// ─── Buffer sizing ────────────────────────────────────────────────────────────
const MAX_PTS  = 560;              // generous upper bound for point arrays
const MAX_TRIS = MAX_PTS * 2 + 20; // Delaunay bound ≈ 2n

// ─── Bowyer-Watson: pre-allocated module-level buffers (no GC per frame) ──────
const _px  = new Float64Array(MAX_PTS + 3); // +3 for super-triangle
const _py  = new Float64Array(MAX_PTS + 3);
const _tA  = new Int16Array(MAX_TRIS * 2);  // intermediate state may exceed MAX_TRIS
const _tB  = new Int16Array(MAX_TRIS * 2);
const _tC  = new Int16Array(MAX_TRIS * 2);
let   _tN  = 0;
const _bad  = new Uint8Array(MAX_TRIS * 2);
const _bndU = new Int16Array(MAX_TRIS);
const _bndV = new Int16Array(MAX_TRIS);

/**
 * Bowyer-Watson Delaunay triangulation.
 * Input  : px/py arrays of length n (regular JS arrays, copied into typed arrays).
 * Output : triangle indices written into _tA/_tB/_tC[0.._tN-1], all < n.
 */
function triangulate(px: number[], py: number[], n: number): void {
  for (let i = 0; i < n; i++) { _px[i] = px[i]; _py[i] = py[i]; }

  // Super-triangle (guaranteed to contain all points)
  const M = 300;
  _px[n] = 0;    _py[n] = M * 3;
  _px[n+1] = -M*3; _py[n+1] = -M;
  _px[n+2] =  M*3; _py[n+2] = -M;

  _tA[0] = n; _tB[0] = n + 1; _tC[0] = n + 2;
  _tN = 1;

  for (let i = 0; i < n; i++) {
    const xi = _px[i], yi = _py[i];

    // Find bad triangles whose circumcircle contains point i
    let nBad = 0;
    for (let t = 0; t < _tN; t++) {
      const a = _tA[t], b = _tB[t], c = _tC[t];
      const ax = _px[a] - _px[c], ay = _py[a] - _py[c];
      const bx = _px[b] - _px[c], by = _py[b] - _py[c];
      const D  = 2 * (ax * by - ay * bx);
      let bad = 0;
      if (Math.abs(D) > 1e-12) {
        const aa = ax*ax + ay*ay, bb = bx*bx + by*by;
        const ux = (by * aa - ay * bb) / D;
        const uy = (ax * bb - bx * aa) / D;
        const dx = xi - (_px[c] + ux), dy = yi - (_py[c] + uy);
        bad = (dx*dx + dy*dy < ux*ux + uy*uy - 1e-10) ? 1 : 0;
      }
      _bad[t] = bad;
      if (bad) nBad++;
    }
    if (nBad === 0) continue;

    // Collect boundary polygon (edges not shared by two bad triangles)
    let nBnd = 0;
    for (let t = 0; t < _tN; t++) {
      if (!_bad[t]) continue;
      for (let e = 0; e < 3; e++) {
        const u = e === 0 ? _tA[t] : e === 1 ? _tB[t] : _tC[t];
        const v = e === 0 ? _tB[t] : e === 1 ? _tC[t] : _tA[t];
        let shared = false;
        for (let t2 = 0; t2 < _tN; t2++) {
          if (t2 === t || !_bad[t2]) continue;
          const a2=_tA[t2], b2=_tB[t2], c2=_tC[t2];
          if ((a2===u&&b2===v)||(b2===u&&c2===v)||(c2===u&&a2===v)||
              (a2===v&&b2===u)||(b2===v&&c2===u)||(c2===v&&a2===u)) {
            shared = true; break;
          }
        }
        if (!shared) { _bndU[nBnd] = u; _bndV[nBnd] = v; nBnd++; }
      }
    }

    // Compact: remove bad triangles, insert star fill
    let w = 0;
    for (let t = 0; t < _tN; t++) {
      if (!_bad[t]) { _tA[w]=_tA[t]; _tB[w]=_tB[t]; _tC[w]=_tC[t]; w++; }
    }
    for (let e = 0; e < nBnd; e++) {
      _tA[w] = _bndU[e]; _tB[w] = _bndV[e]; _tC[w] = i; w++;
    }
    _tN = w;
  }

  // Strip triangles that touch the super-triangle
  let w = 0;
  for (let t = 0; t < _tN; t++) {
    if (_tA[t] < n && _tB[t] < n && _tC[t] < n) {
      _tA[w]=_tA[t]; _tB[w]=_tB[t]; _tC[w]=_tC[t]; w++;
    }
  }
  _tN = w;
}

// ─── MeshBackground ───────────────────────────────────────────────────────────

class MeshBackground {
  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene:    THREE.Scene;
  private camera:   THREE.OrthographicCamera;

  private bgMaterial: THREE.ShaderMaterial;
  private bgQuad:     THREE.Mesh;

  private fillMesh: THREE.Mesh;
  private fillGeom: THREE.BufferGeometry;
  private fillPos:  Float32Array;
  private fillAttr: THREE.BufferAttribute;

  private edgeLines: THREE.LineSegments;
  private edgeGeom:  THREE.BufferGeometry;
  private edgePos:   Float32Array;
  private edgeAttr:  THREE.BufferAttribute;

  // Base (fixed) points
  private bx: number[] = [];
  private by: number[] = [];

  // Refinement cloud: current, rest, and unit-disk template
  private rx:   number[] = new Array(REFINE_N).fill(0);
  private ry:   number[] = new Array(REFINE_N).fill(0);
  private rrx:  number[] = [];
  private rry:  number[] = [];
  private rtpl: number[] = []; // [x0,y0, x1,y1, ...] in unit disk

  // Combined point list reused each frame
  private allX: number[] = [];
  private allY: number[] = [];

  private mouse       = { x: 0, y: 0, active: false };
  private smoothMouse = { x: 0, y: 0, active: 0 };
  private time = 0;
  private clock = new THREE.Clock();
  private rafId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private aspect = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false, powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(BG_COLOR, 1.0);

    this.scene  = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
    this.camera.position.z = 1;

    // ── Background scalar-field shader ────────────────────────────────────────
    this.bgMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime:        { value: 0 },
        uMouse:       { value: new THREE.Vector2() },
        uMouseActive: { value: 0 },
        uAspect:      { value: 1 },
      },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position, 1.0); }
      `,
      fragmentShader: /* glsl */`
        precision highp float;
        uniform float uTime;
        uniform vec2  uMouse;
        uniform float uMouseActive;
        uniform float uAspect;
        varying vec2 vUv;

        float field(vec2 p, float t) {
          float v = 0.0;
          v += sin(p.x*1.8 + t*0.25) * cos(p.y*1.4 - t*0.18);
          v += 0.55 * sin((p.x+p.y)*2.6 - t*0.21);
          v += 0.35 * cos(p.x*3.7 - p.y*2.2 + t*0.12);
          float d = length(p - uMouse);
          v += 1.6 * exp(-d*d*5.5) * uMouseActive;
          return v;
        }
        vec3 colormap(float t) {
          vec3 c0 = vec3(0.039,0.039,0.059);
          vec3 c1 = vec3(0.055,0.090,0.180);
          vec3 c2 = vec3(0.090,0.220,0.420);
          vec3 c3 = vec3(0.310,0.557,0.969);
          vec3 c4 = vec3(0.976,0.451,0.086);
          vec3 col = mix(c0,c1,smoothstep(0.00,0.30,t));
          col = mix(col,c2,smoothstep(0.30,0.60,t));
          col = mix(col,c3,smoothstep(0.60,0.85,t));
          col = mix(col,c4,smoothstep(0.85,1.00,t));
          return col;
        }
        void main() {
          vec2 p = vUv * 2.0 - 1.0;
          p.x *= uAspect;
          float v = field(p, uTime);
          float t = clamp(v * 0.35 + 0.5, 0.0, 1.0);
          vec3 col = colormap(t) * 0.55 + vec3(0.039,0.039,0.059) * 0.45;
          gl_FragColor = vec4(col, 1.0);
        }
      `,
      depthTest: false, depthWrite: false,
    });
    this.bgQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.bgMaterial);
    this.bgQuad.frustumCulled = false;
    this.scene.add(this.bgQuad);

    // ── Fill geometry ─────────────────────────────────────────────────────────
    this.fillPos  = new Float32Array(MAX_TRIS * 9);
    this.fillAttr = new THREE.BufferAttribute(this.fillPos, 3);
    this.fillAttr.setUsage(THREE.DynamicDrawUsage);
    this.fillGeom = new THREE.BufferGeometry();
    this.fillGeom.setAttribute('position', this.fillAttr);
    this.fillGeom.setDrawRange(0, 0);
    this.fillMesh = new THREE.Mesh(this.fillGeom, new THREE.MeshBasicMaterial({
      color: LINE_COLOR, transparent: true, opacity: 0.04,
      depthTest: false, depthWrite: false,
    }));
    this.fillMesh.frustumCulled = false;
    this.scene.add(this.fillMesh);

    // ── Edge geometry ─────────────────────────────────────────────────────────
    this.edgePos  = new Float32Array(MAX_TRIS * 18);
    this.edgeAttr = new THREE.BufferAttribute(this.edgePos, 3);
    this.edgeAttr.setUsage(THREE.DynamicDrawUsage);
    this.edgeGeom = new THREE.BufferGeometry();
    this.edgeGeom.setAttribute('position', this.edgeAttr);
    this.edgeGeom.setDrawRange(0, 0);
    this.edgeLines = new THREE.LineSegments(this.edgeGeom, new THREE.LineBasicMaterial({
      color: LINE_COLOR, transparent: true, opacity: 0.55,
      depthTest: false, depthWrite: false,
    }));
    this.edgeLines.frustumCulled = false;
    this.scene.add(this.edgeLines);

    this.buildTemplate();

    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(canvas);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerleave', this.onPointerLeave);
    document.addEventListener('mouseleave', this.onPointerLeave);

    this.onResize();
    this.animate();
  }

  // ── Point generation ────────────────────────────────────────────────────────

  /** Fixed unit-disk template for the refinement cloud.
   *  Min separation grows linearly from FINE_D at the center to COARSE_D at
   *  the edge (r=1), so when the cloud is scaled to world space the outer ring
   *  matches base-mesh density and the boundary is invisible.
   *  Radial distribution is center-biased (r = U^1.8) to pack more triangles
   *  near the cursor and fewer toward the cloud perimeter. */
  private buildTemplate() {
    // FINE_D * REFINE_R ≈ 1/3 of base spacing → fine center triangles
    // COARSE_D * REFINE_R ≈ base mesh spacing → seamless outer ring
    const FINE_D   = 0.07;
    const COARSE_D = 0.20;

    const px: number[] = [], py: number[] = [], pr: number[] = [];
    for (let k = 0; px.length < REFINE_N && k < REFINE_N * 200; k++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.pow(Math.random(), 1.8); // power > 1 → center-biased
      const x = Math.cos(angle) * r, y = Math.sin(angle) * r;
      const dC = FINE_D + (COARSE_D - FINE_D) * r; // candidate's min-sep
      let ok = true;
      for (let j = 0; j < px.length; j++) {
        const dJ  = FINE_D + (COARSE_D - FINE_D) * pr[j];
        const sep = Math.max(dC, dJ); // use the stricter of the two
        if ((x - px[j]) ** 2 + (y - py[j]) ** 2 < sep * sep) { ok = false; break; }
      }
      if (ok) { px.push(x); py.push(y); pr.push(r); }
    }
    while (px.length < REFINE_N) {
      const angle = Math.random() * Math.PI * 2;
      px.push(Math.cos(angle) * Math.random());
      py.push(Math.sin(angle) * Math.random());
    }
    this.rtpl = [];
    for (let i = 0; i < REFINE_N; i++) this.rtpl.push(px[i], py[i]);
  }

  /** Regenerate base & rest points after resize. */
  private generatePoints() {
    const a = this.aspect;
    this.bx = []; this.by = [];

    // Boundary anchors (prevent degenerate edge triangles)
    for (let i = 0; i <= BOUND_DIV; i++) {
      const t = i / BOUND_DIV;
      this.bx.push(-a + t*2*a); this.by.push(-1);      // bottom
      this.bx.push(-a + t*2*a); this.by.push( 1);      // top
      this.bx.push(-a);         this.by.push(-1+t*2);  // left
      this.bx.push( a);         this.by.push(-1+t*2);  // right
    }

    // Interior jittered grid
    const cols = Math.max(4, Math.round(Math.sqrt(BASE_N * a)));
    const rows = Math.max(4, Math.round(BASE_N / cols));
    const dx = 2*a/cols, dy = 2/rows;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        this.bx.push(-a + (i+0.5+(Math.random()-0.5)*0.8)*dx);
        this.by.push(-1  + (j+0.5+(Math.random()-0.5)*0.8)*dy);
      }
    }

    // Rest positions for the refinement cloud (same density as base)
    this.rrx = []; this.rry = [];
    const rc = Math.max(2, Math.round(Math.sqrt(REFINE_N * a)));
    const rr = Math.max(2, Math.round(REFINE_N / rc));
    const rdx = 2*a/rc, rdy = 2/rr;
    outer: for (let i = 0; i < rc; i++) {
      for (let j = 0; j < rr; j++) {
        this.rrx.push(-a + (i+0.5+(Math.random()-0.5)*0.7)*rdx);
        this.rry.push(-1  + (j+0.5+(Math.random()-0.5)*0.7)*rdy);
        if (this.rrx.length >= REFINE_N) break outer;
      }
    }

    // Initialise current refine positions to rest
    for (let i = 0; i < REFINE_N; i++) {
      this.rx[i] = this.rrx[i];
      this.ry[i] = this.rry[i];
    }
  }

  // ── Event handlers ──────────────────────────────────────────────────────────

  private onPointerMove = (e: PointerEvent) => {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.mouse.x = ((e.clientX - rect.left) / rect.width  * 2 - 1) * this.aspect;
    this.mouse.y =  -((e.clientY - rect.top)  / rect.height * 2 - 1);
    this.mouse.active = true;
  };

  private onPointerLeave = () => { this.mouse.active = false; };

  private onResize = () => {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    this.renderer.setSize(w, h, false);
    this.aspect = w / h;
    this.bgMaterial.uniforms.uAspect.value = this.aspect;
    this.camera.left = -this.aspect; this.camera.right = this.aspect;
    this.camera.top  =  1;           this.camera.bottom = -1;
    this.camera.updateProjectionMatrix();
    this.generatePoints();
  };

  // ── Mesh build + upload ──────────────────────────────────────────────────────

  private buildMesh() {
    const bLen = this.bx.length;
    const n    = bLen + REFINE_N;

    // Reuse arrays to avoid allocation
    if (this.allX.length !== n) { this.allX.length = n; this.allY.length = n; }
    for (let i = 0; i < bLen; i++)     { this.allX[i]      = this.bx[i]; this.allY[i]      = this.by[i]; }
    for (let i = 0; i < REFINE_N; i++) { this.allX[bLen+i] = this.rx[i]; this.allY[bLen+i] = this.ry[i]; }

    triangulate(this.allX, this.allY, n);

    const fp = this.fillPos, ep = this.edgePos;
    let fi = 0, ei = 0;
    for (let t = 0; t < _tN; t++) {
      const ax = this.allX[_tA[t]], ay = this.allY[_tA[t]];
      const bx = this.allX[_tB[t]], by = this.allY[_tB[t]];
      const cx = this.allX[_tC[t]], cy = this.allY[_tC[t]];

      fp[fi++]=ax; fp[fi++]=ay; fp[fi++]=0;
      fp[fi++]=bx; fp[fi++]=by; fp[fi++]=0;
      fp[fi++]=cx; fp[fi++]=cy; fp[fi++]=0;

      ep[ei++]=ax; ep[ei++]=ay; ep[ei++]=0;
      ep[ei++]=bx; ep[ei++]=by; ep[ei++]=0;
      ep[ei++]=bx; ep[ei++]=by; ep[ei++]=0;
      ep[ei++]=cx; ep[ei++]=cy; ep[ei++]=0;
      ep[ei++]=cx; ep[ei++]=cy; ep[ei++]=0;
      ep[ei++]=ax; ep[ei++]=ay; ep[ei++]=0;
    }

    this.fillAttr.needsUpdate = true;
    this.edgeAttr.needsUpdate = true;
    this.fillGeom.setDrawRange(0, fi / 3);
    this.edgeGeom.setDrawRange(0, ei / 3);
  }

  // ── Animation loop ───────────────────────────────────────────────────────────

  private animate = () => {
    this.rafId = requestAnimationFrame(this.animate);
    const dt = Math.min(0.05, this.clock.getDelta());
    this.time += dt;

    // Smooth mouse
    const k = 1 - Math.exp(-dt * 8);
    this.smoothMouse.x      += (this.mouse.x - this.smoothMouse.x) * k;
    this.smoothMouse.y      += (this.mouse.y - this.smoothMouse.y) * k;
    this.smoothMouse.active += ((this.mouse.active ? 1 : 0) - this.smoothMouse.active) * k;

    const ma = this.smoothMouse.active;
    const mx = this.smoothMouse.x, my = this.smoothMouse.y;

    // Animate refinement cloud: fast toward cursor, slow drift back to rest
    const speed = ma > 0.3 ? 9 : 4;
    const lk    = 1 - Math.exp(-dt * speed);
    for (let i = 0; i < REFINE_N; i++) {
      const tx = this.rrx[i] + (mx + this.rtpl[i*2]   * REFINE_R - this.rrx[i]) * ma;
      const ty = this.rry[i] + (my + this.rtpl[i*2+1] * REFINE_R - this.rry[i]) * ma;
      this.rx[i] += (tx - this.rx[i]) * lk;
      this.ry[i] += (ty - this.ry[i]) * lk;
    }

    // Shader uniforms
    this.bgMaterial.uniforms.uTime.value = this.time;
    (this.bgMaterial.uniforms.uMouse.value as THREE.Vector2).set(mx, my);
    this.bgMaterial.uniforms.uMouseActive.value = ma;

    this.buildMesh();
    this.renderer.render(this.scene, this.camera);
  };

  // ── Cleanup ──────────────────────────────────────────────────────────────────

  destroy() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.resizeObserver?.disconnect();
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerleave', this.onPointerLeave);
    document.removeEventListener('mouseleave', this.onPointerLeave);
    this.bgQuad.geometry.dispose();
    this.bgMaterial.dispose();
    this.fillGeom.dispose();
    (this.fillMesh.material as THREE.Material).dispose();
    this.edgeGeom.dispose();
    (this.edgeLines.material as THREE.Material).dispose();
    this.renderer.dispose();
  }
}

export function initMeshBackground(canvas: HTMLCanvasElement): () => void {
  const mb = new MeshBackground(canvas);
  return () => mb.destroy();
}
