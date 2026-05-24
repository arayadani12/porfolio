// Adaptive triangular-mesh background.
//
// What it does:
//   * Renders a smooth scalar field (sum of low-frequency waves + a Gaussian
//     bump that follows the cursor) as a colored backdrop. This stands in for
//     an FEM solution / error estimator.
//   * Overlays a uniform base grid of quads (each split into 2 triangles).
//     Each base cell is recursively subdivided into 4 quads up to a target
//     refinement level, computed from:
//        - distance to the mouse  (closer  → higher level)
//        - local gradient of the scalar field (steeper → higher level)
//   * Edges drawn in electric blue, faces filled with a very low-opacity blue.
//
// Notes:
//   * Levels are integer; smoothness comes from a smoothed cursor position.
//   * T-junctions between cells of different levels are accepted — they don't
//     matter visually and avoiding them would balloon the implementation.

import * as THREE from 'three';

const BG_COLOR = '#0a0a0f';
const LINE_COLOR = '#4f8ef7';

const BASE_COLS = 24;     // base grid resolution along the wider axis
const MAX_LEVEL = 3;      // 4^MAX_LEVEL sub-quads per base cell at peak refinement
const MOUSE_RADIUS = 0.9; // world units within which mouse refinement decays to 0

// Worst-case sizes for the persistent typed-array buffers
const MAX_BASE_CELLS = BASE_COLS * 64; // generous row cap
const MAX_SUB_QUADS_PER_CELL = Math.pow(4, MAX_LEVEL);
const MAX_TRIS = MAX_BASE_CELLS * MAX_SUB_QUADS_PER_CELL * 2;

class MeshBackground {
  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;

  private bgMaterial: THREE.ShaderMaterial;
  private bgQuad: THREE.Mesh;

  private edges: THREE.LineSegments;
  private edgeGeometry: THREE.BufferGeometry;
  private edgeMaterial: THREE.LineBasicMaterial;
  private edgePositions: Float32Array;
  private edgeAttr: THREE.BufferAttribute;

  private fills: THREE.Mesh;
  private fillGeometry: THREE.BufferGeometry;
  private fillMaterial: THREE.MeshBasicMaterial;
  private fillPositions: Float32Array;
  private fillAttr: THREE.BufferAttribute;

  private mouse = { x: 0, y: 0, active: false };
  private smoothMouse = { x: 0, y: 0, active: 0 };
  private time = 0;
  private clock = new THREE.Clock();
  private rafId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  private aspect = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(BG_COLOR, 1.0);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
    this.camera.position.z = 1;

    // --- Background scalar-field quad ---------------------------------------
    const bgVertex = /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
      }
    `;
    const bgFragment = /* glsl */ `
      precision highp float;
      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uMouseActive;
      uniform float uAspect;
      varying vec2 vUv;

      // Must mirror sampleField() on the TS side so mesh refinement
      // visually tracks the rendered field.
      float field(vec2 p, float t) {
        float v = 0.0;
        v += sin(p.x * 1.8 + t * 0.25) * cos(p.y * 1.4 - t * 0.18);
        v += 0.55 * sin((p.x + p.y) * 2.6 - t * 0.21);
        v += 0.35 * cos(p.x * 3.7 - p.y * 2.2 + t * 0.12);
        float d = length(p - uMouse);
        v += 1.6 * exp(-d * d * 5.5) * uMouseActive;
        return v;
      }

      vec3 colormap(float t) {
        vec3 c0 = vec3(0.039, 0.039, 0.059); // #0a0a0f
        vec3 c1 = vec3(0.055, 0.090, 0.180);
        vec3 c2 = vec3(0.090, 0.220, 0.420);
        vec3 c3 = vec3(0.310, 0.557, 0.969); // #4f8ef7
        vec3 c4 = vec3(0.976, 0.451, 0.086); // #f97316
        vec3 col = mix(c0, c1, smoothstep(0.00, 0.30, t));
        col = mix(col, c2, smoothstep(0.30, 0.60, t));
        col = mix(col, c3, smoothstep(0.60, 0.85, t));
        col = mix(col, c4, smoothstep(0.85, 1.00, t));
        return col;
      }

      void main() {
        vec2 p = vUv * 2.0 - 1.0;
        p.x *= uAspect;
        float v = field(p, uTime);
        float t = clamp(v * 0.35 + 0.5, 0.0, 1.0);
        // Keep the backdrop subdued so mesh lines remain dominant.
        vec3 col = colormap(t) * 0.55 + vec3(0.039, 0.039, 0.059) * 0.45;
        gl_FragColor = vec4(col, 1.0);
      }
    `;
    this.bgMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uMouse: { value: new THREE.Vector2(0, 0) },
        uMouseActive: { value: 0 },
        uAspect: { value: 1 },
      },
      vertexShader: bgVertex,
      fragmentShader: bgFragment,
      depthTest: false,
      depthWrite: false,
    });

    const quadGeom = new THREE.PlaneGeometry(2, 2);
    this.bgQuad = new THREE.Mesh(quadGeom, this.bgMaterial);
    this.bgQuad.frustumCulled = false;
    this.scene.add(this.bgQuad);

    // --- Fill (semi-transparent triangle interiors) -------------------------
    this.fillPositions = new Float32Array(MAX_TRIS * 3 * 3);
    this.fillAttr = new THREE.BufferAttribute(this.fillPositions, 3);
    this.fillAttr.setUsage(THREE.DynamicDrawUsage);
    this.fillGeometry = new THREE.BufferGeometry();
    this.fillGeometry.setAttribute('position', this.fillAttr);
    this.fillGeometry.setDrawRange(0, 0);
    this.fillMaterial = new THREE.MeshBasicMaterial({
      color: LINE_COLOR,
      transparent: true,
      opacity: 0.04,
      depthTest: false,
      depthWrite: false,
    });
    this.fills = new THREE.Mesh(this.fillGeometry, this.fillMaterial);
    this.fills.frustumCulled = false;
    this.scene.add(this.fills);

    // --- Edges --------------------------------------------------------------
    this.edgePositions = new Float32Array(MAX_TRIS * 3 * 2 * 3);
    this.edgeAttr = new THREE.BufferAttribute(this.edgePositions, 3);
    this.edgeAttr.setUsage(THREE.DynamicDrawUsage);
    this.edgeGeometry = new THREE.BufferGeometry();
    this.edgeGeometry.setAttribute('position', this.edgeAttr);
    this.edgeGeometry.setDrawRange(0, 0);
    this.edgeMaterial = new THREE.LineBasicMaterial({
      color: LINE_COLOR,
      transparent: true,
      opacity: 0.55,
      depthTest: false,
      depthWrite: false,
    });
    this.edges = new THREE.LineSegments(this.edgeGeometry, this.edgeMaterial);
    this.edges.frustumCulled = false;
    this.scene.add(this.edges);

    this.onResize();
    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(this.canvas);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerleave', this.onPointerLeave);
    document.addEventListener('mouseleave', this.onPointerLeave);

    this.animate();
  }

  // ------------------------------------------------------------------ events

  private onPointerMove = (e: PointerEvent) => {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    this.mouse.x = nx * this.aspect;
    this.mouse.y = ny;
    this.mouse.active = true;
  };

  private onPointerLeave = () => {
    this.mouse.active = false;
  };

  private onResize = () => {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    this.renderer.setSize(w, h, false);
    this.aspect = w / h;
    this.bgMaterial.uniforms.uAspect.value = this.aspect;
    this.camera.left = -this.aspect;
    this.camera.right = this.aspect;
    this.camera.top = 1;
    this.camera.bottom = -1;
    this.camera.updateProjectionMatrix();
  };

  // ------------------------------------------------------------------ field

  /** Mirrors the GLSL `field` so refinement decisions match the visual. */
  private sampleField(x: number, y: number, t: number, ma: number, mx: number, my: number): number {
    let v = 0;
    v += Math.sin(x * 1.8 + t * 0.25) * Math.cos(y * 1.4 - t * 0.18);
    v += 0.55 * Math.sin((x + y) * 2.6 - t * 0.21);
    v += 0.35 * Math.cos(x * 3.7 - y * 2.2 + t * 0.12);
    const dx = x - mx, dy = y - my;
    v += 1.6 * Math.exp(-(dx * dx + dy * dy) * 5.5) * ma;
    return v;
  }

  private fieldGradMag(x: number, y: number, t: number, ma: number, mx: number, my: number): number {
    const h = 0.04;
    const fx = (this.sampleField(x + h, y, t, ma, mx, my) - this.sampleField(x - h, y, t, ma, mx, my)) / (2 * h);
    const fy = (this.sampleField(x, y + h, t, ma, mx, my) - this.sampleField(x, y - h, t, ma, mx, my)) / (2 * h);
    return Math.sqrt(fx * fx + fy * fy);
  }

  /** Integer refinement level for a cell centered at (cx, cy). */
  private targetLevel(cx: number, cy: number, t: number, ma: number, mx: number, my: number): number {
    let level = 0;
    if (ma > 0.01) {
      const dx = cx - mx, dy = cy - my;
      const d = Math.sqrt(dx * dx + dy * dy);
      // smooth falloff: 0 outside MOUSE_RADIUS, MAX_LEVEL at the cursor
      const mouseLvl = Math.max(0, MAX_LEVEL * (1 - d / MOUSE_RADIUS));
      level = Math.max(level, mouseLvl);
    }
    const g = this.fieldGradMag(cx, cy, t, ma, mx, my);
    const gradLvl = Math.min(MAX_LEVEL - 1, g * 1.2);
    level = Math.max(level, gradLvl);
    return Math.min(MAX_LEVEL, Math.round(level));
  }

  // ----------------------------------------------------------------- meshing

  private buildMesh(t: number, ma: number, mx: number, my: number) {
    const aspect = this.aspect;
    const cellSize = (2 * aspect) / BASE_COLS;
    const rows = Math.max(4, Math.ceil(2 / cellSize));
    const startX = -aspect;
    const startY = -1;

    const tris = this.fillPositions;
    const segs = this.edgePositions;
    let ti = 0; // tris float index
    let si = 0; // segs float index

    const pushTri = (
      ax: number, ay: number,
      bx: number, by: number,
      cx: number, cy: number,
    ) => {
      tris[ti++] = ax; tris[ti++] = ay; tris[ti++] = 0;
      tris[ti++] = bx; tris[ti++] = by; tris[ti++] = 0;
      tris[ti++] = cx; tris[ti++] = cy; tris[ti++] = 0;

      segs[si++] = ax; segs[si++] = ay; segs[si++] = 0;
      segs[si++] = bx; segs[si++] = by; segs[si++] = 0;

      segs[si++] = bx; segs[si++] = by; segs[si++] = 0;
      segs[si++] = cx; segs[si++] = cy; segs[si++] = 0;

      segs[si++] = cx; segs[si++] = cy; segs[si++] = 0;
      segs[si++] = ax; segs[si++] = ay; segs[si++] = 0;
    };

    // Iterative quad subdivision to keep recursion shallow & explicit.
    const subdivide = (x0: number, y0: number, s: number, targetLvl: number) => {
      // Subdivide depth-first using a small inline stack.
      // For MAX_LEVEL <= 4 a recursive call is fine, but inlining keeps GC pressure flat.
      const stack: number[] = [x0, y0, s, 0];
      while (stack.length > 0) {
        const lvl = stack.pop()!;
        const sz = stack.pop()!;
        const yy = stack.pop()!;
        const xx = stack.pop()!;
        if (lvl >= targetLvl) {
          // Emit two triangles for this leaf quad, alternating the diagonal
          // direction so the pattern doesn't show a single dominant axis.
          const ax = xx,      ay = yy;
          const bx = xx + sz, by = yy;
          const cx = xx + sz, cy = yy + sz;
          const dx = xx,      dy = yy + sz;
          // Pick diagonal based on integer grid parity for visual variety.
          const ix = Math.round((xx - startX) / sz);
          const iy = Math.round((yy - startY) / sz);
          if (((ix + iy) & 1) === 0) {
            pushTri(ax, ay, bx, by, cx, cy);
            pushTri(ax, ay, cx, cy, dx, dy);
          } else {
            pushTri(ax, ay, bx, by, dx, dy);
            pushTri(bx, by, cx, cy, dx, dy);
          }
        } else {
          const hs = sz / 2;
          stack.push(xx,      yy,      hs, lvl + 1);
          stack.push(xx + hs, yy,      hs, lvl + 1);
          stack.push(xx,      yy + hs, hs, lvl + 1);
          stack.push(xx + hs, yy + hs, hs, lvl + 1);
        }
      }
    };

    for (let i = 0; i < BASE_COLS; i++) {
      for (let j = 0; j < rows; j++) {
        const x0 = startX + i * cellSize;
        const y0 = startY + j * cellSize;
        const cx = x0 + cellSize / 2;
        const cy = y0 + cellSize / 2;
        const lvl = this.targetLevel(cx, cy, t, ma, mx, my);
        subdivide(x0, y0, cellSize, lvl);
      }
    }

    this.fillAttr.needsUpdate = true;
    this.edgeAttr.needsUpdate = true;
    // setDrawRange uses vertex count, not float count.
    this.fillGeometry.setDrawRange(0, ti / 3);
    this.edgeGeometry.setDrawRange(0, si / 3);
  }

  // ----------------------------------------------------------------- animate

  private animate = () => {
    this.rafId = requestAnimationFrame(this.animate);
    const dt = Math.min(0.05, this.clock.getDelta());
    this.time += dt;

    const k = 1 - Math.exp(-dt * 7);
    this.smoothMouse.x += (this.mouse.x - this.smoothMouse.x) * k;
    this.smoothMouse.y += (this.mouse.y - this.smoothMouse.y) * k;
    const targetActive = this.mouse.active ? 1 : 0;
    this.smoothMouse.active += (targetActive - this.smoothMouse.active) * k;

    this.bgMaterial.uniforms.uTime.value = this.time;
    (this.bgMaterial.uniforms.uMouse.value as THREE.Vector2).set(
      this.smoothMouse.x,
      this.smoothMouse.y,
    );
    this.bgMaterial.uniforms.uMouseActive.value = this.smoothMouse.active;

    this.buildMesh(this.time, this.smoothMouse.active, this.smoothMouse.x, this.smoothMouse.y);
    this.renderer.render(this.scene, this.camera);
  };

  // ----------------------------------------------------------------- cleanup

  destroy() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.resizeObserver?.disconnect();
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerleave', this.onPointerLeave);
    document.removeEventListener('mouseleave', this.onPointerLeave);
    this.bgQuad.geometry.dispose();
    this.bgMaterial.dispose();
    this.edgeGeometry.dispose();
    this.edgeMaterial.dispose();
    this.fillGeometry.dispose();
    this.fillMaterial.dispose();
    this.renderer.dispose();
  }
}

export function initMeshBackground(canvas: HTMLCanvasElement): () => void {
  const mb = new MeshBackground(canvas);
  return () => mb.destroy();
}
