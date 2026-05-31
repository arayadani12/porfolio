// Central project metadata. Add or edit projects here; pages and routes pick
// up changes automatically.

export type ProjectTag =
  | 'FEM'
  | 'Multiphysics'
  | 'Data-driven'
  | 'Optimization'
  | 'Homogenization'
  | 'Phase-field';

export interface Project {
  slug: string;
  title: string;
  shortDescription: string;
  tag: ProjectTag;
  /** Long-form description used on the detail page. Plain text paragraphs. */
  longDescription: string[];
  /** Optional path (relative to /public) for a hero image. TODO: add real images. */
  heroImage?: string;
  /** Optional reference to a simulation data file in /public/data/<slug>/... */
  dataFile?: string;
}

export const projects: Project[] = [
  {
    slug: 'topological-optimization',
    title: 'Topological Optimization of Elastic Structures',
    shortDescription:
      'Density-based topology optimization (SIMP) for stiffness-maximizing structures under volume constraints.',
    tag: 'Optimization',
    longDescription: [
      // TODO: replace with real project write-up.
      'This project explores the SIMP (Solid Isotropic Material with Penalization) method for topology optimization. Given a design domain, loads, and supports, the algorithm distributes material to maximize stiffness for a fixed volume fraction.',
      'The implementation couples a finite-element solver (linear elasticity, Q4 plane-stress elements) with a sensitivity filter and an optimality-criteria update. Sensitivity filtering avoids checkerboard patterns and mesh dependence.',
      'Convergence behavior, mesh sensitivity, and the effect of the penalization exponent are studied across several benchmark problems (MBB beam, cantilever, L-bracket).',
      'Future work targets multi-material extensions and stress-constrained formulations using augmented-Lagrangian schemes.',
    ],
    // dataFile: '/data/topological-optimization/iteration_history.txt',
  },
  {
    slug: 'rve-homogenization',
    title: 'Computational Homogenization of Composite RVEs',
    shortDescription:
      'First-order FE² homogenization of fiber-reinforced composites using periodic boundary conditions.',
    tag: 'Homogenization',
    longDescription: [
      // TODO: replace with real project write-up.
      'Effective elastic and inelastic properties of fiber-reinforced composites are obtained by solving boundary-value problems on a representative volume element (RVE).',
      'Periodic boundary conditions are imposed via Lagrange-multiplier coupling between opposing faces, ensuring consistency with the Hill–Mandel macro-homogeneity condition.',
      'The pipeline takes a microstructure descriptor (volume fraction, fiber arrangement) and returns the full effective stiffness tensor as well as localization fields for post-processing.',
      'Comparisons with Mori–Tanaka and self-consistent estimates are included as sanity checks.',
    ],
    // dataFile: '/data/rve-homogenization/stress_strain.txt',
  },
  {
    slug: 'phase-field-damage-rccp',
    title: 'Phase-Field Damage in Roller-Compacted Concrete Pavements',
    shortDescription:
      'Brittle-fracture phase-field model coupled with a thermo-mechanical solver for RCCP slabs.',
    tag: 'Phase-field',
    longDescription: [
      // TODO: replace with real project write-up.
      'A variational phase-field formulation of brittle fracture is implemented to predict crack initiation and propagation in roller-compacted concrete pavements (RCCP).',
      'The model is coupled with a thermo-mechanical solver to capture restrained-shrinkage cracking during early-age curing.',
      'The non-convex coupled problem is solved with a staggered scheme alternating between displacement and damage subproblems. Convergence is monitored with energy-based residuals.',
      'Calibration against experimental beam tests is in progress; preliminary results reproduce the observed transverse cracking spacing.',
    ],
    // dataFile: '/data/phase-field-damage-rccp/crack_path.txt',
  },
];

export function getProject(slug: string): Project | undefined {
  return projects.find((p) => p.slug === slug);
}
