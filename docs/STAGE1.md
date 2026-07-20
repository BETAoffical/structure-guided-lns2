# Stage 1: Controlled Warehouse Dataset

Map generation and task generation remain separate. One deterministic map is
paired with four independent pressure levels without changing its geometry.

## Active map families

All maps are `28 x 39`, use two-cell ordinary aisles and beltways, and keep all
traversable cells connected.

- `regular_beltway` is the clean structural control.
- `compartmentalized` has exactly two walls and four single-cell gates. Its
  template is one horizontal plus one vertical wall, two horizontal walls, or
  two vertical walls.
- `dead_end_aisles` has exactly two horizontal and two vertical
  shelf-connected caps.

Across all 12 compartment maps, the three templates occur four times each.
The master seed shuffles their map assignment; wall, gate, and cap positions
remain seed-driven.

## Task variants

Every map receives four independent tasks:

| Variant | Agents | Demand |
| --- | ---: | --- |
| `balanced_base` | 36 | 18 left-to-right, 18 right-to-left |
| `balanced_dense` | 60 | 30 left-to-right, 30 right-to-left |
| `balanced_clustered` | 48 | 24/24 with two endpoint hotspots per side |
| `uniform_control` | 36 | Uniform free-cell start/goal sampling |

Starts and goals remain unique within each task. The clustered task uses
hotspot probability `0.7` and radius `4`.

## Dataset and metadata

The full dataset contains 36 maps and 144 task instances. MovingAI `.map` and
`.scen` are the active solver inputs; JSON metadata and legacy `.mapf` are also
written:

- train: 18 maps, 72 instances;
- validation: 6 maps, 24 instances;
- test: 12 maps, 48 instances.

`layout_counts` controls map-family quotas. `layout_variants` controls
structure templates, while `task_variants` supplies ordered per-task
overrides. Manifests record variant names, Agent counts, wall/gate counts,
horizontal/vertical dead-end counts, seeds, and output files.

## Preview behavior

Default SVGs use a minimal palette: light roads, dark shelves, brown structural
walls/caps, and blue stations. Semantic aisle colors and the structural prior
are available only through diagnostic rendering.

The gallery reads three real Validation maps. Task overlays are collapsed by
default so the geometry remains legible.

## Commands

```powershell
python scripts/generate_dataset.py `
  --config configs/stage1_example.json `
  --output build/feasibility-dataset

python scripts/inspect_dataset.py --dataset build/feasibility-dataset
python scripts/generate_gallery.py
python -m unittest tests.data.test_stage1_generators -v
```
