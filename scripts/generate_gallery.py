from __future__ import annotations

import argparse
import collections
import html
import json
import shutil
from pathlib import Path
from typing import Any


LAYOUTS = (
    ("regular_beltway", "规则货架与双格通道"),
    ("compartmentalized", "两面分割墙与四个单格门"),
    ("dead_end_aisles", "两个横向与两个纵向死路"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_map(
    rows: list[dict[str, Any]], layout_mode: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["layout_mode"] == layout_mode:
            grouped[row["map_id"]].append(row)
    if not grouped:
        raise ValueError(f"validation has no {layout_mode} map")
    candidates = sorted(grouped.values(), key=lambda group: group[0]["map_id"])
    if layout_mode == "compartmentalized":
        candidates.sort(
            key=lambda group: (
                group[0].get("layout_variant") != "cross_four_gate",
                group[0]["map_id"],
            )
        )
    return candidates[0]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a three-map gallery from validation data."
    )
    parser.add_argument(
        "--dataset",
        default=str(project_root / "build" / "feasibility-dataset"),
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "build" / "feasibility-gallery"),
    )
    arguments = parser.parse_args()

    dataset = Path(arguments.dataset).resolve()
    validation = dataset / "validation"
    manifest_path = validation / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(
            f"missing {manifest_path}; generate the dataset first"
        )
    output = Path(arguments.output)
    if output.exists():
        shutil.rmtree(output)
    assets = output / "assets"
    assets.mkdir(parents=True)
    manifest = _read_jsonl(manifest_path)
    cards: list[str] = []

    for layout_mode, description in LAYOUTS:
        map_rows = _select_map(manifest, layout_mode)
        map_row = map_rows[0]
        task_row = next(
            (
                row
                for row in map_rows
                if row.get("task_variant") == "balanced_base"
            ),
            map_rows[0],
        )
        map_id = str(map_row["map_id"])
        task_id = str(task_row["task_id"])
        map_document = _read_json(
            validation / "maps" / f"{map_id}.json"
        )
        source_map_svg = validation / "maps" / f"{map_id}.svg"
        source_task_svg = validation / "instances" / f"{task_id}.svg"
        map_svg = assets / f"{layout_mode}_map.svg"
        task_svg = assets / f"{layout_mode}_task.svg"
        shutil.copy2(source_map_svg, map_svg)
        shutil.copy2(source_task_svg, task_svg)

        metadata = map_document["metadata"]
        shelf_cells = sum(
            row.count("S") for row in metadata["obstacle_type_layer"]
        )
        shelf_coverage = 100.0 * shelf_cells / (
            map_document["rows"] * map_document["cols"]
        )
        changes = metadata["structural_changes"]
        dividers = changes["compartment_gates"]
        dead_ends = changes["dead_end_caps"]
        orientations = collections.Counter(
            item.get("orientation", "vertical") for item in dead_ends
        )
        gate_count = sum(len(item["gate_cells"]) for item in dividers)
        variant = map_row.get("layout_variant") or "default"
        details = (
            f"模板 {variant}；墙 {len(dividers)}，门 {gate_count}；"
            f"横向死路 {orientations['horizontal']}，"
            f"纵向死路 {orientations['vertical']}"
        )
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f"<h2>{html.escape(layout_mode)}</h2>",
                    f"<p>{html.escape(description)}</p>",
                    (
                        f'<img src="assets/{layout_mode}_map.svg" '
                        f'alt="{html.escape(layout_mode)}">'
                    ),
                    (
                        "<dl>"
                        f"<dt>Validation ID</dt><dd>{html.escape(map_id)}</dd>"
                        f"<dt>尺寸</dt><dd>{map_document['rows']} × "
                        f"{map_document['cols']}</dd>"
                        f"<dt>货架覆盖率</dt><dd>{shelf_coverage:.1f}%</dd>"
                        f"<dt>结构</dt><dd>{html.escape(details)}</dd>"
                        "</dl>"
                    ),
                    "<details>",
                    (
                        f"<summary>查看任务预览（"
                        f"{task_row['agent_count']} Agents）</summary>"
                    ),
                    (
                        f'<img src="assets/{layout_mode}_task.svg" '
                        f'alt="{html.escape(task_id)}">'
                    ),
                    "</details>",
                    "</article>",
                ]
            )
        )

    document = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 1 小规模可行性地图</title>
  <style>
    body { margin: 0; background: #f3f5f6; color: #17212b;
           font: 15px/1.5 Arial, sans-serif; }
    header { padding: 28px max(24px, 5vw) 12px; }
    main { display: grid; grid-template-columns: repeat(auto-fit,
           minmax(360px, 1fr)); gap: 22px; padding: 18px max(24px, 5vw) 40px; }
    .card { background: white; border-radius: 12px; padding: 18px;
            box-shadow: 0 5px 18px #1c2b3a14; }
    h1, h2 { margin: 0 0 8px; }
    p { margin: 0 0 14px; color: #52616f; }
    img { width: 100%; height: auto; border: 1px solid #dce3e8; }
    dl { display: grid; grid-template-columns: 110px 1fr; gap: 5px 10px; }
    dt { color: #6b7785; } dd { margin: 0; font-weight: 600; }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: #355a74; margin-bottom: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Stage 1 小规模可行性地图</h1>
    <p>三张主图均来自正式 Validation 数据集；任务预览默认折叠。</p>
  </header>
  <main>
""" + "\n".join(cards) + """
  </main>
</body>
</html>
"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
