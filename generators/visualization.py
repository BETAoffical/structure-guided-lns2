from __future__ import annotations

import html

from .models import MapData, TaskData


def ascii_preview(
    map_data: MapData, task_data: TaskData | None = None
) -> str:
    semantic = map_data.metadata["semantic_cell_types"]
    display = [
        ["#" if cell == "@" else cell for cell in row] for row in semantic
    ]
    if task_data is not None:
        for row, col in task_data.starts:
            display[row][col] = "s"
        for row, col in task_data.goals:
            display[row][col] = (
                "*" if display[row][col] == "s" else "g"
            )
    legend = (
        "# shelf, B beltway, H/V aisle, X intersection, "
        "S service, P station"
    )
    if task_data is not None:
        legend += ", s start, g goal, * start+goal"
    return "\n".join(["".join(row) for row in display] + ["", legend])


def svg_preview(
    map_data: MapData,
    task_data: TaskData | None = None,
    cell_size: int = 12,
    diagnostic: bool = False,
) -> str:
    semantic = map_data.metadata["semantic_cell_types"]
    prior = map_data.metadata["structural_congestion_prior"]
    width = map_data.cols * cell_size
    grid_height = map_data.rows * cell_size
    legend_height = 30
    height = grid_height + legend_height
    if diagnostic:
        colors = {
            "@": "#263238",
            "W": "#5d4037",
            "N": "#795548",
            ".": "#ffffff",
            "B": "#e8f5e9",
            "H": "#e3f2fd",
            "V": "#e3f2fd",
            "X": "#fff3e0",
            "S": "#f3e5f5",
            "P": "#1565c0",
        }
    else:
        colors = {
            "@": "#30363a",
            "W": "#8d5a44",
            "N": "#8d5a44",
            ".": "#fafbfc",
            "B": "#fafbfc",
            "H": "#fafbfc",
            "V": "#fafbfc",
            "X": "#fafbfc",
            "S": "#fafbfc",
            "P": "#1976d2",
        }
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        (
            '<marker id="arrow" markerWidth="7" markerHeight="7" '
            'refX="6" refY="3.5" orient="auto">'
            '<path d="M0,0 L7,3.5 L0,7 z" fill="#7b1fa2"/></marker>'
        ),
        "</defs>",
        f"<title>{html.escape(map_data.map_id)}</title>",
        (
            f'<rect x="0" y="0" width="{width}" height="{height}" '
            'fill="#ffffff"/>'
        ),
    ]
    for row in range(map_data.rows):
        for col in range(map_data.cols):
            kind = semantic[row][col]
            parts.append(
                f'<rect x="{col * cell_size}" y="{row * cell_size}" '
                f'width="{cell_size}" height="{cell_size}" '
                f'fill="{colors.get(kind, "#ffffff")}" '
                'stroke="#e1e6e9" stroke-width="0.3"/>'
            )
            if diagnostic and kind != "@" and prior[row][col] >= 0.65:
                opacity = min(0.65, prior[row][col] * 0.65)
                parts.append(
                    f'<rect x="{col * cell_size}" y="{row * cell_size}" '
                    f'width="{cell_size}" height="{cell_size}" '
                    f'fill="#ef5350" opacity="{opacity:.3f}"/>'
                )

    if task_data is not None:
        for start, goal in zip(task_data.starts, task_data.goals):
            sx = (start[1] + 0.5) * cell_size
            sy = (start[0] + 0.5) * cell_size
            gx = (goal[1] + 0.5) * cell_size
            gy = (goal[0] + 0.5) * cell_size
            parts.append(
                f'<line x1="{sx}" y1="{sy}" x2="{gx}" y2="{gy}" '
                'stroke="#7b1fa2" stroke-width="0.8" opacity="0.22" '
                'marker-end="url(#arrow)"/>'
            )
        for row, col in task_data.starts:
            parts.append(
                f'<circle cx="{(col + 0.5) * cell_size}" '
                f'cy="{(row + 0.5) * cell_size}" '
                f'r="{cell_size * 0.24}" fill="#2e7d32"/>'
            )
        for row, col in task_data.goals:
            parts.append(
                f'<circle cx="{(col + 0.5) * cell_size}" '
                f'cy="{(row + 0.5) * cell_size}" '
                f'r="{cell_size * 0.18}" fill="#c62828"/>'
            )
    if diagnostic:
        legend = [
            ("#263238", "shelf"),
            ("#5d4037", "closure/divider"),
            ("#795548", "narrowing"),
            ("#1565c0", "station"),
            ("#ef5350", "high prior"),
        ]
    else:
        legend = [
            ("#30363a", "shelf"),
            ("#8d5a44", "wall/closure"),
            ("#1976d2", "station"),
        ]
    if task_data is not None:
        legend.extend(
            [
                ("#2e7d32", "start"),
                ("#c62828", "goal"),
            ]
        )
    item_width = width / len(legend)
    legend_y = grid_height + 9
    for index, (color, label) in enumerate(legend):
        x = index * item_width + 4
        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y}" width="10" height="10" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 14:.1f}" y="{legend_y + 9}" '
            'font-family="Arial, sans-serif" font-size="8" '
            f'fill="#263238">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
