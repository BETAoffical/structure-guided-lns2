from __future__ import annotations

import csv
import html
import io
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json as _read_json, sha256_file


SCHEMA = "lns2.result_consolidation.v1"
ALLOWED_STATUSES = {
    "confirmed",
    "external_support",
    "unsupported",
    "inconclusive",
    "mechanism",
    "engineering",
}
STATUS_LABELS = {
    "confirmed": "已确认",
    "external_support": "外部支持但未确认",
    "unsupported": "未通过",
    "inconclusive": "证据不足",
    "mechanism": "机制证据",
    "engineering": "工程结果",
}
STATUS_COLORS = {
    "confirmed": "#19764a",
    "external_support": "#2878b5",
    "unsupported": "#bd3b32",
    "inconclusive": "#8a6d1d",
    "mechanism": "#6f56a5",
    "engineering": "#3d6f79",
}


class EvidenceVerificationError(ValueError):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _repository_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise EvidenceVerificationError(
            f"evidence paths must be repository-relative: {relative}"
        )
    root = root.resolve()
    resolved = (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise EvidenceVerificationError(f"evidence path escapes repository: {relative}")
    return resolved


def json_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(component)]
            except (ValueError, IndexError) as error:
                raise KeyError(path) from error
        elif isinstance(current, dict) and component in current:
            current = current[component]
        else:
            raise KeyError(path)
    return current


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if config.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        errors.append("experiments must be a non-empty list")
        experiments = []
    identifiers: set[str] = set()
    for experiment in experiments:
        identifier = str(experiment.get("id", ""))
        if not identifier or identifier in identifiers:
            errors.append(f"experiment id is missing or duplicated: {identifier!r}")
        identifiers.add(identifier)
        status = experiment.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{identifier}: unsupported status {status!r}")
        source = experiment.get("source", {})
        path = str(source.get("path", ""))
        digest = str(source.get("sha256", ""))
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            errors.append(f"{identifier}: source path must be repository-relative")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"{identifier}: source SHA256 must be lowercase hexadecimal")
        metric_ids: set[str] = set()
        for metric in experiment.get("metrics", []):
            metric_id = str(metric.get("id", ""))
            if not metric_id or metric_id in metric_ids:
                errors.append(f"{identifier}: missing or duplicated metric id {metric_id!r}")
            metric_ids.add(metric_id)
            if not metric.get("json_path"):
                errors.append(f"{identifier}/{metric_id}: json_path is required")
        for check in experiment.get("checks", []):
            if "json_path" not in check or "expected" not in check:
                errors.append(f"{identifier}: malformed source check")
    by_id = {experiment.get("id"): experiment for experiment in experiments}
    movingai = by_id.get("movingai_ood_closed_loop", {})
    if movingai.get("status") != "external_support":
        errors.append("MovingAI OOD must remain external_support")
    if movingai.get("decision") != "stop_cross_layout_claim_and_consolidate_results":
        errors.append("MovingAI OOD must retain its registered stopping decision")
    movingai_metrics = {
        metric.get("id"): metric.get("value") for metric in movingai.get("metrics", [])
    }
    if float(movingai_metrics.get("auc_improvement", 1.0)) >= 0.05:
        errors.append("MovingAI OOD AUC improvement must remain below the 5% gate")
    static_context = by_id.get("context_primary_audit", {})
    if static_context.get("status") == "confirmed":
        errors.append("static-context audit cannot be marked confirmed")
    boundary = config.get("research_boundary", {})
    if bool(boundary.get("static_context_increment_confirmed")):
        errors.append("the frozen boundary cannot confirm static context")
    if bool(boundary.get("movingai_cross_layout_confirmed")):
        errors.append("the frozen boundary cannot confirm MovingAI cross-layout transfer")
    if not bool(boundary.get("same_family_generalization_confirmed")):
        errors.append("the frozen boundary must preserve same-family confirmation")
    if bool(boundary.get("rl_trained")):
        errors.append("the frozen research boundary must record rl_trained=false")
    if errors:
        raise EvidenceVerificationError("; ".join(errors))


def _verify_commit(root: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def verify_build_sources(
    config: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    errors: list[str] = []
    sources: list[dict[str, Any]] = []
    commits: set[str] = set()
    for experiment in config["experiments"]:
        identifier = experiment["id"]
        source = experiment["source"]
        try:
            source_path = _repository_path(repository_root, source["path"])
        except EvidenceVerificationError as error:
            errors.append(str(error))
            continue
        if not source_path.is_file():
            errors.append(f"{identifier}: missing source {source['path']}")
            continue
        actual_hash = sha256_file(source_path)
        if actual_hash != source["sha256"]:
            errors.append(
                f"{identifier}: SHA256 mismatch for {source['path']}: "
                f"expected {source['sha256']}, got {actual_hash}"
            )
            continue
        try:
            source_json = _read_json(source_path)
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"{identifier}: cannot read source JSON: {error}")
            continue
        checked_paths = 0
        for item in [*experiment.get("checks", []), *experiment.get("metrics", [])]:
            path = item["json_path"]
            expected = item.get("expected", item.get("value"))
            try:
                actual = json_path(source_json, path)
            except KeyError:
                errors.append(f"{identifier}: missing JSON path {path}")
                continue
            if not _values_equal(actual, expected):
                errors.append(
                    f"{identifier}: {path} expected {expected!r}, got {actual!r}"
                )
            checked_paths += 1
        registration = experiment.get("registration", {})
        for key in ("commit", "amendment_commit", "result_commit"):
            if registration.get(key):
                commits.add(str(registration[key]))
        sources.append(
            {
                "experiment_id": identifier,
                "path": source["path"],
                "sha256": actual_hash,
                "checked_json_paths": checked_paths,
            }
        )
    missing_commits = sorted(commit for commit in commits if not _verify_commit(repository_root, commit))
    if missing_commits:
        errors.append(f"missing registered Git commits: {missing_commits}")

    parity = config["solver_baseline"]
    cmake_path = _repository_path(repository_root, parity["parity_source"])
    if not cmake_path.is_file():
        errors.append(f"missing parity source {parity['parity_source']}")
    elif parity["expected_path_sha256"] not in cmake_path.read_text(encoding="utf-8"):
        errors.append("official parity SHA256 is not registered in CMakeLists.txt")
    if errors:
        raise EvidenceVerificationError("\n".join(errors))
    return {
        "status": "passed",
        "source_count": len(sources),
        "commit_count": len(commits),
        "sources": sources,
        "parity_hash_registered": True,
    }


def _config_sha256(config_path: Path) -> str:
    return sha256_file(config_path)


def build_evidence_manifest(
    config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    experiments = []
    for experiment in config["experiments"]:
        experiments.append(
            {
                "id": experiment["id"],
                "title_zh": experiment["title_zh"],
                "category": experiment["category"],
                "status": experiment["status"],
                "decision": experiment["decision"],
                "data_boundary": experiment["data_boundary"],
                "claim": experiment["claim"],
                "document": experiment["document"],
                "registration": experiment["registration"],
                "source": experiment["source"],
                "metrics": [
                    {
                        key: metric[key]
                        for key in ("id", "label_zh", "value", "unit", "role")
                    }
                    for metric in experiment.get("metrics", [])
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "snapshot_date": config["snapshot_date"],
        "config_sha256": config_sha256,
        "active_claim_zh": config["active_claim_zh"],
        "research_boundary": config["research_boundary"],
        "solver_baseline": config["solver_baseline"],
        "experiments": experiments,
    }


def render_metrics_csv(manifest: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "experiment_id",
            "category",
            "status",
            "metric_id",
            "label_zh",
            "value",
            "unit",
            "role",
        ]
    )
    for experiment in manifest["experiments"]:
        for metric in experiment["metrics"]:
            value = metric["value"]
            if isinstance(value, float):
                value = format(value, ".15g")
            writer.writerow(
                [
                    experiment["id"],
                    experiment["category"],
                    experiment["status"],
                    metric["id"],
                    metric["label_zh"],
                    value,
                    metric["unit"],
                    metric["role"],
                ]
            )
    return stream.getvalue()


def _experiment(manifest: dict[str, Any], identifier: str) -> dict[str, Any]:
    for experiment in manifest["experiments"]:
        if experiment["id"] == identifier:
            return experiment
    raise KeyError(identifier)


def _metric(manifest: dict[str, Any], experiment_id: str, metric_id: str) -> Any:
    experiment = _experiment(manifest, experiment_id)
    for metric in experiment["metrics"]:
        if metric["id"] == metric_id:
            return metric["value"]
    raise KeyError(f"{experiment_id}.{metric_id}")


def _svg_bar_chart(
    title: str,
    rows: Iterable[tuple[str, float, str]],
    maximum: float,
    unit: str,
    threshold: float | None = None,
) -> str:
    rows = list(rows)
    width = 920
    left = 260
    right = 70
    top = 84
    row_height = 54
    height = top + row_height * len(rows) + 66
    plot_width = width - left - right
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="38" font-family="sans-serif" font-size="22" font-weight="700" '
        f'letter-spacing="0" fill="#17212b">{html.escape(title)}</text>',
    ]
    for index, (label, value, status) in enumerate(rows):
        y = top + index * row_height
        bar_width = max(0.0, min(plot_width, plot_width * value / maximum))
        color = STATUS_COLORS.get(status, "#65727e")
        lines.extend(
            [
                f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" '
                f'font-family="sans-serif" font-size="15" letter-spacing="0" fill="#263542">'
                f"{html.escape(label)}</text>",
                f'<rect x="{left}" y="{y}" width="{plot_width}" height="28" fill="#edf1f4"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.3f}" height="28" fill="{color}"/>',
                f'<text x="{min(left + bar_width + 9, width - 62):.3f}" y="{y + 20}" '
                f'font-family="sans-serif" font-size="14" letter-spacing="0" fill="#17212b">'
                f"{value:.2f}{html.escape(unit)}</text>",
            ]
        )
    if threshold is not None:
        x = left + plot_width * threshold / maximum
        lines.extend(
            [
                f'<line x1="{x:.3f}" y1="{top - 12}" x2="{x:.3f}" '
                f'y2="{top + row_height * len(rows) - 18}" stroke="#111111" '
                'stroke-width="2" stroke-dasharray="5 4"/>',
                f'<text x="{x + 6:.3f}" y="{height - 28}" font-family="sans-serif" '
                f'font-size="13" letter-spacing="0" fill="#111111">门槛 {threshold:.1f}{html.escape(unit)}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_figures(manifest: dict[str, Any]) -> dict[str, str]:
    offline = _svg_bar_chart(
        "自然分布独立确认：Pareto top-1 命中率",
        [
            (
                "仅候选来源特征",
                100 * _metric(manifest, "natural_distribution_confirmation", "proposal_top1"),
                "unsupported",
            ),
            (
                "动态状态 + 实际邻域",
                100 * _metric(manifest, "natural_distribution_confirmation", "realized_top1"),
                "confirmed",
            ),
            (
                "再加入静态上下文",
                100 * _metric(manifest, "natural_distribution_confirmation", "context_top1"),
                "external_support",
            ),
        ],
        55.0,
        "%",
    )
    closed_loop = _svg_bar_chart(
        "冻结策略相对 Adaptive 的固定预算冲突 AUC 改善",
        [
            (
                "6 张同族新地图",
                100 * _metric(manifest, "closed_loop_confirmation", "auc_improvement"),
                "confirmed",
            ),
            (
                "12 张同族地图 × 3 seeds",
                100 * _metric(manifest, "closed_loop_multiseed", "auc_improvement"),
                "confirmed",
            ),
            (
                "12 张 MovingAI 跨布局地图",
                100 * _metric(manifest, "movingai_ood_closed_loop", "auc_improvement"),
                "external_support",
            ),
        ],
        65.0,
        "%",
        threshold=5.0,
    )
    movingai = _svg_bar_chart(
        "MovingAI OOD：144 个 episode 的成功数",
        [
            (
                "官方 Adaptive",
                _metric(manifest, "movingai_ood_closed_loop", "adaptive_successes"),
                "inconclusive",
            ),
            (
                "固定 Target",
                _metric(manifest, "movingai_ood_closed_loop", "target_successes"),
                "mechanism",
            ),
            (
                "固定 Collision",
                _metric(manifest, "movingai_ood_closed_loop", "collision_successes"),
                "mechanism",
            ),
            (
                "固定 Random",
                _metric(manifest, "movingai_ood_closed_loop", "random_successes"),
                "mechanism",
            ),
            (
                "冻结 realized_dynamic v1",
                _metric(manifest, "movingai_ood_closed_loop", "realized_successes"),
                "external_support",
            ),
        ],
        144.0,
        "",
    )
    counts = Counter(experiment["status"] for experiment in manifest["experiments"])
    status_order = (
        "confirmed",
        "external_support",
        "mechanism",
        "engineering",
        "inconclusive",
        "unsupported",
    )
    audit = _svg_bar_chart(
        "冻结证据登记：实验结论数量（不是效果量）",
        [
            (STATUS_LABELS[status], float(counts[status]), status)
            for status in status_order
            if counts[status]
        ],
        float(max(counts.values())),
        "",
    )
    return {
        "offline_evidence.svg": offline,
        "closed_loop_evidence.svg": closed_loop,
        "movingai_ood.svg": movingai,
        "audit_outcomes.svg": audit,
    }


def _percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def _registration_text(registration: dict[str, Any]) -> str:
    status = registration["status"]
    commit = registration.get("commit")
    result = registration.get("result_commit")
    if status == "separate":
        amendment = registration.get("amendment_commit")
        middle = f"，数据前修订 `{amendment}`" if amendment else ""
        return f"独立预注册 `{commit}`{middle}，结果 `{result}`"
    if status == "same_commit":
        return f"实现与结果同一提交 `{result}`，不作为独立预注册"
    return f"登记状态：{status}"


def _report_document_link(document: str) -> str:
    if document.startswith(("https://", "http://")):
        return document
    path = Path(document)
    if path.parts[:1] == ("docs",):
        return Path(*path.parts[1:]).as_posix()
    return "../" + path.as_posix()


def render_chinese_report(
    config: dict[str, Any], manifest: dict[str, Any]
) -> str:
    stability = _experiment(manifest, "realized_neighborhood_stability")
    movingai = _experiment(manifest, "movingai_ood_closed_loop")
    figure_prefix = config["report_figure_prefix"].rstrip("/")
    negative_rows = [
        experiment
        for experiment in manifest["experiments"]
        if experiment["status"] in {"unsupported", "inconclusive", "mechanism"}
    ]
    evidence_rows = manifest["experiments"]
    related_rows = config["related_work"]
    lines = [
        "# InitLNS 动态显式邻域控制研究报告",
        "",
        "> 本报告由冻结证据清单生成。所有数值来自已登记的正式 JSON；未重新训练模型、采集标签或修改实验门槛。",
        "",
        "## 摘要",
        "",
        config["active_claim_zh"],
        "",
        "研究最初希望利用地图拓扑、静态 OD、agent 密度和当前冲突共同控制 InitLNS。两轮静态上下文审计均未证明这些全局特征在动态状态之外具有稳定增量价值。后续机制实验发现，名义 `(seed, rule, size)` 动作是随机邻域生成分布，而实际生成的 agent 集合具有更稳定的修复价值。因此，最终方法转为依据当前冲突状态、路径局部结构和实际候选集合进行显式邻域排序。",
        "",
        "冻结策略在同族未见地图、多 solver seed 的闭环实验中显著降低冲突 AUC；MovingAI 外部布局上成功数也提高，但主 AUC 改善为 4.105%，低于预注册 5% 门槛。因此，本项目确认的是同族地图上的动态显式邻域控制，不确认静态上下文迁移，也不宣称严格的跨布局泛化或 RL 结果。",
        "",
        "## 1. 问题定义",
        "",
        "项目研究 MAPF-LNS2 的 InitLNS 首次可行解阶段。PP 初始规划可能产生冲突，InitLNS 反复选择一组 agent、删除其路径并用 PP+SIPPS 修复，直到得到首个无冲突解。研究对象是高层邻域选择，不是已有可行解后的 anytime 成本优化，也不替换低层 PP+SIPPS。",
        "",
        "主要指标是固定 100 步冲突 AUC、成功率、time-to-feasible、修复次数和 SIPPS 搜索工作。SOC 是次要指标，因为研究边界终止于首个可行解。",
        "",
        "## 2. 完整 LNS2 基线",
        "",
        f"项目内置官方 MAPF-LNS2 提交 `{manifest['solver_baseline']['upstream_commit']}`。扩展仅增加单步修复 API、proposal/observer 接口、显式邻域动作和低层计数。官方模式在固定实例上的路径 SHA256 仍为 `{manifest['solver_baseline']['expected_path_sha256']}`。",
        "",
        "InitLNS 对照为 Adaptive、Target、Collision 和 Random；无冲突后的 anytime LNS 所用 RandomWalk、Intersection、Random 和 Adaptive 属于另一阶段，不混入本项目主指标。",
        "",
        "## 3. 数据设计",
        "",
        "开发数据使用 regular_beltway、compartmentalized 和 dead_end_aisles 三类互补结构，任务为 balanced/bottleneck 静态 OD，agent 数量主要为 80/100。零冲突任务保留为 PP 直接可行，高冲突任务不因困难被丢弃。训练、Validation、独立确认和 MovingAI OOD 使用不同地图与任务 seed。",
        "",
        "这些任务是静态 OD，不包含 release time、任务队列或 lifelong MAPF。MovingAI 外部确认固定使用 12 张 Random、Maze、Room、Warehouse 和 Game 地图、两个 random scenario、预注册 agent 数量及 solver seeds `[1,2,3]`，不按结果替换实例。",
        "",
        "## 4. 动作空间演变",
        "",
        "1. 最初动作是冲突 seed、Target/Collision/Random 和邻域大小 4/8/16。",
        "2. 八 trial 稳定性分析表明，同一名义动作会生成差异较大的 agent 集合，尤其 Random 和 Collision。",
        f"3. 固定实际 agent 集合后，动作 eta-squared 达到 {_metric(manifest, stability['id'], 'realized_eta'):.3f}，split-half Spearman 达到 {_metric(manifest, stability['id'], 'rank_spearman'):.3f}。",
        "4. 最终控制器先调用官方生成器获得最多 18 个去重显式邻域，再用冻结排序器选择一个集合执行修复。第一版不从头自回归生成 agent 子集。",
        "",
        "## 5. 方法",
        "",
        "`realized_dynamic` 使用生成动作前可见的信息：当前冲突数、冲突图分量与 degree、delay/path 分布、修复阶段、低层搜索历史，以及候选集合覆盖的内部/边界冲突边、冲突 agent/分量、selected-agent delay 与路径统计、路径重合、空间范围、局部障碍率、节点度和 articulation 暴露。",
        "",
        "模型是固定参数的 pairwise HistGradientBoostingClassifier。候选只在同一状态内构造 Pareto 支配对；trial 先按候选聚合，不能当作独立样本。候选得分来自对其他候选的预测胜率，候选哈希负责确定性平局。修复后路径、冲突下降、runtime 和 generated nodes 不进入输入特征。",
        "",
        "静态地图类别、OD、密度和流量统计只作为消融。它们没有进入最终冻结控制器。RL 始终处于门控状态，本项目没有训练 RL policy。",
        "",
        "## 6. 实验协议",
        "",
        "离线评估按地图留出，状态是有效独立单位；trial 和同地图任务不是额外独立地图样本。闭环策略必须从相同初始 fingerprint 开始，proposal 不得修改状态，非法动作不得回退到 Adaptive。所有门槛在读取确认标签前固定；失败结果也写入正式报告。",
        "",
        "![离线邻域排序证据](" + figure_prefix + "/offline_evidence.svg)",
        "",
        "## 7. 主要结果",
        "",
        "### 7.1 显式邻域与离线排序",
        "",
        f"显式重放覆盖 {_metric(manifest, stability['id'], 'state_count')} 个状态、{_metric(manifest, stability['id'], 'candidate_count')} 个邻域和 {_metric(manifest, stability['id'], 'outcome_count')} 个结果，完整性错误为零。自然分布独立确认中，`realized_dynamic` 的 Pareto top-1 从 {_percent(_metric(manifest, 'natural_distribution_confirmation', 'proposal_top1'))} 提高到 {_percent(_metric(manifest, 'natural_distribution_confirmation', 'realized_top1'))}，冲突 regret 降低 {_percent(_metric(manifest, 'natural_distribution_confirmation', 'regret_reduction'))}，10/12 张地图不劣。",
        "",
        "### 7.2 同族地图闭环",
        "",
        f"首轮 6 张新地图闭环中，两种策略均成功 24/24，冻结策略将冲突 AUC 降低 {_percent(_metric(manifest, 'closed_loop_confirmation', 'auc_improvement'))}，6/6 地图不劣。多 seed 确认覆盖 12 张新地图、144 个 episode；两种策略均成功 144/144，AUC 改善 {_percent(_metric(manifest, 'closed_loop_multiseed', 'auc_improvement'))}，11/12 地图不劣，三条 solver seed 流分别保持正改善。",
        "",
        "控制器 hardening 在保持所有 episode、transition、邻域和冲突轨迹完全一致的前提下，将回放中的 realized_dynamic 平均控制时间从 11.47 秒降至 0.455 秒，约减少 96%。多 seed 正式实验中端到端墙钟仍为 0.715 秒，对照为 0.273 秒，因此不宣称部署速度更快。",
        "",
        "![闭环冲突 AUC](" + figure_prefix + "/closed_loop_evidence.svg)",
        "",
        "### 7.3 MovingAI 外部布局",
        "",
        f"12 张 MovingAI 地图的 720 个五策略 episode 全部完成。冻结策略成功 {_metric(manifest, movingai['id'], 'realized_successes')}/144，Adaptive 为 {_metric(manifest, movingai['id'], 'adaptive_successes')}/144；冻结策略在 9/9 有修复状态的地图和 5/5 地图族上平均不劣，地图 bootstrap 下界为正。",
        "",
        f"但固定预算冲突 AUC 只改善 {_percent(_metric(manifest, movingai['id'], 'auc_improvement'), 3)}，低于预注册 5% 主门槛。严格 decision 保持 `{movingai['decision']}`。这属于广泛外部支持，不是确认的跨布局主张。",
        "",
        "![MovingAI 成功数](" + figure_prefix + "/movingai_ood.svg)",
        "",
        "## 8. 迁移边界",
        "",
        "静态地图结构是一类输入信息；跨地图迁移是一种训练/测试关系，两者并不等价。当前动态状态没有输入地图身份，但路径、冲突图、局部障碍率和 articulation 暴露是地图、OD、密度和历史修复共同作用后的结果，因此隐式包含与当前决策相关的局部地图信息。",
        "",
        "现有证据支持“动态局部表示在同族未见地图上泛化”。它不支持“手工全局地图/OD/密度特征提供额外迁移价值”。MovingAI 结果接近但未通过预注册主门槛，所以严格跨布局泛化仍未确认。",
        "",
        "## 9. 相关工作定位",
        "",
        "| 工作 | 主要阶段与方法 | 与本项目的边界 |",
        "| --- | --- | --- |",
    ]
    for work in related_rows:
        lines.append(
            f"| [{work['name']}]({work['url']}) | {work['method_zh']} | {work['boundary_zh']} |"
        )
    lines.extend(
        [
            "",
            "本项目不声称首次学习邻域、首次可变大小或首次将 RL 用于 LNS2。可辨认的研究位置是：在 InitLNS 从有冲突到首个可行解的阶段，排序由官方生成器产生的实际 agent 邻域，并系统报告静态上下文、长期信用和 repair order 的负面门控结果。",
            "",
            "## 10. 负面结果与停止规则",
            "",
            "| 实验 | 状态 | 结论 |",
            "| --- | --- | --- |",
        ]
    )
    for experiment in negative_rows:
        lines.append(
            f"| [{experiment['title_zh']}]({_report_document_link(experiment['document'])}) | "
            f"{STATUS_LABELS[experiment['status']]} | {experiment['claim']} |"
        )
    lines.extend(
        [
            "",
            "H4 审计虽然显示 23.7% oracle AUC 机会，但 split-half Spearman、Pareto Jaccard 和最佳集合重合均低于 0.5，不能形成稳定长期标签。repair order 会实质改变结果，但当前上下文 selector 只改善 1.16% 且降低可行率。增加 GBDT 容量造成跨地图过拟合，MLP、DeepSets、GNN 和图统计 GBDT 也未超过冻结 v1。按照预注册规则，这些结果停止了 RL 与继续调参。",
            "",
            "![实验结论登记](" + figure_prefix + "/audit_outcomes.svg)",
            "",
            "## 11. 局限",
            "",
            "- 最强确认仍来自三类人工结构，不能代表全部 MAPF 地图。",
            "- MovingAI 中只有 74/144 episode 进入修复，地图级区间较宽。",
            "- 冻结排序器依赖手工聚合特征，约 19.6% 的 MovingAI 被选特征超出开发范围。",
            "- 候选生成仍需枚举多个 seed/rule/size，虽然已批处理加速，但墙钟未稳定优于 Adaptive。",
            "- 稳定的长期价值标签和 RL 信用分配没有得到验证。",
            "",
            "## 12. 后续路线",
            "",
            "当前研究先以本报告收束。若开启独立新课题，优先验证 outcome-blind 的候选剪枝和批处理，在冻结 v1 选择质量不退化的前提下降低 proposal、特征和两两推理成本。新的时空图模型或 RL 必须使用新的预注册数据和门槛，不能用于改写本轮 MovingAI 结论。",
            "",
            "## 附录 A：冻结证据登记",
            "",
            "| 实验 | 类别 | 状态 | 登记 | 正式来源 | SHA256 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for experiment in evidence_rows:
        lines.append(
            f"| {experiment['title_zh']} | {experiment['category']} | "
            f"{STATUS_LABELS[experiment['status']]} | "
            f"{_registration_text(experiment['registration'])} | "
            f"`{experiment['source']['path']}` | `{experiment['source']['sha256'][:12]}...` |"
        )
    lines.extend(
        [
            "",
            "完整指标见 `artifacts/initlns-research-evidence-v1/evidence_manifest.json` 和 `metrics.csv`。严格复核命令：",
            "",
            "```powershell",
            "python scripts/consolidate_research_results.py --verify-build",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_result_consolidation(
    config_path: str | Path,
    output: str | Path,
    report_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_build: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[1]
    )
    config = _read_json(config_path)
    validate_config(config)
    verification = (
        verify_build_sources(config, root)
        if verify_build
        else {"status": "not_requested", "source_count": 0, "commit_count": 0}
    )
    manifest = build_evidence_manifest(config, _config_sha256(config_path))
    output = Path(output).resolve()
    _write_json(output / "evidence_manifest.json", manifest)
    _atomic_write_text(output / "metrics.csv", render_metrics_csv(manifest))
    _write_json(output / "verification.json", verification)
    figures = render_figures(manifest)
    for name, content in figures.items():
        _atomic_write_text(output / "figures" / name, content)
    report = render_chinese_report(config, manifest)
    _atomic_write_text(Path(report_path).resolve(), report)
    counts = Counter(experiment["status"] for experiment in manifest["experiments"])
    return {
        "schema": SCHEMA,
        "experiment_count": len(manifest["experiments"]),
        "status_counts": dict(sorted(counts.items())),
        "verification": verification["status"],
        "output": str(output),
        "report": str(Path(report_path).resolve()),
    }


__all__ = [
    "EvidenceVerificationError",
    "build_evidence_manifest",
    "json_path",
    "render_chinese_report",
    "render_figures",
    "render_metrics_csv",
    "run_result_consolidation",
    "sha256_file",
    "validate_config",
    "verify_build_sources",
]
