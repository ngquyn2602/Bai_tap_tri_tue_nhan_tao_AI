# Repository: https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI
"""
TÔ MÀU ĐỒ THỊ THỦ ĐỨC - INTERACTIVE PRO
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib

# Backend Agg chạy được trong môi trường headless như Jupyter/Colab.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. DỮ LIỆU MẶC ĐỊNH
# ============================================================

DEFAULT_REGIONS = [
    "Linh Xuân",
    "Bình Chiểu",
    "Linh Trung",
    "Tam Bình",
    "Tam Phú",
    "Hiệp Bình Phước",
    "Hiệp Bình Chánh",
    "Linh Đông",
    "Linh Tây",
    "Linh Chiểu",
    "Trường Thọ",
    "Bình Thọ",
]

DEFAULT_EDGES = [
    ("Linh Xuân", "Bình Chiểu"),
    ("Linh Xuân", "Linh Trung"),
    ("Bình Chiểu", "Tam Bình"),
    ("Bình Chiểu", "Tam Phú"),
    ("Linh Trung", "Tam Phú"),
    ("Linh Trung", "Linh Tây"),
    ("Tam Bình", "Hiệp Bình Phước"),
    ("Tam Bình", "Tam Phú"),
    ("Tam Phú", "Hiệp Bình Chánh"),
    ("Tam Phú", "Linh Đông"),
    ("Hiệp Bình Phước", "Hiệp Bình Chánh"),
    ("Hiệp Bình Chánh", "Linh Đông"),
    ("Linh Đông", "Linh Tây"),
    ("Linh Tây", "Linh Chiểu"),
    ("Linh Chiểu", "Trường Thọ"),
    ("Trường Thọ", "Bình Thọ"),
    ("Bình Thọ", "Linh Đông"),
    ("Bình Thọ", "Linh Chiểu"),
]

# Vị trí vẽ chỉ để minh họa. Không phải tọa độ địa lý chính xác.
DEFAULT_POSITIONS = {
    "Linh Xuân": (0.2, 4.2),
    "Bình Chiểu": (2.1, 4.2),
    "Linh Trung": (0.9, 3.2),
    "Tam Bình": (3.6, 3.2),
    "Tam Phú": (2.1, 2.6),
    "Hiệp Bình Phước": (5.4, 3.2),
    "Hiệp Bình Chánh": (5.1, 2.0),
    "Linh Đông": (3.1, 1.6),
    "Linh Tây": (1.25, 1.6),
    "Linh Chiểu": (0.8, 0.55),
    "Trường Thọ": (2.45, 0.15),
    "Bình Thọ": (4.15, 0.55),
}

COLOR_PALETTES = {
    3: [
        ("Đỏ", "#ffb3b3"),
        ("Xanh", "#b9dcff"),
        ("Vàng", "#fff0a6"),
    ],
    4: [
        ("Đỏ", "#ffb3b3"),
        ("Xanh", "#b9dcff"),
        ("Vàng", "#fff0a6"),
        ("Tím", "#dfc8ff"),
    ],
    5: [
        ("Đỏ", "#ffb3b3"),
        ("Xanh", "#b9dcff"),
        ("Vàng", "#fff0a6"),
        ("Tím", "#dfc8ff"),
        ("Xanh lá", "#c9f7c5"),
    ],
}

ALGORITHM_LABELS = {
    "normal": "Backtracking - Theo danh sách",
    "degree": "Backtracking - Bậc lớn trước",
    "mrv": "Backtracking - MRV + Degree",
    "greedy": "Greedy - Bậc lớn trước",
}


# ============================================================
# 2. DATACLASS
# ============================================================

@dataclass
class TreeNode:
    node_id: int
    parent_id: Optional[int]
    label: str
    status: str
    depth: int


@dataclass
class RunResult:
    found: bool
    assignment: Dict[str, str]
    tree_nodes: List[TreeNode]
    logs: List[str]
    output_dir: Path
    graph_image: Path
    tree_image: Path
    csv_file: Path
    html_file: Path
    stats: Dict[str, int]


# ============================================================
# 3. PARSE DỮ LIỆU CẠNH
# ============================================================

def edges_to_text(edges: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{a} - {b}" for a, b in edges)


def parse_edges_text(raw_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Nhận text dạng:
        Linh Xuân - Bình Chiểu
        Linh Xuân, Linh Trung

    Trả về:
        regions, edges
    """
    edges: List[Tuple[str, str]] = []
    regions: List[str] = []

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if " - " in line:
            parts = line.split(" - ", 1)
        elif "," in line:
            parts = line.split(",", 1)
        elif "-" in line:
            parts = line.split("-", 1)
        else:
            raise ValueError(
                f"Dòng {line_no} sai định dạng: {raw_line}\n"
                "Dùng dạng: Khu vực A - Khu vực B"
            )

        a = parts[0].strip()
        b = parts[1].strip()

        if not a or not b:
            raise ValueError(f"Dòng {line_no} thiếu tên khu vực.")

        if a == b:
            raise ValueError(f"Dòng {line_no}: một khu vực không thể kề chính nó.")

        edge = (a, b)
        reverse = (b, a)
        if edge not in edges and reverse not in edges:
            edges.append(edge)

        if a not in regions:
            regions.append(a)
        if b not in regions:
            regions.append(b)

    if not edges:
        raise ValueError("Danh sách cạnh đang rỗng.")

    return regions, edges


# ============================================================
# 4. SOLVER
# ============================================================

class GraphColoringSolver:
    def __init__(
        self,
        regions: List[str],
        edges: List[Tuple[str, str]],
        color_names: List[str],
        algorithm: str,
    ) -> None:
        self.regions = regions
        self.edges = edges
        self.color_names = color_names
        self.algorithm = algorithm
        self.graph = self._build_graph()

        self.assignment: Dict[str, str] = {}
        self.tree_nodes: List[TreeNode] = []
        self.logs: List[str] = []

        self.try_count = 0
        self.violation_count = 0
        self.backtrack_count = 0

    def _build_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = defaultdict(list)

        for a, b in self.edges:
            if b not in graph[a]:
                graph[a].append(b)
            if a not in graph[b]:
                graph[b].append(a)

        for region in self.regions:
            graph.setdefault(region, [])

        return dict(graph)

    def degree(self, region: str) -> int:
        return len(self.graph.get(region, []))

    def add_tree_node(
        self,
        parent_id: Optional[int],
        label: str,
        status: str,
        depth: int,
    ) -> int:
        node_id = len(self.tree_nodes)
        self.tree_nodes.append(
            TreeNode(
                node_id=node_id,
                parent_id=parent_id,
                label=label,
                status=status,
                depth=depth,
            )
        )
        return node_id

    def is_safe(self, region: str, color: str) -> Tuple[bool, str]:
        for neighbor in self.graph[region]:
            if self.assignment.get(neighbor) == color:
                return False, f"{neighbor} đã tô {color}"
        return True, "Hợp lệ"

    def available_colors(self, region: str) -> List[str]:
        result = []
        for color in self.color_names:
            ok, _ = self.is_safe(region, color)
            if ok:
                result.append(color)
        return result

    def select_region(self) -> Optional[str]:
        unassigned = [r for r in self.regions if r not in self.assignment]

        if not unassigned:
            return None

        if self.algorithm == "normal":
            return unassigned[0]

        if self.algorithm in {"degree", "greedy"}:
            return max(unassigned, key=lambda r: self.degree(r))

        # MRV + Degree:
        # Ưu tiên vùng có ít màu hợp lệ nhất.
        # Nếu hòa, chọn vùng có bậc lớn hơn.
        return min(
            unassigned,
            key=lambda r: (len(self.available_colors(r)), -self.degree(r)),
        )

    def solve(self) -> bool:
        self.assignment.clear()
        self.tree_nodes.clear()
        self.logs.clear()
        self.try_count = 0
        self.violation_count = 0
        self.backtrack_count = 0

        if self.algorithm == "greedy":
            return self.solve_greedy()

        root_id = self.add_tree_node(None, "Assignment = {}", "root", 0)

        self.logs.append("Bắt đầu thuật toán Backtracking.")
        self.logs.append(f"Thuật toán: {ALGORITHM_LABELS.get(self.algorithm, self.algorithm)}")
        self.logs.append(f"Số vùng: {len(self.regions)}")
        self.logs.append(f"Số cạnh: {len(self.edges)}")
        self.logs.append(f"Màu: {', '.join(self.color_names)}")
        self.logs.append("-" * 70)

        found = self._backtrack(parent_id=root_id, depth=0)

        self.logs.append("-" * 70)
        self.logs.append("KẾT LUẬN: " + ("Tìm được nghiệm hợp lệ." if found else "Không tìm được nghiệm."))

        return found

    def _backtrack(self, parent_id: int, depth: int) -> bool:
        if len(self.assignment) == len(self.regions):
            self.add_tree_node(
                parent_id,
                "HOÀN THÀNH\nTìm được nghiệm",
                "solution",
                depth + 1,
            )
            return True

        region = self.select_region()
        if region is None:
            return False

        self.logs.append(f"Chọn vùng: {region}")

        for color in self.color_names:
            self.try_count += 1
            ok, reason = self.is_safe(region, color)

            if ok:
                self.assignment[region] = color
                child_id = self.add_tree_node(
                    parent_id,
                    f"{region} = {color}\nOK",
                    "ok",
                    depth + 1,
                )
                self.logs.append(f"  Thử {region} = {color}: OK")

                if self._backtrack(child_id, depth + 1):
                    return True

                self.backtrack_count += 1
                self.logs.append(f"  Quay lui: bỏ {region} = {color}")

                self.tree_nodes[child_id].status = "backtrack"
                self.tree_nodes[child_id].label = f"{region} = {color}\nQuay lui"
                del self.assignment[region]
            else:
                self.violation_count += 1
                self.add_tree_node(
                    parent_id,
                    f"{region} = {color}\nVi phạm\n{reason}",
                    "violation",
                    depth + 1,
                )
                self.logs.append(f"  Thử {region} = {color}: VI PHẠM ({reason})")

        return False

    def solve_greedy(self) -> bool:
        root_id = self.add_tree_node(None, "Greedy Coloring", "root", 0)
        parent_id = root_id

        order = sorted(self.regions, key=lambda r: self.degree(r), reverse=True)

        self.logs.append("Bắt đầu thuật toán Greedy.")
        self.logs.append(f"Thứ tự tô: {' -> '.join(order)}")
        self.logs.append(f"Màu: {', '.join(self.color_names)}")
        self.logs.append("-" * 70)

        for depth, region in enumerate(order, start=1):
            assigned = False

            for color in self.color_names:
                self.try_count += 1
                ok, reason = self.is_safe(region, color)

                if ok:
                    self.assignment[region] = color
                    parent_id = self.add_tree_node(
                        parent_id,
                        f"{region} = {color}\nOK",
                        "ok",
                        depth,
                    )
                    self.logs.append(f"Gán {region} = {color}: OK")
                    assigned = True
                    break

                self.violation_count += 1
                self.add_tree_node(
                    parent_id,
                    f"{region} = {color}\nVi phạm\n{reason}",
                    "violation",
                    depth,
                )
                self.logs.append(f"Thử {region} = {color}: VI PHẠM ({reason})")

            if not assigned:
                self.logs.append(f"Greedy thất bại tại vùng {region}.")
                return False

        self.add_tree_node(
            parent_id,
            "HOÀN THÀNH\nGreedy tìm được nghiệm",
            "solution",
            len(order) + 1,
        )

        self.logs.append("-" * 70)
        self.logs.append("KẾT LUẬN: Greedy tìm được nghiệm hợp lệ.")
        return True

    def validate_solution(self) -> Tuple[bool, List[str]]:
        errors = []

        for region in self.regions:
            if region not in self.assignment:
                errors.append(f"{region} chưa được tô màu.")

        for a, b in self.edges:
            if self.assignment.get(a) == self.assignment.get(b):
                errors.append(f"{a} và {b} cùng màu {self.assignment.get(a)}.")

        return len(errors) == 0, errors


# ============================================================
# 5. VẼ HÌNH
# ============================================================

def ensure_positions(regions: List[str]) -> Dict[str, Tuple[float, float]]:
    positions = dict(DEFAULT_POSITIONS)
    missing = [r for r in regions if r not in positions]

    if not missing:
        return positions

    radius = 2.7
    center_x = 3.0
    center_y = 2.0
    total = len(missing)

    for i, region in enumerate(missing):
        angle = 2 * math.pi * i / max(total, 1)
        positions[region] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )

    return positions


def get_color_hex(color_name: str, color_count: int) -> str:
    for name, color_hex in COLOR_PALETTES[color_count]:
        if name == color_name:
            return color_hex
    return "#ffffff"


def draw_colored_graph(
    solver: GraphColoringSolver,
    color_count: int,
    output_path: Path,
) -> None:
    positions = ensure_positions(solver.regions)
    fig, ax = plt.subplots(figsize=(13, 8))

    for a, b in solver.edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1.4, zorder=1)

    for region in solver.regions:
        x, y = positions[region]
        color_name = solver.assignment.get(region, "Chưa tô")
        face_color = get_color_hex(color_name, color_count)

        ax.scatter(
            x,
            y,
            s=2300,
            color=face_color,
            edgecolors="#222222",
            linewidths=1.7,
            zorder=3,
        )

        label = f"{region}\n{color_name}\nkề: {solver.degree(region)}"
        ax.text(x, y, label, ha="center", va="center", fontsize=8.3, zorder=4)

    ax.set_title(
        "Kết quả tô màu đồ thị khu vực Thủ Đức",
        fontsize=15,
        fontweight="bold",
    )
    ax.axis("off")
    ax.margins(0.16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def shorten_label(text: str, width: int = 18, max_lines: int = 4) -> str:
    lines: List[str] = []

    for part in text.splitlines():
        wrapped = textwrap.wrap(part, width=width) or [part]
        lines.extend(wrapped)

    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]

    return "\n".join(lines)


def draw_backtracking_tree(
    tree_nodes: List[TreeNode],
    output_path: Path,
    max_nodes: int,
) -> None:
    if not tree_nodes:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Chưa có dữ liệu cây", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    visible_nodes = tree_nodes[:max_nodes]
    visible_ids = {node.node_id for node in visible_nodes}

    children: Dict[int, List[int]] = defaultdict(list)
    for node in visible_nodes:
        if node.parent_id is not None and node.parent_id in visible_ids:
            children[node.parent_id].append(node.node_id)

    root_id = visible_nodes[0].node_id
    positions: Dict[int, Tuple[float, float]] = {}
    next_x = [0.0]

    def layout(node_id: int, depth: int) -> float:
        child_list = children.get(node_id, [])

        if not child_list:
            x = next_x[0]
            next_x[0] += 1.0
        else:
            child_xs = [layout(child, depth + 1) for child in child_list]
            x = sum(child_xs) / len(child_xs)

        positions[node_id] = (x, -depth)
        return x

    layout(root_id, 0)

    width = max(14, min(34, next_x[0] * 0.65))
    height = max(8, min(26, max(node.depth for node in visible_nodes) * 1.3 + 2))

    fig, ax = plt.subplots(figsize=(width, height))

    for node in visible_nodes:
        if node.parent_id is None:
            continue
        if node.parent_id not in positions:
            continue

        x1, y1 = positions[node.parent_id]
        x2, y2 = positions[node.node_id]
        ax.plot([x1, x2], [y1, y2], color="#666666", linewidth=1.0, zorder=1)

    status_style = {
        "root": ("#f2f2f2", "#222222"),
        "ok": ("#d9fdd3", "#16831a"),
        "violation": ("#ffd6d6", "#b00020"),
        "backtrack": ("#eeeeee", "#777777"),
        "solution": ("#cce5ff", "#0057b8"),
    }

    for node in visible_nodes:
        x, y = positions[node.node_id]
        face_color, edge_color = status_style.get(node.status, ("#ffffff", "#222222"))

        ax.scatter(
            x,
            y,
            s=1850,
            color=face_color,
            edgecolors=edge_color,
            linewidths=1.5,
            zorder=3,
        )

        ax.text(
            x,
            y,
            shorten_label(node.label),
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )

    title = "Cây Backtracking / quá trình thử màu"
    if len(tree_nodes) > max_nodes:
        title += f" - hiển thị {max_nodes}/{len(tree_nodes)} node"

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. XUẤT CSV + HTML
# ============================================================

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def export_csv(solver: GraphColoringSolver, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Khu vực", "Màu", "Số vùng kề", "Các vùng kề"])

        for region in solver.regions:
            writer.writerow(
                [
                    region,
                    solver.assignment.get(region, ""),
                    solver.degree(region),
                    ", ".join(solver.graph[region]),
                ]
            )


def result_table_html(solver: GraphColoringSolver) -> str:
    rows = []
    for region in solver.regions:
        rows.append(
            f"""
            <tr style="background:#ffffff !important; color:#111827 !important; opacity:1 !important;">
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{html.escape(region)}</td>
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">
                    <span style="display:inline-block; background:#dbeafe !important; color:#003fc7 !important; border-radius:999px; padding:4px 12px; font-weight:900 !important; opacity:1 !important;">
                        {html.escape(solver.assignment.get(region, ""))}
                    </span>
                </td>
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{solver.degree(region)}</td>
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{html.escape(", ".join(solver.graph[region]))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def edges_table_html(solver: GraphColoringSolver) -> str:
    rows = []
    for index, (a, b) in enumerate(solver.edges, start=1):
        rows.append(
            f"""
            <tr style="background:#ffffff !important; color:#111827 !important; opacity:1 !important;">
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{index}</td>
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{html.escape(a)}</td>
                <td style="color:#111827 !important; background:#ffffff !important; opacity:1 !important; font-weight:700 !important;">{html.escape(b)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def export_html_report(
    solver: GraphColoringSolver,
    found: bool,
    color_count: int,
    algorithm: str,
    graph_image: Path,
    tree_image: Path,
    output_path: Path,
) -> None:
    graph_b64 = image_to_base64(graph_image)
    tree_b64 = image_to_base64(tree_image)

    valid, errors = solver.validate_solution()

    conclusion_ok = found and valid
    conclusion_class = "success" if conclusion_ok else "warning"
    conclusion_text = "Tìm được nghiệm hợp lệ" if conclusion_ok else "Chưa có nghiệm hợp lệ"

    if errors:
        validation_html = "<ul>" + "".join(f"<li>{html.escape(e)}</li>" for e in errors) + "</ul>"
    else:
        validation_html = "<p class='ok-text'>Không có cạnh nào bị trùng màu.</p>"

    logs_html = "<br>".join(html.escape(line) for line in solver.logs)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo cáo tô màu đồ thị Thủ Đức</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #f4f6fb;
            color: #1f2937;
        }}
        .header {{
            background: linear-gradient(135deg, #1d4ed8, #7c3aed);
            color: white;
            padding: 30px 42px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 30px;
        }}
        .header p {{
            margin: 8px 0 0;
            opacity: 0.95;
        }}
        .container {{
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 20px 40px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 18px;
        }}
        .card {{
            background: white;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }}
        .metric {{
            font-size: 28px;
            font-weight: 700;
            color: #1d4ed8;
        }}
        .metric-label {{
            font-size: 13px;
            color: #6b7280;
            margin-top: 4px;
        }}
        .success {{
            background: #dcfce7;
            color: #166534;
            border-left: 6px solid #22c55e;
        }}
        .warning {{
            background: #fee2e2;
            color: #991b1b;
            border-left: 6px solid #ef4444;
        }}
        h2 {{
            margin-top: 0;
            color: #111827;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
            padding: 10px;
            vertical-align: top;
        }}
        th {{
            background: #f9fafb;
            color: #374151;
        }}
        img {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background: white;
        }}
        .badge {{
            display: inline-block;
            background: #e0ecff;
            color: #1d4ed8;
            border-radius: 999px;
            padding: 4px 10px;
            font-weight: 600;
            font-size: 13px;
        }}
        .log {{
            background: #0f172a;
            color: #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            font-family: Consolas, Monaco, monospace;
            font-size: 13px;
            max-height: 430px;
            overflow: auto;
            line-height: 1.55;
        }}
        .ok-text {{
            color: #166534;
            font-weight: 600;
        }}
        .note {{
            color: #6b7280;
            font-size: 14px;
            margin-top: -6px;
        }}
        @media (max-width: 900px) {{
            .grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        @media (max-width: 560px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body><div class="report-root">
    <div class="header">
        <h1>Báo cáo tô màu đồ thị khu vực Thủ Đức</h1>
        <p>Thuật toán Graph Coloring: {html.escape(ALGORITHM_LABELS.get(algorithm, algorithm))}</p>
    </div>

    <div class="container">
        <div class="card {conclusion_class}">
            <h2>Kết luận</h2>
            <p><strong>{html.escape(conclusion_text)}</strong></p>
            <p>Điều kiện: hai khu vực kề nhau không được trùng màu.</p>
            {validation_html}
        </div>

        <div class="grid">
            <div class="card">
                <div class="metric">{len(solver.regions)}</div>
                <div class="metric-label">Số khu vực</div>
            </div>
            <div class="card">
                <div class="metric">{len(solver.edges)}</div>
                <div class="metric-label">Số cạnh kề nhau</div>
            </div>
            <div class="card">
                <div class="metric">{color_count}</div>
                <div class="metric-label">Số màu sử dụng</div>
            </div>
            <div class="card">
                <div class="metric">{solver.try_count}</div>
                <div class="metric-label">Số lần thử màu</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="metric">{solver.violation_count}</div>
                <div class="metric-label">Số nhánh vi phạm</div>
            </div>
            <div class="card">
                <div class="metric">{solver.backtrack_count}</div>
                <div class="metric-label">Số lần quay lui</div>
            </div>
            <div class="card">
                <div class="metric">{len(solver.tree_nodes)}</div>
                <div class="metric-label">Số node trong cây</div>
            </div>
            <div class="card">
                <div class="metric">{html.escape(algorithm.upper())}</div>
                <div class="metric-label">Chiến lược</div>
            </div>
        </div>

        <div class="card">
            <h2>Đồ thị sau khi tô màu</h2>
            <p class="note">Mỗi node là một khu vực. Mỗi cạnh biểu diễn quan hệ giáp ranh.</p>
            <img src="data:image/png;base64,{graph_b64}" alt="Đồ thị tô màu">
        </div>

        <div class="card">
            <h2>Cây Backtracking / quá trình thử màu</h2>
            <p class="note">Xanh lá: hợp lệ, đỏ: vi phạm, xám: quay lui, xanh dương: nghiệm.</p>
            <img src="data:image/png;base64,{tree_b64}" alt="Cây backtracking">
        </div>

        <div class="card">
            <h2>Bảng kết quả tô màu</h2>
            <table>
                <thead>
                    <tr>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Khu vực</th>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Màu</th>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Số vùng kề</th>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Các vùng kề</th>
                    </tr>
                </thead>
                <tbody>
                    {result_table_html(solver)}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Dữ liệu cạnh</h2>
            <table>
                <thead>
                    <tr>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">#</th>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Khu vực A</th>
                        <th style="color:#000000 !important; background:#dbeafe !important; opacity:1 !important; font-weight:900 !important;">Khu vực B</th>
                    </tr>
                </thead>
                <tbody>
                    {edges_table_html(solver)}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Log thuật toán</h2>
            <div class="log">{logs_html}</div>
        </div>
    </div>
</div></body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")



# ============================================================
# 7. HÀM HIỂN THỊ KẾT QUẢ RÕ CHỮ TRONG JUPYTER/COLAB
# ============================================================

def build_summary_html(solver, found, color_count, algorithm, graph_image, tree_image, csv_file, html_file):
    result_text = "TÌM ĐƯỢC NGHIỆM" if found else "KHÔNG CÓ NGHIỆM"
    result_bg = "#dcfce7" if found else "#fee2e2"
    result_color = "#14532d" if found else "#7f1d1d"

    assignment_rows = []
    for region in solver.regions:
        assignment_rows.append(f"""
        <tr style="background:#ffffff !important; color:#111827 !important; opacity:1 !important;">
            <td style="padding:8px 10px; color:#111827 !important; background:#ffffff !important; font-weight:800 !important; border-bottom:1px solid #d1d5db; opacity:1 !important;">
                {html.escape(region)}
            </td>
            <td style="padding:8px 10px; color:#111827 !important; background:#ffffff !important; font-weight:800 !important; border-bottom:1px solid #d1d5db; opacity:1 !important;">
                →
            </td>
            <td style="padding:8px 10px; color:#003fc7 !important; background:#ffffff !important; font-weight:900 !important; border-bottom:1px solid #d1d5db; opacity:1 !important;">
                {html.escape(solver.assignment.get(region, "Chưa tô"))}
            </td>
        </tr>
        """)

    return f"""
    <div style="
        background:#ffffff !important;
        color:#111827 !important;
        border:1px solid #d1d5db;
        border-radius:16px;
        padding:18px;
        margin:14px 0;
        font-family:Arial, Helvetica, sans-serif;
        box-shadow:0 8px 24px rgba(15,23,42,0.08);
        opacity:1 !important;
        filter:none !important;
        -webkit-filter:none !important;
    ">
        <h2 style="margin:0 0 12px; color:#111827 !important; font-weight:900 !important;">
            Tô màu đồ thị Thủ Đức - Kết quả chạy
        </h2>

        <div style="
            background:{result_bg} !important;
            color:{result_color} !important;
            border-left:6px solid {'#22c55e' if found else '#ef4444'};
            padding:12px 14px;
            border-radius:10px;
            margin-bottom:14px;
            font-weight:900 !important;
            opacity:1 !important;
        ">
            Kết quả: {result_text}
        </div>

        <table style="
            width:100%;
            border-collapse:collapse;
            background:#ffffff !important;
            color:#111827 !important;
            opacity:1 !important;
        ">
            <tr style="background:#dbeafe !important;">
                <th style="text-align:left; padding:8px 10px; background:#dbeafe !important; color:#000000 !important; font-weight:900 !important; opacity:1 !important;">Thông tin</th>
                <th style="text-align:left; padding:8px 10px; background:#dbeafe !important; color:#000000 !important; font-weight:900 !important; opacity:1 !important;">Giá trị</th>
            </tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Thuật toán</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{html.escape(ALGORITHM_LABELS[algorithm])}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số màu</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{color_count}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số khu vực</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{len(solver.regions)}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số cạnh</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{len(solver.edges)}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số lần thử màu</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{solver.try_count}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số nhánh vi phạm</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{solver.violation_count}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số lần quay lui</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{solver.backtrack_count}</td></tr>
            <tr><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">Số node cây</td><td style="padding:8px 10px; color:#111827 !important; font-weight:800 !important;">{len(solver.tree_nodes)}</td></tr>
        </table>

        <h3 style="color:#111827 !important; font-weight:900 !important; margin:16px 0 8px;">Bảng tô màu</h3>

        <table style="width:100%; border-collapse:collapse; background:#ffffff !important; color:#111827 !important; opacity:1 !important;">
            <tr style="background:#dbeafe !important;">
                <th style="text-align:left; padding:8px 10px; background:#dbeafe !important; color:#000000 !important; font-weight:900 !important;">Khu vực</th>
                <th style="text-align:left; padding:8px 10px; background:#dbeafe !important; color:#000000 !important; font-weight:900 !important;"></th>
                <th style="text-align:left; padding:8px 10px; background:#dbeafe !important; color:#000000 !important; font-weight:900 !important;">Màu</th>
            </tr>
            {''.join(assignment_rows)}
        </table>

        <h3 style="color:#111827 !important; font-weight:900 !important; margin:16px 0 8px;">File đã xuất</h3>
        <ul style="color:#111827 !important; font-weight:800 !important; line-height:1.8; opacity:1 !important;">
            <li style="color:#111827 !important;">Ảnh đồ thị: <code style="color:#003fc7 !important; font-weight:900 !important;">{html.escape(str(graph_image))}</code></li>
            <li style="color:#111827 !important;">Ảnh cây: <code style="color:#003fc7 !important; font-weight:900 !important;">{html.escape(str(tree_image))}</code></li>
            <li style="color:#111827 !important;">CSV: <code style="color:#003fc7 !important; font-weight:900 !important;">{html.escape(str(csv_file))}</code></li>
            <li style="color:#111827 !important;">HTML report: <code style="color:#003fc7 !important; font-weight:900 !important;">{html.escape(str(html_file))}</code></li>
        </ul>
    </div>
    """


def print_cli_summary(solver, found, color_count, algorithm, graph_image, tree_image, csv_file, html_file):
    print("=" * 78)
    print("TÔ MÀU ĐỒ THỊ THỦ ĐỨC - INTERACTIVE PRO")
    print("=" * 78)
    print("Kết quả:", "TÌM ĐƯỢC NGHIỆM" if found else "KHÔNG CÓ NGHIỆM")
    print("Thuật toán:", ALGORITHM_LABELS[algorithm])
    print("Số màu:", color_count)
    print("Số khu vực:", len(solver.regions))
    print("Số cạnh:", len(solver.edges))
    print("Số lần thử màu:", solver.try_count)
    print("Số nhánh vi phạm:", solver.violation_count)
    print("Số lần quay lui:", solver.backtrack_count)
    print("Số node cây:", len(solver.tree_nodes))
    print("-" * 78)
    print("BẢNG TÔ MÀU:")
    for region in solver.regions:
        print(f"{region:22s} -> {solver.assignment.get(region, 'Chưa tô')}")
    print("-" * 78)
    print("FILE ĐÃ XUẤT:")
    print("Ảnh đồ thị:", graph_image)
    print("Ảnh cây:", tree_image)
    print("CSV:", csv_file)
    print("HTML report:", html_file)
    print("=" * 78)


# ============================================================
# 7. HÀM CHẠY CHÍNH
# ============================================================

def run_coloring(
    color_count: int = 3,
    algorithm: str = "mrv",
    edges_text: Optional[str] = None,
    max_tree_nodes: int = 120,
    output_dir: str = "ket_qua_to_mau_thu_duc_interactive",
    show_report: bool = True,
) -> RunResult:
    """
    Hàm chạy chính.

    Tham số:
    - color_count: 3, 4 hoặc 5
    - algorithm:
        normal  = Backtracking theo danh sách
        degree  = Backtracking chọn bậc lớn trước
        mrv     = Backtracking MRV + Degree
        greedy  = Greedy bậc lớn trước
    - edges_text: text cạnh. Nếu None thì dùng dữ liệu mặc định.
    - max_tree_nodes: số node tối đa hiển thị trong ảnh cây.
    - output_dir: thư mục xuất kết quả.
    - show_report: True nếu muốn hiển thị HTML ngay trong notebook.
    """
    if color_count not in COLOR_PALETTES:
        raise ValueError("color_count phải là 3, 4 hoặc 5.")

    if algorithm not in ALGORITHM_LABELS:
        raise ValueError(f"algorithm phải thuộc: {', '.join(ALGORITHM_LABELS)}")

    if edges_text is None:
        regions = list(DEFAULT_REGIONS)
        edges = list(DEFAULT_EDGES)
    else:
        regions, edges = parse_edges_text(edges_text)

    color_names = [name for name, _ in COLOR_PALETTES[color_count]]

    solver = GraphColoringSolver(
        regions=regions,
        edges=edges,
        color_names=color_names,
        algorithm=algorithm,
    )

    found = solver.solve()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_image = out_dir / "do_thi_to_mau_thu_duc.png"
    tree_image = out_dir / "cay_backtracking_to_mau_thu_duc.png"
    csv_file = out_dir / "ket_qua_to_mau_thu_duc.csv"
    html_file = out_dir / "bao_cao_to_mau_thu_duc.html"

    draw_colored_graph(solver, color_count, graph_image)
    draw_backtracking_tree(solver.tree_nodes, tree_image, max_tree_nodes)
    export_csv(solver, csv_file)
    export_html_report(
        solver=solver,
        found=found,
        color_count=color_count,
        algorithm=algorithm,
        graph_image=graph_image,
        tree_image=tree_image,
        output_path=html_file,
    )

    stats = {
        "regions": len(solver.regions),
        "edges": len(solver.edges),
        "colors": color_count,
        "tries": solver.try_count,
        "violations": solver.violation_count,
        "backtracks": solver.backtrack_count,
        "tree_nodes": len(solver.tree_nodes),
    }

    if show_report:
        try:
            from IPython.display import HTML, display
            display(HTML(build_summary_html(
                solver=solver,
                found=found,
                color_count=color_count,
                algorithm=algorithm,
                graph_image=graph_image,
                tree_image=tree_image,
                csv_file=csv_file,
                html_file=html_file,
            )))
        except Exception:
            print_cli_summary(
                solver=solver,
                found=found,
                color_count=color_count,
                algorithm=algorithm,
                graph_image=graph_image,
                tree_image=tree_image,
                csv_file=csv_file,
                html_file=html_file,
            )
    else:
        print_cli_summary(
            solver=solver,
            found=found,
            color_count=color_count,
            algorithm=algorithm,
            graph_image=graph_image,
            tree_image=tree_image,
            csv_file=csv_file,
            html_file=html_file,
        )

    if show_report:
        try:
            from IPython.display import HTML, display
            report_html = html_file.read_text(encoding="utf-8")
            display(HTML(report_html))
        except Exception:
            print("Không hiển thị HTML trực tiếp được. Hãy mở file:", html_file)

    return RunResult(
        found=found,
        assignment=dict(solver.assignment),
        tree_nodes=list(solver.tree_nodes),
        logs=list(solver.logs),
        output_dir=out_dir,
        graph_image=graph_image,
        tree_image=tree_image,
        csv_file=csv_file,
        html_file=html_file,
        stats=stats,
    )


# ============================================================
# 8. GIAO DIỆN JUPYTER BẰNG IPYWIDGETS
# ============================================================

def launch_widget_app() -> None:
    """
    Mở giao diện chọn trực tiếp trong Jupyter/Colab.

    Giao diện mới có bố cục dashboard, thẻ hướng dẫn, vùng nhập dữ liệu rõ hơn,
    kiểm tra nhanh dữ liệu cạnh và hiển thị kết quả trong cùng một khu vực.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML, clear_output
    except Exception as exc:
        print("Không tải được ipywidgets.")
        print("Cài bằng lệnh: !pip install ipywidgets")
        print("Lỗi:", exc)
        print("\nChạy fallback bằng dòng lệnh:")
        print("run_coloring(color_count=3, algorithm='mrv')")
        return

    app_css = widgets.HTML(
        """
        <style>
            .gc-wrap {
                font-family: Arial, Helvetica, sans-serif;
                color: #0f172a;
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
                border: 1px solid #dbeafe;
                border-radius: 22px;
                padding: 18px;
                box-shadow: 0 18px 46px rgba(15, 23, 42, 0.10);
            }
            .gc-hero {
                display: grid;
                grid-template-columns: 1.4fr .8fr;
                gap: 16px;
                align-items: stretch;
                background: linear-gradient(135deg, #2563eb, #7c3aed);
                color: #fff;
                border-radius: 20px;
                padding: 24px;
                margin-bottom: 16px;
                overflow: hidden;
            }
            .gc-title { margin: 0; font-size: 28px; font-weight: 900; line-height: 1.2; }
            .gc-subtitle { margin: 8px 0 0; opacity: .95; font-size: 15px; line-height: 1.5; }
            .gc-pill-row { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
            .gc-pill {
                display: inline-block;
                background: rgba(255,255,255,.16);
                border: 1px solid rgba(255,255,255,.26);
                border-radius: 999px;
                padding: 7px 12px;
                font-weight: 800;
                font-size: 13px;
            }
            .gc-hero-card {
                background: rgba(255,255,255,.14);
                border: 1px solid rgba(255,255,255,.26);
                border-radius: 18px;
                padding: 16px;
            }
            .gc-hero-number { font-size: 34px; font-weight: 900; margin: 0; }
            .gc-hero-note { margin: 8px 0 0; font-size: 13px; line-height: 1.45; opacity: .95; }
            .gc-section {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 14px;
                box-shadow: 0 8px 24px rgba(15,23,42,.06);
            }
            .gc-section h3 { margin: 0 0 10px; color: #111827; font-size: 18px; }
            .gc-help {
                background: #eff6ff;
                border-left: 5px solid #2563eb;
                border-radius: 14px;
                padding: 12px 14px;
                line-height: 1.55;
                color: #111827;
                margin: 12px 0;
            }
            .gc-small { color:#64748b; font-size:13px; line-height:1.5; margin: 6px 0 0; }
            .gc-metric-grid {
                display:grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin: 12px 0 0;
            }
            .gc-metric {
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:12px;
            }
            .gc-metric b { display:block; font-size:22px; color:#2563eb; margin-bottom:3px; }
            .gc-metric span { font-size:12px; color:#64748b; font-weight:700; }
            @media(max-width: 850px){
                .gc-hero { grid-template-columns: 1fr; }
                .gc-metric-grid { grid-template-columns: repeat(2, 1fr); }
            }
        </style>
        """
    )

    title = widgets.HTML(
        """
        <div class="gc-wrap">
            <div class="gc-hero">
                <div>
                    <h1 class="gc-title">Tô màu đồ thị Thủ Đức</h1>
                    <p class="gc-subtitle">
                        Chọn thuật toán, số màu, chỉnh dữ liệu cạnh và chạy thử trực tiếp.
                        Hệ thống sẽ xuất ảnh đồ thị, cây thử màu, CSV và báo cáo HTML.
                    </p>
                    <div class="gc-pill-row">
                        <span class="gc-pill">Graph Coloring</span>
                        <span class="gc-pill">Backtracking</span>
                        <span class="gc-pill">MRV + Degree</span>
                        <span class="gc-pill">Greedy</span>
                    </div>
                </div>
                <div class="gc-hero-card">
                    <p class="gc-hero-number">AI Search</p>
                    <p class="gc-hero-note">
                        Bài toán minh họa cách AI tìm kiếm nghiệm sao cho hai khu vực kề nhau không trùng màu.
                    </p>
                </div>
            </div>
        """
    )

    color_dropdown = widgets.Dropdown(
        options=[
            ("3 màu: Đỏ, Xanh, Vàng", 3),
            ("4 màu: thêm Tím", 4),
            ("5 màu: thêm Xanh lá", 5),
        ],
        value=3,
        description="Số màu:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="100%"),
    )

    algorithm_dropdown = widgets.Dropdown(
        options=[
            ("Backtracking - MRV + Degree", "mrv"),
            ("Backtracking - Bậc lớn trước", "degree"),
            ("Backtracking - Theo danh sách", "normal"),
            ("Greedy - Bậc lớn trước", "greedy"),
        ],
        value="mrv",
        description="Thuật toán:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="100%"),
    )

    max_tree_slider = widgets.IntSlider(
        value=120,
        min=20,
        max=300,
        step=10,
        description="Max node cây:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="100%"),
    )

    output_text = widgets.Text(
        value="ket_qua_to_mau_thu_duc_interactive",
        description="Thư mục xuất:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="100%"),
    )

    edges_area = widgets.Textarea(
        value=edges_to_text(DEFAULT_EDGES),
        description="Dữ liệu cạnh:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="100%", height="280px"),
    )

    metrics_html = widgets.HTML()
    status_html = widgets.HTML()

    def refresh_metrics(*_args):
        try:
            regions, edges = parse_edges_text(edges_area.value)
            degrees = defaultdict(int)
            for a, b in edges:
                degrees[a] += 1
                degrees[b] += 1
            max_degree = max(degrees.values()) if degrees else 0
            metrics_html.value = f"""
            <div class="gc-metric-grid">
                <div class="gc-metric"><b>{len(regions)}</b><span>Khu vực</span></div>
                <div class="gc-metric"><b>{len(edges)}</b><span>Cạnh kề nhau</span></div>
                <div class="gc-metric"><b>{max_degree}</b><span>Bậc lớn nhất</span></div>
                <div class="gc-metric"><b>{color_dropdown.value}</b><span>Màu đang chọn</span></div>
            </div>
            """
            status_html.value = "<p class='gc-small'>Dữ liệu hợp lệ. Bạn có thể bấm chạy tô màu.</p>"
        except Exception as exc:
            metrics_html.value = """
            <div class="gc-metric-grid">
                <div class="gc-metric"><b>!</b><span>Cần sửa dữ liệu</span></div>
            </div>
            """
            status_html.value = f"<p class='gc-small' style='color:#b91c1c;font-weight:800;'>Lỗi dữ liệu: {html.escape(str(exc))}</p>"

    edges_area.observe(refresh_metrics, names="value")
    color_dropdown.observe(refresh_metrics, names="value")

    guide = widgets.HTML(
        """
        <div class="gc-help">
            <b>Cách nhập dữ liệu cạnh:</b><br>
            Mỗi dòng là một quan hệ giáp ranh, ví dụ: <code>Linh Xuân - Bình Chiểu</code>.<br>
            Bạn có thể thay toàn bộ danh sách này bằng dữ liệu đồ thị khác. Không nhập một vùng kề với chính nó.
        </div>
        """
    )

    run_button = widgets.Button(
        description="Chạy tô màu",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="170px", height="42px"),
    )

    reset_button = widgets.Button(
        description="Reset dữ liệu",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="170px", height="42px"),
    )

    sample_button = widgets.Button(
        description="Xem nhanh dữ liệu",
        button_style="info",
        icon="search",
        layout=widgets.Layout(width="180px", height="42px"),
    )

    output = widgets.Output()

    control_section = widgets.VBox(
        [
            widgets.HTML('<div class="gc-section"><h3>1. Thiết lập thuật toán</h3>'),
            widgets.HBox(
                [color_dropdown, algorithm_dropdown],
                layout=widgets.Layout(gap="12px", width="100%"),
            ),
            max_tree_slider,
            output_text,
            widgets.HTML('</div>'),
        ],
        layout=widgets.Layout(width="100%"),
    )

    data_section = widgets.VBox(
        [
            widgets.HTML('<div class="gc-section"><h3>2. Dữ liệu đồ thị</h3>'),
            guide,
            edges_area,
            metrics_html,
            status_html,
            widgets.HBox([run_button, reset_button, sample_button], layout=widgets.Layout(gap="10px", margin="12px 0 0")),
            widgets.HTML('</div>'),
        ],
        layout=widgets.Layout(width="100%"),
    )

    result_section = widgets.VBox(
        [
            widgets.HTML('<div class="gc-section"><h3>3. Kết quả chạy</h3><p class="gc-small">Kết quả sẽ hiện ngay bên dưới sau khi bấm chạy.</p>'),
            output,
            widgets.HTML('</div></div>'),
        ],
        layout=widgets.Layout(width="100%"),
    )

    def on_reset_clicked(_button):
        edges_area.value = edges_to_text(DEFAULT_EDGES)
        color_dropdown.value = 3
        algorithm_dropdown.value = "mrv"
        max_tree_slider.value = 120
        output_text.value = "ket_qua_to_mau_thu_duc_interactive"
        refresh_metrics()
        with output:
            clear_output()
            display(HTML("<div class='gc-help'><b>Đã reset dữ liệu mặc định.</b></div>"))

    def on_sample_clicked(_button):
        with output:
            clear_output()
            try:
                regions, edges = parse_edges_text(edges_area.value)
                display(HTML(f"""
                    <div class="gc-help">
                        <b>Kiểm tra dữ liệu:</b><br>
                        Số khu vực: <b>{len(regions)}</b><br>
                        Số cạnh: <b>{len(edges)}</b><br>
                        5 cạnh đầu: <code>{html.escape(', '.join(f'{a}-{b}' for a, b in edges[:5]))}</code>
                    </div>
                """))
            except Exception as exc:
                display(HTML(f"<div class='gc-help' style='border-left-color:#ef4444;background:#fef2f2;'><b>Lỗi:</b> {html.escape(str(exc))}</div>"))

    def on_run_clicked(_button):
        with output:
            clear_output()
            try:
                display(HTML("<div class='gc-help'><b>Đang chạy thuật toán...</b> Kết quả sẽ xuất thành ảnh, CSV và HTML.</div>"))
                result = run_coloring(
                    color_count=color_dropdown.value,
                    algorithm=algorithm_dropdown.value,
                    edges_text=edges_area.value,
                    max_tree_nodes=max_tree_slider.value,
                    output_dir=output_text.value,
                    show_report=True,
                )
                display(HTML(f"""
                    <div class="gc-help" style="background:#ecfdf5;border-left-color:#22c55e;">
                        <b>Hoàn tất.</b><br>
                        Thư mục xuất: <code>{html.escape(str(result.output_dir))}</code><br>
                        HTML report: <code>{html.escape(str(result.html_file))}</code>
                    </div>
                """))
            except Exception as exc:
                display(HTML(f"<div class='gc-help' style='border-left-color:#ef4444;background:#fef2f2;'><b>Lỗi khi chạy:</b> {html.escape(str(exc))}</div>"))

    run_button.on_click(on_run_clicked)
    reset_button.on_click(on_reset_clicked)
    sample_button.on_click(on_sample_clicked)

    refresh_metrics()
    display(widgets.VBox([app_css, title, control_section, data_section, result_section], layout=widgets.Layout(width="100%")))


# ============================================================
# 9. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tô màu đồ thị Thủ Đức - bản Interactive Pro.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "widget", "cli"],
        default="auto",
        help="auto: tự chọn, widget: giao diện Jupyter, cli: chạy dòng lệnh.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        choices=[3, 4, 5],
        default=3,
        help="Số màu.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["normal", "degree", "mrv", "greedy"],
        default="mrv",
        help="Thuật toán.",
    )
    parser.add_argument(
        "--max-tree-nodes",
        type=int,
        default=120,
        help="Số node tối đa hiển thị trong cây.",
    )
    parser.add_argument(
        "--output-dir",
        default="ket_qua_to_mau_thu_duc_interactive",
        help="Thư mục xuất kết quả.",
    )
    args, unknown = parser.parse_known_args()
    return args


def running_inside_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    if args.mode == "widget":
        launch_widget_app()
        return

    if args.mode == "cli":
        run_coloring(
            color_count=args.colors,
            algorithm=args.algorithm,
            max_tree_nodes=args.max_tree_nodes,
            output_dir=args.output_dir,
            show_report=running_inside_ipython(),
        )
        return

    # auto
    if running_inside_ipython():
        launch_widget_app()
    else:
        run_coloring(
            color_count=args.colors,
            algorithm=args.algorithm,
            max_tree_nodes=args.max_tree_nodes,
            output_dir=args.output_dir,
            show_report=False,
        )


if __name__ == "__main__":
    main()
