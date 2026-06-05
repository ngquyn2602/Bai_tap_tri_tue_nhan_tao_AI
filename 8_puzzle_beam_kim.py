# https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI

"""
Ứng dụng minh họa bài toán 8-Puzzle bằng 2 thuật toán:
1. Local Beam Search
2. Simulated Annealing

File này được viết để chạy trong Jupyter Notebook / Google Colab.
Mục tiêu chính:
- Có giao diện nhập Start / Goal.
- Có bảng Node, Frontier, Reached.
- Có bảng chạy từng bước.
- Có hiển thị ma trận 3x3 rõ ràng.
- Có thể xem mô phỏng từng bước bằng nút điều khiển.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import random

from IPython.display import HTML, display, clear_output
import ipywidgets as widgets


# ============================================================
# 1. KHAI BÁO KIỂU DỮ LIỆU VÀ CẤU HÌNH CƠ BẢN
# ============================================================

# Một trạng thái 8-puzzle được lưu bằng tuple 9 phần tử.
# Ví dụ:
# (1, 2, 3,
#  5, 0, 6,
#  4, 7, 8)
# Trong đó số 0 biểu diễn ô trống.
State = Tuple[int, ...]

# Trạng thái bắt đầu mặc định.
DEFAULT_START: State = (
    1, 2, 3,
    5, 0, 6,
    4, 7, 8
)

# Trạng thái đích mặc định.
DEFAULT_GOAL: State = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)

# Thứ tự sinh hành động.
# L: trái, R: phải, U: lên, D: xuống.
ACTION_ORDER = ["L", "R", "U", "D"]


@dataclass
class PuzzleNode:
    """
    Lớp Node dùng chung cho cả Beam Search và Simulated Annealing.

    id       : mã node, ví dụ N0, N1, N2
    state    : trạng thái ma trận 8-puzzle dạng tuple 9 số
    parent   : id của node cha
    action   : hành động từ node cha đến node hiện tại
    depth    : độ sâu của node trong cây tìm kiếm
    h        : giá trị heuristic
    path     : chuỗi đường đi từ Start đến node hiện tại
    """

    id: str
    state: State
    parent: Optional[str]
    action: Optional[str]
    depth: int
    h: int
    path: str


# ============================================================
# 2. CÁC HÀM XỬ LÝ TRẠNG THÁI 8-PUZZLE
# ============================================================

def parse_state(raw_text: str) -> State:
    """
    Chuyển dữ liệu người dùng nhập thành tuple trạng thái.

    Người dùng có thể nhập theo nhiều dạng:
    1 2 3
    5 0 6
    4 7 8

    hoặc:
    1 2 3 / 5 0 6 / 4 7 8

    Hàm sẽ kiểm tra:
    - Có đúng 9 số hay không.
    - Có đủ các số từ 0 đến 8 hay không.
    """

    cleaned = raw_text.replace("/", " ").replace(",", " ")
    numbers = [int(x) for x in cleaned.split()]

    if len(numbers) != 9:
        raise ValueError("Trạng thái phải có đúng 9 số.")

    if sorted(numbers) != list(range(9)):
        raise ValueError("Trạng thái phải chứa đủ các số từ 0 đến 8, không được trùng hoặc thiếu.")

    return tuple(numbers)


def state_to_text(state: State) -> str:
    """
    Chuyển state thành chuỗi một dòng để dùng trong bảng.
    """

    rows = []
    for i in range(0, 9, 3):
        rows.append(" ".join(str(x) for x in state[i:i + 3]))
    return " / ".join(rows)


def misplaced_tiles(state: State, goal: State) -> int:
    """
    Hàm heuristic h(n): số ô sai vị trí.

    Quy ước:
    - Không tính ô trống 0.
    - h càng nhỏ thì trạng thái càng gần Goal.
    - h = 0 nghĩa là đã đạt Goal.
    """

    total = 0
    for index, value in enumerate(state):
        if value != 0 and value != goal[index]:
            total += 1
    return total


def manhattan_distance(state: State, goal: State) -> int:
    """
    Hàm Manhattan Distance.

    Trong file này thuật toán mặc định dùng misplaced_tiles.
    Hàm Manhattan vẫn được viết thêm để nếu cần mở rộng thì có thể đổi heuristic dễ dàng.
    """

    distance = 0

    for value in range(1, 9):
        current_index = state.index(value)
        goal_index = goal.index(value)

        current_row, current_col = divmod(current_index, 3)
        goal_row, goal_col = divmod(goal_index, 3)

        distance += abs(current_row - goal_row) + abs(current_col - goal_col)

    return distance


def move_blank(state: State, action: str) -> Optional[State]:
    """
    Di chuyển ô trống 0 theo hành động L/R/U/D.

    L nghĩa là ô trống đi sang trái.
    R nghĩa là ô trống đi sang phải.
    U nghĩa là ô trống đi lên.
    D nghĩa là ô trống đi xuống.

    Nếu hành động không hợp lệ, hàm trả về None.
    """

    zero_index = state.index(0)
    row, col = divmod(zero_index, 3)

    if action == "L":
        new_row, new_col = row, col - 1
    elif action == "R":
        new_row, new_col = row, col + 1
    elif action == "U":
        new_row, new_col = row - 1, col
    elif action == "D":
        new_row, new_col = row + 1, col
    else:
        return None

    if new_row < 0 or new_row >= 3 or new_col < 0 or new_col >= 3:
        return None

    new_zero_index = new_row * 3 + new_col
    new_state = list(state)
    new_state[zero_index], new_state[new_zero_index] = new_state[new_zero_index], new_state[zero_index]

    return tuple(new_state)


def expand_state(state: State) -> List[Tuple[str, State]]:
    """
    Sinh các trạng thái con từ state hiện tại.

    Kết quả là danh sách gồm:
    (hành động, trạng thái mới)

    Thứ tự sinh con cố định theo ACTION_ORDER để bảng chạy ổn định.
    """

    children: List[Tuple[str, State]] = []

    for action in ACTION_ORDER:
        next_state = move_blank(state, action)
        if next_state is not None:
            children.append((action, next_state))

    return children


def node_index(node: PuzzleNode) -> int:
    """
    Lấy phần số trong mã node.

    Ví dụ:
    N0 -> 0
    N12 -> 12

    Hàm này dùng khi cần sắp xếp node ổn định.
    """

    try:
        return int(node.id.replace("N", ""))
    except Exception:
        return 0


def build_solution_path(nodes: List[PuzzleNode], goal_id: str) -> List[PuzzleNode]:
    """
    Truy vết đường đi lời giải từ Goal về Start thông qua parent.

    Cách làm:
    - Tạo dictionary id -> node.
    - Bắt đầu từ goal_id.
    - Lần ngược parent cho đến khi parent = None.
    - Đảo ngược danh sách để được đường đi Start -> Goal.
    """

    table: Dict[str, PuzzleNode] = {node.id: node for node in nodes}
    path: List[PuzzleNode] = []

    current = table[goal_id]

    while True:
        path.append(current)

        if current.parent is None:
            break

        current = table[current.parent]

    path.reverse()
    return path


# ============================================================
# 3. THUẬT TOÁN LOCAL BEAM SEARCH
# ============================================================

def local_beam_search(start: State, goal: State, beam_width: int = 2, max_level: int = 30):
    """
    Cài đặt Local Beam Search cho bài toán 8-puzzle.

    Ý tưởng:
    - Frontier ban đầu chỉ có Start.
    - Ở mỗi tầng, mở rộng toàn bộ node trong Frontier hiện tại.
    - Gom tất cả node con sinh ra thành danh sách candidates.
    - Sắp xếp candidates theo h tăng dần.
    - Chỉ giữ lại k node tốt nhất làm Frontier mới.

    Tham số:
    start      : trạng thái ban đầu
    goal       : trạng thái đích
    beam_width : số node tốt nhất được giữ lại ở mỗi tầng
    max_level  : giới hạn số tầng tìm kiếm

    Kết quả trả về:
    goal_node  : node đích nếu tìm thấy, ngược lại None
    all_nodes  : toàn bộ node đã tạo
    trace      : danh sách thông tin từng bước để hiển thị bảng
    """

    next_id = 0

    start_node = PuzzleNode(
        id=f"N{next_id}",
        state=start,
        parent=None,
        action=None,
        depth=0,
        h=misplaced_tiles(start, goal),
        path=""
    )
    next_id += 1

    frontier: List[PuzzleNode] = [start_node]
    reached_set = {start}
    reached_order: List[State] = [start]
    all_nodes: List[PuzzleNode] = [start_node]
    trace: List[Dict[str, Any]] = []
    goal_node: Optional[PuzzleNode] = None

    for level in range(max_level + 1):
        frontier_before = list(frontier)
        expanded_nodes: List[PuzzleNode] = []
        generated_nodes: List[PuzzleNode] = []
        candidates: List[PuzzleNode] = []

        # Kiểm tra xem Goal đã nằm trong Frontier chưa.
        for node in frontier_before:
            if node.state == goal:
                goal_node = node
                break

        if goal_node is not None:
            trace.append({
                "algo": "beam",
                "level": level,
                "frontier_before": frontier_before,
                "expanded": [],
                "generated": [],
                "selected": frontier_before,
                "reached": list(reached_order),
                "note": f"Goal đã có trong Frontier tại {goal_node.id}."
            })
            return goal_node, all_nodes, trace

        # Mở rộng từng node trong Frontier.
        for parent in frontier_before:
            expanded_nodes.append(parent)

            for action, child_state in expand_state(parent.state):
                # Beam Search trong bản này vẫn dùng Reached để tránh lặp trạng thái.
                if child_state in reached_set:
                    continue

                child_node = PuzzleNode(
                    id=f"N{next_id}",
                    state=child_state,
                    parent=parent.id,
                    action=action,
                    depth=parent.depth + 1,
                    h=misplaced_tiles(child_state, goal),
                    path=parent.path + action
                )
                next_id += 1

                all_nodes.append(child_node)
                generated_nodes.append(child_node)
                candidates.append(child_node)

                reached_set.add(child_state)
                reached_order.append(child_state)

                if child_state == goal:
                    goal_node = child_node

        # Chọn k node tốt nhất dựa trên h, sau đó depth, sau đó id.
        candidates.sort(key=lambda item: (item.h, item.depth, node_index(item)))
        selected_nodes = candidates[:beam_width]

        if goal_node is not None:
            selected_nodes = [goal_node]
            note = f"Sinh ra Goal tại {goal_node.id}, thuật toán dừng."
        elif len(selected_nodes) > 0:
            note = "Chọn các node có h nhỏ nhất: " + ", ".join(
                f"{node.id}(h={node.h})" for node in selected_nodes
            )
        else:
            note = "Không sinh được node mới, thuật toán dừng."

        trace.append({
            "algo": "beam",
            "level": level,
            "frontier_before": frontier_before,
            "expanded": expanded_nodes,
            "generated": generated_nodes,
            "selected": selected_nodes,
            "reached": list(reached_order),
            "note": note
        })

        if goal_node is not None or len(selected_nodes) == 0:
            break

        frontier = selected_nodes

    return goal_node, all_nodes, trace


# ============================================================
# 4. THUẬT TOÁN SIMULATED ANNEALING
# ============================================================

def simulated_annealing(
    start: State,
    goal: State,
    initial_temperature: float = 5.0,
    cooling_rate: float = 0.8,
    min_temperature: float = 0.01,
    max_steps: int = 100,
    random_seed: int = 0
):
    """
    Cài đặt thuật toán Simulated Annealing cho bài toán 8-puzzle.

    Ý tưởng:
    - Bắt đầu tại Start.
    - Ở mỗi bước, sinh các hàng xóm của trạng thái hiện tại.
    - Chọn ngẫu nhiên một hàng xóm làm next.
    - Nếu next tốt hơn, nhận luôn.
    - Nếu next tệ hơn, vẫn có thể nhận với xác suất p = e^(-delta / T).
    - Nhiệt độ T giảm dần theo công thức T = T * cooling_rate.

    Trong đó:
    delta = h(next) - h(current)

    Nếu delta < 0:
    - next tốt hơn current.
    - nhận next luôn.

    Nếu delta >= 0:
    - next không tốt hơn current.
    - nhận next theo xác suất.

    Kết quả trả về:
    goal_node : node đích nếu tìm thấy
    all_nodes : các node được nhận vào đường đi
    trace     : bảng chạy từng bước
    """

    random.seed(random_seed)

    next_id = 0

    current = PuzzleNode(
        id=f"N{next_id}",
        state=start,
        parent=None,
        action=None,
        depth=0,
        h=misplaced_tiles(start, goal),
        path=""
    )
    next_id += 1

    all_nodes: List[PuzzleNode] = [current]
    reached_set = {start}
    reached_order: List[State] = [start]
    trace: List[Dict[str, Any]] = []

    temperature = initial_temperature

    for step in range(max_steps):
        if current.state == goal:
            break

        if temperature <= min_temperature:
            break

        current_before = current

        # Frontier của Simulated Annealing là toàn bộ hàng xóm của current.
        frontier: List[Dict[str, Any]] = []
        for action, next_state in expand_state(current.state):
            frontier.append({
                "id": f"F{step}_{action}",
                "title": f"Đi {action}",
                "action": action,
                "state": next_state,
                "h": misplaced_tiles(next_state, goal)
            })

        if not frontier:
            break

        # Chọn ngẫu nhiên một hàng xóm.
        chosen = random.choice(frontier)
        delta = chosen["h"] - current_before.h

        # Nếu hàng xóm tốt hơn thì nhận luôn.
        if delta < 0:
            probability = 1.0
            random_value = None
            accepted = True
            reason = "Delta < 0 nên trạng thái mới tốt hơn, nhận luôn."
        else:
            probability = math.exp(-delta / temperature)
            random_value = random.random()
            accepted = random_value < probability

            if accepted:
                reason = "Trạng thái mới không tốt hơn nhưng được nhận theo xác suất."
            else:
                reason = "Trạng thái mới không được nhận vì r >= p."

        new_node: Optional[PuzzleNode] = None

        # Nếu được nhận thì tạo node mới trên đường đi.
        if accepted:
            new_node = PuzzleNode(
                id=f"N{next_id}",
                state=chosen["state"],
                parent=current_before.id,
                action=chosen["action"],
                depth=current_before.depth + 1,
                h=chosen["h"],
                path=current_before.path + chosen["action"]
            )
            next_id += 1

            all_nodes.append(new_node)
            current = new_node

        # Reached dùng để ghi nhận các trạng thái đã từng được chọn xét.
        if chosen["state"] not in reached_set:
            reached_set.add(chosen["state"])
            reached_order.append(chosen["state"])

        trace.append({
            "algo": "annealing",
            "step": step,
            "temperature": temperature,
            "current_before": current_before,
            "frontier": frontier,
            "chosen": chosen,
            "delta": delta,
            "probability": probability,
            "random_value": random_value,
            "accepted": accepted,
            "new_node": new_node,
            "current_after": current,
            "reached": list(reached_order),
            "note": reason
        })

        temperature *= cooling_rate

        if current.state == goal:
            break

    goal_node = current if current.state == goal else None
    return goal_node, all_nodes, trace


# ============================================================
# 5. CSS VÀ CÁC HÀM TẠO HTML HIỂN THỊ
# ============================================================

def inject_css():
    """
    Nạp CSS cho giao diện.
    Màu nền sáng, chữ đậm rõ để tránh bị chìm trong Jupyter.
    """

    display(HTML("""
    <style>
        * {
            font-family: Arial, sans-serif !important;
            box-sizing: border-box;
        }

        .main-title {
            background: linear-gradient(90deg, #1d4ed8, #0891b2);
            color: #ffffff !important;
            padding: 20px 24px;
            border-radius: 18px;
            font-size: 28px;
            font-weight: 900;
            margin: 12px 0 18px 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        }

        .box {
            background: #ffffff;
            color: #0f172a !important;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        .box h2, .box h3 {
            color: #0f172a !important;
            margin-top: 4px;
        }

        .info {
            background: #eff6ff;
            color: #1e3a8a !important;
            border-left: 6px solid #2563eb;
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0;
            font-size: 15px;
            line-height: 1.55;
        }

        .ok {
            background: #f0fdf4;
            color: #14532d !important;
            border-left: 6px solid #22c55e;
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0;
            font-size: 15px;
            line-height: 1.55;
        }

        .bad {
            background: #fff7ed;
            color: #7c2d12 !important;
            border-left: 6px solid #f97316;
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0;
            font-size: 15px;
            line-height: 1.55;
        }

        .card-area {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin: 12px 0;
        }

        .state-card {
            background: #ffffff;
            color: #0f172a !important;
            border: 1px solid #dbeafe;
            border-top: 5px solid #2563eb;
            border-radius: 16px;
            padding: 12px;
            min-width: 180px;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.08);
        }

        .state-card-title {
            text-align: center;
            font-weight: 900;
            color: #0f172a !important;
            margin-bottom: 8px;
        }

        .state-meta {
            text-align: center;
            color: #334155 !important;
            font-size: 13px;
            margin-top: 8px;
            line-height: 1.55;
        }

        table.puzzle-table {
            border-collapse: separate;
            border-spacing: 6px;
            margin: auto;
        }

        table.puzzle-table td {
            width: 38px;
            height: 38px;
            background: #dbeafe;
            color: #0f172a !important;
            border: 1px solid #93c5fd;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 900;
            text-align: center;
            vertical-align: middle;
        }

        table.puzzle-table td.empty-cell {
            background: #f8fafc;
            color: #94a3b8 !important;
            border: 1px dashed #94a3b8;
        }

        table.small-puzzle td {
            width: 30px !important;
            height: 30px !important;
            font-size: 15px !important;
            border-radius: 8px !important;
        }

        table.trace-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            color: #0f172a !important;
            margin: 12px 0;
            font-size: 14px;
        }

        table.trace-table th {
            background: #0f172a;
            color: #ffffff !important;
            padding: 10px;
            text-align: center;
            border: 1px solid #334155;
            vertical-align: middle;
        }

        table.trace-table td {
            padding: 10px;
            border: 1px solid #cbd5e1;
            color: #0f172a !important;
            text-align: center;
            vertical-align: top;
            background: #ffffff;
        }

        table.trace-table tr:nth-child(even) td {
            background: #f8fafc;
        }

        .mini-card {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 8px;
            min-width: 135px;
            display: inline-block;
            margin: 4px;
            vertical-align: top;
            color: #0f172a !important;
        }

        .mini-area {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }

        textarea, input {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #94a3b8 !important;
        }

        label, .widget-label {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
    </style>
    """))


def matrix_html(state: State, small: bool = False) -> str:
    """
    Tạo HTML hiển thị state dưới dạng ma trận 3x3.
    """

    css_class = "puzzle-table small-puzzle" if small else "puzzle-table"
    html = f"<table class='{css_class}'>"

    for i in range(0, 9, 3):
        html += "<tr>"
        for value in state[i:i + 3]:
            if value == 0:
                html += "<td class='empty-cell'>0</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += "</table>"
    return html


def state_card_html(
    state: State,
    title: str,
    h_value: Optional[int] = None,
    node_id: Optional[str] = None,
    action: Optional[str] = None,
    color: str = "#2563eb"
) -> str:
    """
    Tạo một thẻ hiển thị trạng thái.
    Thẻ này dùng trong phần mô phỏng từng bước.
    """

    meta_lines = []

    if node_id is not None:
        meta_lines.append(f"Node: <b>{node_id}</b>")

    if action is not None:
        meta_lines.append(f"Action: <b>{action}</b>")

    if h_value is not None:
        meta_lines.append(f"h = <b>{h_value}</b>")

    return f"""
    <div class="state-card" style="border-top-color:{color};">
        <div class="state-card-title">{title}</div>
        {matrix_html(state)}
        <div class="state-meta">{'<br>'.join(meta_lines)}</div>
    </div>
    """


def node_cards_html(nodes: List[PuzzleNode], color: str = "#2563eb") -> str:
    """
    Hiển thị danh sách PuzzleNode thành nhiều card.
    """

    if not nodes:
        return "<div class='info'>Không có node.</div>"

    html = "<div class='card-area'>"

    for node in nodes:
        html += state_card_html(
            state=node.state,
            title=node.id,
            h_value=node.h,
            node_id=node.id,
            action=node.action if node.action is not None else "Start",
            color=color
        )

    html += "</div>"
    return html


def temp_cards_html(items: List[Dict[str, Any]], color: str = "#0891b2") -> str:
    """
    Hiển thị danh sách trạng thái tạm.
    Dùng cho Frontier của Simulated Annealing vì các trạng thái này chưa chắc được nhận thành node thật.
    """

    if not items:
        return "<div class='info'>Không có trạng thái.</div>"

    html = "<div class='card-area'>"

    for item in items:
        html += state_card_html(
            state=item["state"],
            title=item.get("title", item.get("id", "State")),
            h_value=item.get("h"),
            node_id=item.get("id"),
            action=item.get("action"),
            color=color
        )

    html += "</div>"
    return html


def mini_node_html(node: Optional[PuzzleNode]) -> str:
    """
    Tạo card nhỏ cho node trong bảng trace.
    """

    if node is None:
        return "-"

    parent = node.parent if node.parent is not None else "-"
    action = node.action if node.action is not None else "Start"
    path = node.path if node.path else "-"

    return f"""
    <div class='mini-card'>
        <b>{node.id}</b><br>
        Parent: {parent}<br>
        Action: {action}<br>
        Depth: {node.depth}<br>
        h: {node.h}<br>
        Path: {path}<br>
        {matrix_html(node.state, small=True)}
    </div>
    """


def mini_temp_html(item: Dict[str, Any]) -> str:
    """
    Tạo card nhỏ cho trạng thái tạm trong Frontier của Simulated Annealing.
    """

    return f"""
    <div class='mini-card'>
        <b>{item.get('id', '-')}</b><br>
        Action: {item.get('action', '-')}<br>
        h: {item.get('h', '-')}<br>
        {matrix_html(item['state'], small=True)}
    </div>
    """


def mini_node_list_html(nodes: List[PuzzleNode]) -> str:
    """
    Hiển thị nhiều node nhỏ trong một ô bảng.
    """

    if not nodes:
        return "-"

    return "<div class='mini-area'>" + "".join(mini_node_html(node) for node in nodes) + "</div>"


def mini_temp_list_html(items: List[Dict[str, Any]]) -> str:
    """
    Hiển thị nhiều trạng thái tạm nhỏ trong một ô bảng.
    """

    if not items:
        return "-"

    return "<div class='mini-area'>" + "".join(mini_temp_html(item) for item in items) + "</div>"


def reached_html(reached: List[State], goal: State, limit: int = 8) -> str:
    """
    Hiển thị danh sách Reached.

    Để bảng không quá dài, chỉ hiển thị một số trạng thái đầu tiên.
    Nếu số lượng nhiều hơn limit thì ghi thêm dòng thông báo.
    """

    if not reached:
        return "-"

    html = f"<b>{len(reached)} trạng thái</b><br>"
    html += "<div class='mini-area'>"

    for index, state in enumerate(reached[:limit]):
        html += f"""
        <div class='mini-card'>
            <b>R{index}</b><br>
            h = {misplaced_tiles(state, goal)}
            {matrix_html(state, small=True)}
        </div>
        """

    html += "</div>"

    if len(reached) > limit:
        html += f"<div style='color:#64748b;'>... còn {len(reached) - limit} trạng thái nữa</div>"

    return html


def node_table_html(nodes: List[PuzzleNode]) -> str:
    """
    Tạo bảng tổng hợp toàn bộ node đã tạo.
    """

    html = """
    <table class='trace-table'>
        <tr>
            <th>Node</th>
            <th>Parent</th>
            <th>Action</th>
            <th>Depth</th>
            <th>h</th>
            <th>Path</th>
            <th>State</th>
        </tr>
    """

    for node in nodes:
        parent = node.parent if node.parent is not None else "-"
        action = node.action if node.action is not None else "Start"
        path = node.path if node.path else "-"

        html += f"""
        <tr>
            <td><b>{node.id}</b></td>
            <td>{parent}</td>
            <td>{action}</td>
            <td>{node.depth}</td>
            <td><b>{node.h}</b></td>
            <td>{path}</td>
            <td>{matrix_html(node.state, small=True)}</td>
        </tr>
        """

    html += "</table>"
    return html


# ============================================================
# 6. CÁC BẢNG TRACE CHI TIẾT
# ============================================================

def beam_trace_table_html(trace: List[Dict[str, Any]], goal: State) -> str:
    """
    Tạo bảng chạy chi tiết cho Local Beam Search.
    """

    html = """
    <div class='box'>
        <h2>Bảng chạy Local Beam Search</h2>
        <table class='trace-table'>
            <tr>
                <th>Tầng</th>
                <th>Node đang xét</th>
                <th>Frontier trước</th>
                <th>Node con sinh ra</th>
                <th>Frontier sau khi chọn k</th>
                <th>Reached</th>
                <th>Ghi chú</th>
            </tr>
    """

    for row in trace:
        html += f"""
        <tr>
            <td><b>{row['level']}</b></td>
            <td>{mini_node_list_html(row['expanded'])}</td>
            <td>{mini_node_list_html(row['frontier_before'])}</td>
            <td>{mini_node_list_html(row['generated'])}</td>
            <td>{mini_node_list_html(row['selected'])}</td>
            <td>{reached_html(row['reached'], goal, limit=8)}</td>
            <td style='text-align:left;'>{row['note']}</td>
        </tr>
        """

    html += "</table></div>"
    return html


def annealing_trace_table_html(trace: List[Dict[str, Any]], goal: State) -> str:
    """
    Tạo bảng chạy chi tiết cho Simulated Annealing.
    """

    html = """
    <div class='box'>
        <h2>Bảng chạy Simulated Annealing</h2>
        <table class='trace-table'>
            <tr>
                <th>Bước</th>
                <th>T</th>
                <th>Node hiện tại</th>
                <th>Frontier</th>
                <th>Next được chọn</th>
                <th>Delta</th>
                <th>p</th>
                <th>r</th>
                <th>Kết quả</th>
                <th>Node sau bước</th>
                <th>Reached</th>
            </tr>
    """

    for row in trace:
        random_text = "-" if row["random_value"] is None else f"{row['random_value']:.4f}"
        accepted_text = "Nhận" if row["accepted"] else "Không nhận"

        html += f"""
        <tr>
            <td><b>{row['step']}</b></td>
            <td>{row['temperature']:.4f}</td>
            <td>{mini_node_html(row['current_before'])}</td>
            <td>{mini_temp_list_html(row['frontier'])}</td>
            <td>{mini_temp_list_html([row['chosen']])}</td>
            <td><b>{row['delta']}</b></td>
            <td>{row['probability']:.4f}</td>
            <td>{random_text}</td>
            <td><b>{accepted_text}</b><br>{row['note']}</td>
            <td>{mini_node_html(row['current_after'])}</td>
            <td>{reached_html(row['reached'], goal, limit=8)}</td>
        </tr>
        """

    html += "</table></div>"
    return html


# ============================================================
# 7. HIỂN THỊ TỪNG BƯỚC MÔ PHỎNG
# ============================================================

def render_beam_step(row: Dict[str, Any], goal: State):
    """
    Hiển thị một tầng của Local Beam Search.
    """

    html = f"""
    <div class='box'>
        <h2>Local Beam Search - Tầng {row['level']}</h2>
        <div class='info'>{row['note']}</div>

        <h3>1. Node đang xét</h3>
        {node_cards_html(row['expanded'], color='#2563eb')}

        <h3>2. Frontier trước khi mở rộng</h3>
        {node_cards_html(row['frontier_before'], color='#0891b2')}

        <h3>3. Các node con sinh ra</h3>
        {node_cards_html(row['generated'], color='#f97316')}

        <h3>4. Frontier mới sau khi chọn k node tốt nhất</h3>
        {node_cards_html(row['selected'], color='#22c55e')}

        <h3>5. Reached</h3>
        {reached_html(row['reached'], goal, limit=18)}
    </div>
    """

    display(HTML(html))


def render_annealing_step(row: Dict[str, Any], goal: State):
    """
    Hiển thị một bước của Simulated Annealing.
    """

    random_text = "-" if row["random_value"] is None else f"{row['random_value']:.4f}"
    result_text = "NHẬN" if row["accepted"] else "KHÔNG NHẬN"

    html = f"""
    <div class='box'>
        <h2>Simulated Annealing - Bước {row['step']}</h2>

        <div class='info'>
            T = <b>{row['temperature']:.4f}</b><br>
            Delta = h(next) - h(current) = <b>{row['delta']}</b><br>
            p = <b>{row['probability']:.4f}</b>, r = <b>{random_text}</b><br>
            Kết quả: <b>{result_text}</b><br>
            {row['note']}
        </div>

        <h3>1. Node hiện tại</h3>
        {node_cards_html([row['current_before']], color='#2563eb')}

        <h3>2. Frontier: các hàng xóm có thể chọn</h3>
        {temp_cards_html(row['frontier'], color='#0891b2')}

        <h3>3. Next được chọn ngẫu nhiên</h3>
        {temp_cards_html([row['chosen']], color='#f97316')}

        <h3>4. Node hiện tại sau bước này</h3>
        {node_cards_html([row['current_after']], color='#22c55e' if row['accepted'] else '#64748b')}

        <h3>5. Reached</h3>
        {reached_html(row['reached'], goal, limit=18)}
    </div>
    """

    display(HTML(html))


def solution_path_html(path_nodes: List[PuzzleNode], title: str) -> str:
    """
    Hiển thị đường đi lời giải bằng các card ma trận.
    """

    html = f"<div class='box'><h2>{title}</h2><div class='card-area'>"

    for index, node in enumerate(path_nodes):
        title_text = "Start" if index == 0 else f"Bước {index}: {node.action}"
        html += state_card_html(
            state=node.state,
            title=title_text,
            h_value=node.h,
            node_id=node.id,
            action=node.action if node.action is not None else "Start",
            color="#22c55e" if node.h == 0 else "#2563eb"
        )

    html += "</div></div>"
    return html


def start_goal_preview_html(start: State, goal: State) -> str:
    """
    Hiển thị Start và Goal trước phần kết quả.
    """

    html = "<div class='box'><h2>Start và Goal</h2><div class='card-area'>"
    html += state_card_html(start, "Start", h_value=misplaced_tiles(start, goal), color="#2563eb")
    html += state_card_html(goal, "Goal", h_value=misplaced_tiles(goal, goal), color="#22c55e")
    html += "</div></div>"
    return html


# ============================================================
# 8. TẠO GIAO DIỆN WIDGET
# ============================================================

# Nạp giao diện CSS ngay khi chạy cell.
inject_css()

# Tiêu đề chính của ứng dụng.
display(HTML("""
<div class='main-title'>
    8-Puzzle App - Local Beam Search & Simulated Annealing
</div>
<div class='info'>
    Giao diện mô phỏng có đầy đủ Node, Frontier, Reached, bảng chạy chi tiết và ma trận 3x3.
</div>
"""))

# Ô nhập trạng thái Start.
start_input = widgets.Textarea(
    value="1 2 3\n5 0 6\n4 7 8",
    description="Start",
    layout=widgets.Layout(width="360px", height="110px")
)

# Ô nhập trạng thái Goal.
goal_input = widgets.Textarea(
    value="1 2 3\n4 5 6\n7 8 0",
    description="Goal",
    layout=widgets.Layout(width="360px", height="110px")
)

# Thanh chọn k cho Beam Search.
beam_width_slider = widgets.IntSlider(
    value=2,
    min=1,
    max=5,
    step=1,
    description="Beam k"
)

# Thanh chọn số tầng tối đa cho Beam Search.
beam_level_slider = widgets.IntSlider(
    value=30,
    min=1,
    max=100,
    step=1,
    description="Max level"
)

# Nhiệt độ ban đầu cho Simulated Annealing.
temperature_input = widgets.FloatText(
    value=5.0,
    description="T0"
)

# Hệ số làm nguội alpha.
cooling_slider = widgets.FloatSlider(
    value=0.8,
    min=0.1,
    max=0.99,
    step=0.01,
    description="alpha"
)

# Nhiệt độ nhỏ nhất để dừng thuật toán.
min_temperature_input = widgets.FloatText(
    value=0.01,
    description="Tmin"
)

# Số bước tối đa cho Simulated Annealing.
annealing_step_slider = widgets.IntSlider(
    value=100,
    min=1,
    max=300,
    step=1,
    description="Max steps"
)

# Seed để kết quả random có thể lặp lại.
seed_input = widgets.IntText(
    value=0,
    description="Seed"
)

# Các nút chạy thuật toán.
run_beam_button = widgets.Button(description="Chạy Beam Search", button_style="info")
run_annealing_button = widgets.Button(description="Chạy luyện kim", button_style="success")
run_both_button = widgets.Button(description="Chạy cả hai", button_style="warning")

# Các nút điều khiển xem từng bước.
first_button = widgets.Button(description="Về đầu")
previous_button = widgets.Button(description="← Bước trước")
next_button = widgets.Button(description="Bước sau →")
last_button = widgets.Button(description="Tới cuối")

# Dòng trạng thái hiện tại.
status_label = widgets.HTML("<b style='color:#0f172a;'>Trạng thái:</b> Chưa chạy thuật toán.")

# Output tổng quát và output từng bước.
summary_output = widgets.Output()
step_output = widgets.Output()

# Biến APP_STATE lưu trạng thái hiện tại của giao diện.
APP_STATE: Dict[str, Any] = {
    "mode": None,
    "goal_node": None,
    "nodes": [],
    "trace": [],
    "goal": DEFAULT_GOAL,
    "index": 0
}


def read_input_states() -> Tuple[State, State]:
    """
    Đọc Start và Goal từ giao diện.
    Nếu nhập sai, parse_state sẽ báo lỗi.
    """

    start = parse_state(start_input.value)
    goal = parse_state(goal_input.value)
    return start, goal


def update_app_state(mode: str, goal_node: Optional[PuzzleNode], nodes: List[PuzzleNode], trace: List[Dict[str, Any]], goal: State):
    """
    Cập nhật dữ liệu thuật toán đang được xem trong phần mô phỏng từng bước.
    """

    APP_STATE["mode"] = mode
    APP_STATE["goal_node"] = goal_node
    APP_STATE["nodes"] = nodes
    APP_STATE["trace"] = trace
    APP_STATE["goal"] = goal
    APP_STATE["index"] = 0
    draw_current_step()


def draw_current_step():
    """
    Vẽ bước hiện tại trong phần mô phỏng.
    """

    with step_output:
        clear_output()
        inject_css()

        trace = APP_STATE["trace"]

        if not trace:
            display(HTML("<div class='bad'>Chưa có bước chạy nào.</div>"))
            return

        index = APP_STATE["index"]
        total = len(trace)
        row = trace[index]

        status_label.value = f"<b style='color:#0f172a;'>Đang xem:</b> {APP_STATE['mode']} - bước {index + 1}/{total}"

        if row["algo"] == "beam":
            render_beam_step(row, APP_STATE["goal"])
        else:
            render_annealing_step(row, APP_STATE["goal"])


def show_algorithm_summary(mode: str, goal_node: Optional[PuzzleNode], nodes: List[PuzzleNode], trace: List[Dict[str, Any]], start: State, goal: State):
    """
    Hiển thị kết quả tổng quát của một thuật toán.
    """

    with summary_output:
        clear_output()
        inject_css()

        display(HTML(start_goal_preview_html(start, goal)))
        display(HTML(f"<div class='box'><h2>Kết quả {mode}</h2></div>"))

        if goal_node is not None:
            path_nodes = build_solution_path(nodes, goal_node.id)
            path_text = " → ".join(goal_node.path) if goal_node.path else "Start"

            display(HTML(f"""
            <div class='ok'>
                <b>Tìm thấy Goal!</b><br>
                Node Goal: <b>{goal_node.id}</b><br>
                Đường đi: <b>{path_text}</b><br>
                Số bước: <b>{len(path_nodes) - 1}</b>
            </div>
            """))

            display(HTML(solution_path_html(path_nodes, f"Đường đi lời giải - {mode}")))
        else:
            display(HTML("""
            <div class='bad'>
                Chưa tìm thấy Goal trong giới hạn đã đặt.
                Có thể tăng Max level / Max steps hoặc đổi Seed.
            </div>
            """))

        display(HTML("<div class='box'><h2>Bảng Node</h2>"))
        display(HTML(node_table_html(nodes)))
        display(HTML("</div>"))

        if mode == "Local Beam Search":
            display(HTML(beam_trace_table_html(trace, goal)))
        else:
            display(HTML(annealing_trace_table_html(trace, goal)))

        display(HTML("""
        <div class='info'>
            Dùng các nút <b>Về đầu</b>, <b>Bước trước</b>, <b>Bước sau</b>, <b>Tới cuối</b>
            để xem mô phỏng từng bước rõ hơn.
        </div>
        """))


def on_run_beam(_):
    """
    Sự kiện khi bấm nút Chạy Beam Search.
    """

    try:
        start, goal = read_input_states()

        goal_node, nodes, trace = local_beam_search(
            start=start,
            goal=goal,
            beam_width=beam_width_slider.value,
            max_level=beam_level_slider.value
        )

        update_app_state("Local Beam Search", goal_node, nodes, trace, goal)
        show_algorithm_summary("Local Beam Search", goal_node, nodes, trace, start, goal)

    except Exception as error:
        with summary_output:
            clear_output()
            display(HTML(f"<div class='bad'><b>Lỗi:</b> {error}</div>"))


def on_run_annealing(_):
    """
    Sự kiện khi bấm nút Chạy luyện kim.
    """

    try:
        start, goal = read_input_states()

        goal_node, nodes, trace = simulated_annealing(
            start=start,
            goal=goal,
            initial_temperature=temperature_input.value,
            cooling_rate=cooling_slider.value,
            min_temperature=min_temperature_input.value,
            max_steps=annealing_step_slider.value,
            random_seed=seed_input.value
        )

        update_app_state("Simulated Annealing", goal_node, nodes, trace, goal)
        show_algorithm_summary("Simulated Annealing", goal_node, nodes, trace, start, goal)

    except Exception as error:
        with summary_output:
            clear_output()
            display(HTML(f"<div class='bad'><b>Lỗi:</b> {error}</div>"))


def on_run_both(_):
    """
    Sự kiện khi bấm nút Chạy cả hai.
    Hàm này chạy Beam Search và Simulated Annealing rồi tạo bảng so sánh.
    """

    try:
        start, goal = read_input_states()

        beam_goal, beam_nodes, beam_trace = local_beam_search(
            start=start,
            goal=goal,
            beam_width=beam_width_slider.value,
            max_level=beam_level_slider.value
        )

        annealing_goal, annealing_nodes, annealing_trace = simulated_annealing(
            start=start,
            goal=goal,
            initial_temperature=temperature_input.value,
            cooling_rate=cooling_slider.value,
            min_temperature=min_temperature_input.value,
            max_steps=annealing_step_slider.value,
            random_seed=seed_input.value
        )

        with summary_output:
            clear_output()
            inject_css()

            display(HTML(start_goal_preview_html(start, goal)))
            display(HTML("<div class='box'><h2>So sánh 2 thuật toán</h2>"))

            beam_path = " → ".join(beam_goal.path) if beam_goal is not None and beam_goal.path else "-"
            annealing_path = " → ".join(annealing_goal.path) if annealing_goal is not None and annealing_goal.path else "-"

            compare_html = f"""
            <table class='trace-table'>
                <tr>
                    <th>Thuật toán</th>
                    <th>Tìm thấy Goal</th>
                    <th>Đường đi</th>
                    <th>Số Node</th>
                    <th>Số bước trace</th>
                </tr>
                <tr>
                    <td><b>Local Beam Search</b></td>
                    <td>{'Có' if beam_goal is not None else 'Không'}</td>
                    <td>{beam_path}</td>
                    <td>{len(beam_nodes)}</td>
                    <td>{len(beam_trace)}</td>
                </tr>
                <tr>
                    <td><b>Simulated Annealing</b></td>
                    <td>{'Có' if annealing_goal is not None else 'Không'}</td>
                    <td>{annealing_path}</td>
                    <td>{len(annealing_nodes)}</td>
                    <td>{len(annealing_trace)}</td>
                </tr>
            </table>
            """

            display(HTML(compare_html))
            display(HTML("</div>"))

            if beam_goal is not None:
                display(HTML(solution_path_html(
                    build_solution_path(beam_nodes, beam_goal.id),
                    "Đường đi Local Beam Search"
                )))

            if annealing_goal is not None:
                display(HTML(solution_path_html(
                    build_solution_path(annealing_nodes, annealing_goal.id),
                    "Đường đi Simulated Annealing"
                )))

            display(HTML("<div class='box'><h2>Bảng Node - Local Beam Search</h2>"))
            display(HTML(node_table_html(beam_nodes)))
            display(HTML("</div>"))
            display(HTML(beam_trace_table_html(beam_trace, goal)))

            display(HTML("<div class='box'><h2>Bảng Node - Simulated Annealing</h2>"))
            display(HTML(node_table_html(annealing_nodes)))
            display(HTML("</div>"))
            display(HTML(annealing_trace_table_html(annealing_trace, goal)))

        # Sau khi chạy cả hai, phần mô phỏng từng bước mặc định hiển thị Beam Search.
        update_app_state("Local Beam Search", beam_goal, beam_nodes, beam_trace, goal)

    except Exception as error:
        with summary_output:
            clear_output()
            display(HTML(f"<div class='bad'><b>Lỗi:</b> {error}</div>"))


def on_previous(_):
    """
    Chuyển về bước trước trong phần mô phỏng.
    """

    if APP_STATE["trace"]:
        APP_STATE["index"] = max(0, APP_STATE["index"] - 1)
        draw_current_step()


def on_next(_):
    """
    Chuyển sang bước sau trong phần mô phỏng.
    """

    if APP_STATE["trace"]:
        APP_STATE["index"] = min(len(APP_STATE["trace"]) - 1, APP_STATE["index"] + 1)
        draw_current_step()


def on_first(_):
    """
    Quay về bước đầu tiên.
    """

    if APP_STATE["trace"]:
        APP_STATE["index"] = 0
        draw_current_step()


def on_last(_):
    """
    Nhảy đến bước cuối cùng.
    """

    if APP_STATE["trace"]:
        APP_STATE["index"] = len(APP_STATE["trace"]) - 1
        draw_current_step()


# Gắn sự kiện cho các nút chạy thuật toán.
run_beam_button.on_click(on_run_beam)
run_annealing_button.on_click(on_run_annealing)
run_both_button.on_click(on_run_both)

# Gắn sự kiện cho các nút xem từng bước.
previous_button.on_click(on_previous)
next_button.on_click(on_next)
first_button.on_click(on_first)
last_button.on_click(on_last)

# Bố cục giao diện chính.
app_layout = widgets.VBox([
    widgets.HTML("<div class='box'><h2>Nhập trạng thái 8-puzzle</h2></div>"),
    widgets.HBox([start_input, goal_input]),

    widgets.HTML("<div class='box'><h2>Cấu hình Local Beam Search</h2></div>"),
    widgets.HBox([beam_width_slider, beam_level_slider]),

    widgets.HTML("<div class='box'><h2>Cấu hình Simulated Annealing</h2></div>"),
    widgets.HBox([temperature_input, cooling_slider, min_temperature_input, seed_input]),
    annealing_step_slider,

    widgets.HTML("<div class='box'><h2>Chạy thuật toán</h2></div>"),
    widgets.HBox([run_beam_button, run_annealing_button, run_both_button]),

    widgets.HTML("<div class='box'><h2>Điều khiển xem từng bước</h2></div>"),
    widgets.HBox([first_button, previous_button, next_button, last_button]),
    status_label,

    widgets.HTML("<hr><h2 style='color:#0f172a;'>Kết quả tổng quát</h2>"),
    summary_output,

    widgets.HTML("<hr><h2 style='color:#0f172a;'>Mô phỏng từng bước: Node - Frontier - Reached</h2>"),
    step_output
])

# Hiển thị app ra notebook.
display(app_layout)
