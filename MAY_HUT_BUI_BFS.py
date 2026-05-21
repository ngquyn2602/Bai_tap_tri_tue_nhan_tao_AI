

#https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI
# ============================================================
# MÁY HÚT BỤI BFS - CODE MỚI
# Có PEAS + BFS 2 cách + Animation + Bảng Frontier/Reached
# Chạy trong Jupyter Notebook hoặc Google Colab
# ============================================================

from collections import deque
import random
import time
import threading
import html

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output


# ============================================================
# CẤU HÌNH
# ============================================================

BFS_MAX_DEPTH = 20
BFS_MAX_TRACE_ROWS = 20
BFS_AUTO_SPEED = 0.55

CELL_CLEAN = 0
CELL_DIRTY = 1
ROBOT_CLEAN = 2
ROBOT_DIRTY = 3


DEFAULT_MATRIX_TEXT = """2 1 0
0 1 0
0 0 1"""


# ============================================================
# PEAS
# ============================================================

PEAS_HTML = """
<h2 style="color:#1f4e79;">PEAS cho bài toán Máy hút bụi BFS</h2>

<table style="border-collapse:collapse; width:100%; font-size:15px;">
    <tr style="background:#e9f2fb;">
        <th style="border:1px solid #777; padding:8px;">Thành phần</th>
        <th style="border:1px solid #777; padding:8px;">Ý nghĩa</th>
    </tr>

    <tr>
        <td style="border:1px solid #777; padding:8px;"><b>P - Performance Measure</b></td>
        <td style="border:1px solid #777; padding:8px;">
            Robot hút sạch tất cả ô bẩn, số bước hợp lý, tránh lặp trạng thái.
        </td>
    </tr>

    <tr>
        <td style="border:1px solid #777; padding:8px;"><b>E - Environment</b></td>
        <td style="border:1px solid #777; padding:8px;">
            Môi trường là ma trận gồm ô sạch, ô bẩn và vị trí robot.
        </td>
    </tr>

    <tr>
        <td style="border:1px solid #777; padding:8px;"><b>A - Actuators</b></td>
        <td style="border:1px solid #777; padding:8px;">
            Robot có thể hút bụi tại ô hiện tại hoặc di chuyển lên, xuống, trái, phải.
        </td>
    </tr>

    <tr>
        <td style="border:1px solid #777; padding:8px;"><b>S - Sensors</b></td>
        <td style="border:1px solid #777; padding:8px;">
            Robot biết vị trí hiện tại và biết ô nào đang bẩn trong ma trận.
        </td>
    </tr>
</table>
"""


# ============================================================
# XỬ LÝ INPUT MA TRẬN
# ============================================================

def parse_matrix(text):
    """
    Đọc ma trận người dùng nhập.

    Quy ước:
    0: ô sạch
    1: ô bẩn
    2: robot ở ô sạch
    3: robot ở ô bẩn
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        raise ValueError("Bạn chưa nhập ma trận.")

    matrix = []

    for line in lines:
        row = line.replace(",", " ").split()
        row = [int(x) for x in row]
        matrix.append(row)

    col_count = len(matrix[0])

    if col_count == 0:
        raise ValueError("Ma trận không hợp lệ.")

    for row in matrix:
        if len(row) != col_count:
            raise ValueError("Các hàng trong ma trận phải có cùng số cột.")

        for value in row:
            if value not in [0, 1, 2, 3]:
                raise ValueError("Ma trận chỉ được chứa các số 0, 1, 2, 3.")

    robot_count = 0

    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] in [2, 3]:
                robot_count += 1

    if robot_count != 1:
        raise ValueError("Ma trận phải có đúng 1 robot, ký hiệu là 2 hoặc 3.")

    return matrix


def matrix_to_state(matrix):
    """
    Chuyển ma trận sang state lõi.

    State mới không lưu cả ma trận.
    State chỉ lưu:
    - hàng robot
    - cột robot
    - dirty_tuple: tuple 0/1 cho biết ô nào còn bẩn

    Ví dụ:
    state = (robot_row, robot_col, dirty_tuple)
    """
    rows = len(matrix)
    cols = len(matrix[0])

    robot_row = None
    robot_col = None

    dirty = []

    for r in range(rows):
        for c in range(cols):
            value = matrix[r][c]

            if value in [1, 3]:
                dirty.append(1)
            else:
                dirty.append(0)

            if value in [2, 3]:
                robot_row = r
                robot_col = c

    state = (robot_row, robot_col, tuple(dirty))
    return state, rows, cols


def state_to_matrix(state, rows, cols):
    """
    Chuyển state lõi về ma trận 0/1/2/3 để hiển thị.
    """
    robot_row, robot_col, dirty = state

    matrix = []

    for r in range(rows):
        row = []

        for c in range(cols):
            idx = r * cols + c
            is_dirty = dirty[idx] == 1
            is_robot = (r == robot_row and c == robot_col)

            if is_robot and is_dirty:
                row.append(3)
            elif is_robot and not is_dirty:
                row.append(2)
            elif is_dirty:
                row.append(1)
            else:
                row.append(0)

        matrix.append(row)

    return matrix


def matrix_to_text(matrix):
    return "\n".join(" ".join(str(x) for x in row) for row in matrix)


def dirty_count(state):
    return sum(state[2])


def is_goal(state):
    """
    Goal là khi không còn ô bẩn nào.
    """
    return dirty_count(state) == 0


# ============================================================
# HÀM SINH TRẠNG THÁI KẾ TIẾP
# ============================================================

def get_successors(state, rows, cols):
    """
    Sinh các trạng thái kế tiếp.

    Các hành động:
    - Suck
    - Up
    - Down
    - Left
    - Right
    """
    robot_row, robot_col, dirty = state
    successors = []

    current_idx = robot_row * cols + robot_col

    # Nếu ô hiện tại bẩn thì có thể hút bụi
    if dirty[current_idx] == 1:
        new_dirty = list(dirty)
        new_dirty[current_idx] = 0

        new_state = (robot_row, robot_col, tuple(new_dirty))
        successors.append(("Suck", new_state))

    moves = [
        ("Up", -1, 0),
        ("Down", 1, 0),
        ("Left", 0, -1),
        ("Right", 0, 1)
    ]

    for move_name, dr, dc in moves:
        new_row = robot_row + dr
        new_col = robot_col + dc

        if 0 <= new_row < rows and 0 <= new_col < cols:
            new_state = (new_row, new_col, dirty)
            successors.append((move_name, new_state))

    return successors


# ============================================================
# FORMAT STATE / FRONTIER / REACHED
# ============================================================

def dirty_map_text(state, rows, cols):
    """
    Hiển thị bản đồ ô bẩn dạng ngắn.
    Ví dụ:
    010/010/001
    """
    dirty = state[2]
    parts = []

    for r in range(rows):
        start = r * cols
        end = start + cols
        parts.append("".join(str(x) for x in dirty[start:end]))

    return "/".join(parts)


def state_compact(state, rows, cols):
    robot_row, robot_col, dirty = state
    return f"R({robot_row},{robot_col}) D:{dirty_map_text(state, rows, cols)}"


def action_path_text(actions):
    if not actions:
        return "Start"
    return " → ".join(actions)


def format_queue(nodes, rows, cols, limit=8):
    """
    BFS dùng Queue:
    - FRONT nằm bên trái
    - BACK nằm bên phải
    - Pop ở FRONT
    - Thêm node mới vào BACK
    """
    if not nodes:
        return "Rỗng"

    shown = nodes[:limit]
    lines = []

    for i, node in enumerate(shown):
        text = state_compact(node["state"], rows, cols)

        if len(nodes) == 1:
            text += " ← FRONT/BACK"
        elif i == 0:
            text += " ← FRONT bên trái"
        elif i == len(shown) - 1 and len(nodes) <= limit:
            text += " ← BACK bên phải"

        lines.append(text)

    if len(nodes) > limit:
        lines.append(f"... ({len(nodes) - limit} state khác)")
        lines.append("BACK nằm bên phải Queue")

    return " |\n".join(lines)


def format_state_set(states, rows, cols, limit=8):
    if states == "Không dùng":
        return "Không dùng"

    states = list(states)

    if not states:
        return "Rỗng"

    shown = states[:limit]
    lines = [state_compact(s, rows, cols) for s in shown]

    if len(states) > limit:
        lines.append(f"... ({len(states) - limit} state khác)")

    return " |\n".join(lines)


def cell_style(value):
    if value == 0:
        return "background:#f8f9fa; color:#555;"
    if value == 1:
        return "background:#ffd6d6; color:#8a0000; font-weight:bold;"
    if value == 2:
        return "background:#d7ecff; color:#004b8d; font-weight:bold;"
    return "background:#ffe7a8; color:#7a4b00; font-weight:bold;"


def state_grid_html(state, rows, cols):
    matrix = state_to_matrix(state, rows, cols)

    html_text = """
    <table style="border-collapse:collapse; font-family:monospace;">
    """

    for r in range(rows):
        html_text += "<tr>"

        for c in range(cols):
            value = matrix[r][c]

            label = str(value)

            if value == 0:
                title = "Clean"
            elif value == 1:
                title = "Dirty"
            elif value == 2:
                title = "Robot"
            else:
                title = "Robot + Dirty"

            html_text += f"""
            <td title="{title}" style="
                border:1px solid #777;
                width:36px;
                height:34px;
                text-align:center;
                vertical-align:middle;
                {cell_style(value)}
            ">
                {label}
            </td>
            """

        html_text += "</tr>"

    html_text += "</table>"
    return html_text


# ============================================================
# BẢNG HTML ĐẸP
# ============================================================

def make_html_table(rows, title):
    if not rows:
        return f"<p>Chưa có dữ liệu cho {html.escape(title)}.</p>"

    columns = list(rows[0].keys())

    width_map = {
        "Step": "60px",
        "Move": "90px",
        "Node": "170px",
        "State": "150px",
        "State Grid": "150px",
        "State Compact": "170px",
        "Robot Position": "120px",
        "Dirty Left": "90px",
        "Depth": "70px",
        "Action Path": "230px",
        "Children Added": "260px",
        "Frontier Before Pop": "280px",
        "Frontier After Expand": "280px",
        "Reached": "280px",
        "Path Checking": "280px",
        "Note": "170px"
    }

    css = """
    <style>
        .table-box {
            overflow: auto;
            max-height: 560px;
            max-width: 100%;
            border: 1px solid #999;
            margin-top: 10px;
            margin-bottom: 22px;
            background: white;
        }

        table.search-table {
            border-collapse: collapse;
            table-layout: fixed;
            min-width: 1450px;
            width: max-content;
            font-size: 14px;
        }

        table.search-table th,
        table.search-table td {
            border: 1px solid #777;
            padding: 8px 10px;
            vertical-align: top;
            text-align: left;
            line-height: 1.35;
        }

        table.search-table th {
            background: #eeeeee;
            position: sticky;
            top: 0;
            z-index: 2;
        }

        table.search-table tr:nth-child(even) {
            background: #fafafa;
        }

        .long-cell {
            white-space: pre-line;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
    </style>
    """

    html_text = css
    html_text += f"<h3>{html.escape(title)}</h3>"
    html_text += '<div class="table-box">'
    html_text += '<table class="search-table">'

    html_text += "<colgroup>"
    for col in columns:
        html_text += f'<col style="width:{width_map.get(col, "180px")};">'
    html_text += "</colgroup>"

    html_text += "<tr>"
    for col in columns:
        html_text += f"<th>{html.escape(str(col))}</th>"
    html_text += "</tr>"

    for row in rows:
        html_text += "<tr>"

        for col in columns:
            value = row[col]

            if col in ["State", "State Grid"]:
                cell = value
            else:
                cell = html.escape(str(value))

            html_text += f'<td class="long-cell">{cell}</td>'

        html_text += "</tr>"

    html_text += "</table></div>"

    return html_text


# ============================================================
# BFS CÁCH 1: QUEUE + REACHED TOÀN CỤC
# ============================================================

def bfs_queue_reached(start_state, rows, cols):
    queue = deque()

    first_node = {
        "id": 1,
        "state": start_state,
        "actions": [],
        "states": [start_state],
        "depth": 0,
        "path_states": [start_state]
    }

    queue.append(first_node)

    reached = {start_state}
    trace = []

    next_id = 2
    expanded = 0
    step = 1

    while queue:
        frontier_before = list(queue)

        current = queue.popleft()
        current_state = current["state"]
        current_depth = current["depth"]
        current_actions = current["actions"]

        expanded += 1

        if is_goal(current_state):
            if len(trace) < BFS_MAX_TRACE_ROWS:
                trace.append({
                    "Step": step,
                    "Node": state_compact(current_state, rows, cols),
                    "State": state_grid_html(current_state, rows, cols),
                    "Depth": current_depth,
                    "Action Path": action_path_text(current_actions),
                    "Children Added": "Không cần sinh con vì đã đạt Goal",
                    "Frontier Before Pop": format_queue(frontier_before, rows, cols),
                    "Frontier After Expand": format_queue(list(queue), rows, cols),
                    "Reached": format_state_set(reached, rows, cols),
                    "Note": "Tìm thấy Goal"
                })

            return {
                "found": True,
                "states": current["states"],
                "actions": current["actions"],
                "trace": trace,
                "expanded": expanded,
                "visited": len(reached),
                "message": "BFS Cách 1 đã tìm thấy lời giải."
            }

        children_text = []

        if current_depth < BFS_MAX_DEPTH:
            for action, next_state in get_successors(current_state, rows, cols):
                if next_state not in reached:
                    reached.add(next_state)

                    new_node = {
                        "id": next_id,
                        "state": next_state,
                        "actions": current["actions"] + [action],
                        "states": current["states"] + [next_state],
                        "depth": current_depth + 1,
                        "path_states": current["path_states"] + [next_state]
                    }

                    next_id += 1
                    queue.append(new_node)

                    children_text.append(
                        f"{action}: {state_compact(next_state, rows, cols)}"
                    )
                else:
                    children_text.append(
                        f"{action}: bỏ qua vì đã có trong Reached"
                    )
        else:
            children_text.append("Không sinh con vì đạt BFS_MAX_DEPTH")

        if len(trace) < BFS_MAX_TRACE_ROWS:
            trace.append({
                "Step": step,
                "Node": state_compact(current_state, rows, cols),
                "State": state_grid_html(current_state, rows, cols),
                "Depth": current_depth,
                "Action Path": action_path_text(current_actions),
                "Children Added": " |\n".join(children_text),
                "Frontier Before Pop": format_queue(frontier_before, rows, cols),
                "Frontier After Expand": format_queue(list(queue), rows, cols),
                "Reached": format_state_set(reached, rows, cols),
                "Note": "Đã mở rộng node theo BFS"
            })

        step += 1

    return {
        "found": False,
        "states": [],
        "actions": [],
        "trace": trace,
        "expanded": expanded,
        "visited": len(reached),
        "message": "Không tìm thấy lời giải trong giới hạn hiện tại."
    }


# ============================================================
# BFS CÁCH 2: QUEUE + PATH CHECKING
# ============================================================

def bfs_queue_path_checking(start_state, rows, cols):
    queue = deque()

    first_node = {
        "id": 1,
        "state": start_state,
        "actions": [],
        "states": [start_state],
        "depth": 0,
        "path_states": [start_state]
    }

    queue.append(first_node)

    trace = []
    next_id = 2
    expanded = 0
    generated = 1
    step = 1

    while queue:
        frontier_before = list(queue)

        current = queue.popleft()
        current_state = current["state"]
        current_depth = current["depth"]
        current_actions = current["actions"]
        current_path_states = current["path_states"]

        expanded += 1

        if is_goal(current_state):
            if len(trace) < BFS_MAX_TRACE_ROWS:
                trace.append({
                    "Step": step,
                    "Node": state_compact(current_state, rows, cols),
                    "State": state_grid_html(current_state, rows, cols),
                    "Depth": current_depth,
                    "Action Path": action_path_text(current_actions),
                    "Children Added": "Không cần sinh con vì đã đạt Goal",
                    "Frontier Before Pop": format_queue(frontier_before, rows, cols),
                    "Frontier After Expand": format_queue(list(queue), rows, cols),
                    "Reached": "Không dùng",
                    "Path Checking": format_state_set(current_path_states, rows, cols),
                    "Note": "Tìm thấy Goal"
                })

            return {
                "found": True,
                "states": current["states"],
                "actions": current["actions"],
                "trace": trace,
                "expanded": expanded,
                "visited": generated,
                "message": "BFS Cách 2 đã tìm thấy lời giải."
            }

        children_text = []

        if current_depth < BFS_MAX_DEPTH:
            for action, next_state in get_successors(current_state, rows, cols):
                if next_state not in current_path_states:
                    new_node = {
                        "id": next_id,
                        "state": next_state,
                        "actions": current["actions"] + [action],
                        "states": current["states"] + [next_state],
                        "depth": current_depth + 1,
                        "path_states": current["path_states"] + [next_state]
                    }

                    next_id += 1
                    generated += 1
                    queue.append(new_node)

                    children_text.append(
                        f"{action}: {state_compact(next_state, rows, cols)}"
                    )
                else:
                    children_text.append(
                        f"{action}: bỏ qua vì lặp trong đường đi hiện tại"
                    )
        else:
            children_text.append("Không sinh con vì đạt BFS_MAX_DEPTH")

        if len(trace) < BFS_MAX_TRACE_ROWS:
            trace.append({
                "Step": step,
                "Node": state_compact(current_state, rows, cols),
                "State": state_grid_html(current_state, rows, cols),
                "Depth": current_depth,
                "Action Path": action_path_text(current_actions),
                "Children Added": " |\n".join(children_text),
                "Frontier Before Pop": format_queue(frontier_before, rows, cols),
                "Frontier After Expand": format_queue(list(queue), rows, cols),
                "Reached": "Không dùng",
                "Path Checking": format_state_set(current_path_states, rows, cols),
                "Note": "Chỉ kiểm tra lặp trong path hiện tại"
            })

        step += 1

    return {
        "found": False,
        "states": [],
        "actions": [],
        "trace": trace,
        "expanded": expanded,
        "visited": generated,
        "message": "Không tìm thấy lời giải trong giới hạn hiện tại."
    }


# ============================================================
# TẠO MA TRẬN RANDOM
# ============================================================

def make_random_matrix(rows=3, cols=3):
    total_cells = rows * cols

    robot_idx = random.randrange(total_cells)

    dirty_number = random.randint(2, min(4, total_cells))
    dirty_positions = set(random.sample(range(total_cells), dirty_number))

    matrix = []

    for r in range(rows):
        row = []

        for c in range(cols):
            idx = r * cols + c

            if idx == robot_idx and idx in dirty_positions:
                row.append(3)
            elif idx == robot_idx:
                row.append(2)
            elif idx in dirty_positions:
                row.append(1)
            else:
                row.append(0)

        matrix.append(row)

    return matrix


# ============================================================
# APP GIAO DIỆN
# ============================================================

class VacuumBFSApp:
    def __init__(self):
        self.rows = 3
        self.cols = 3

        self.start_state, self.rows, self.cols = matrix_to_state(
            parse_matrix(DEFAULT_MATRIX_TEXT)
        )

        self.current_state = self.start_state

        self.solution_states = []
        self.solution_actions = []
        self.solution_rows = []
        self.trace_rows = []

        self.auto_running = False
        self.run_id = 0
        self.algorithm_name = "Chưa chạy"

        self.matrix_input = widgets.Textarea(
            value=DEFAULT_MATRIX_TEXT,
            description="Matrix:",
            layout=widgets.Layout(width="420px", height="120px")
        )

        self.algorithm_select = widgets.Dropdown(
            options=[
                ("BFS Cách 1 - Queue + Reached", "reached"),
                ("BFS Cách 2 - Queue + Path Checking", "path")
            ],
            value="reached",
            description="Thuật toán:",
            layout=widgets.Layout(width="420px")
        )

        self.load_button = widgets.Button(
            description="Load Matrix",
            button_style="info"
        )

        self.random_button = widgets.Button(
            description="Random Again",
            button_style="warning"
        )

        self.solve_button = widgets.Button(
            description="Solve BFS",
            button_style="success"
        )

        self.stop_button = widgets.Button(
            description="Stop",
            button_style="danger"
        )

        self.table_button = widgets.Button(
            description="Show Tables",
            button_style="primary"
        )

        self.reset_button = widgets.Button(
            description="Reset"
        )

        self.status = widgets.HTML(value="")
        self.board_output = widgets.Output()
        self.solution_output = widgets.Output()
        self.trace_output = widgets.Output()

        self.load_button.on_click(self.load_matrix)
        self.random_button.on_click(self.random_again)
        self.solve_button.on_click(self.solve_bfs)
        self.stop_button.on_click(self.stop_animation)
        self.table_button.on_click(self.show_tables)
        self.reset_button.on_click(self.reset)

        self.controls = widgets.HBox([
            self.load_button,
            self.random_button,
            self.solve_button,
            self.stop_button,
            self.table_button,
            self.reset_button
        ])

        self.app = widgets.VBox([
            widgets.HTML(PEAS_HTML),
            widgets.HTML("""
                <h2 style="color:#1f4e79;">Máy hút bụi thông minh - BFS</h2>
                <p>
                    Quy ước ma trận:
                    <b>0</b> = sạch,
                    <b>1</b> = bẩn,
                    <b>2</b> = robot ở ô sạch,
                    <b>3</b> = robot ở ô bẩn.
                </p>
                <p>
                    BFS dùng <b>Queue</b>: FRONT nằm bên trái, BACK nằm bên phải.
                </p>
            """),
            self.matrix_input,
            self.algorithm_select,
            self.controls,
            self.status,
            self.board_output,
            self.solution_output,
            self.trace_output
        ])

        self.render_board(
            self.current_state,
            "Trạng thái ban đầu",
            "Sẵn sàng chạy BFS."
        )

    def render_board(self, state, title="", note=""):
        with self.board_output:
            clear_output(wait=True)

            legend = """
            <p>
                <b>0</b>: sạch |
                <b>1</b>: bẩn |
                <b>2</b>: robot ở ô sạch |
                <b>3</b>: robot ở ô bẩn
            </p>
            """

            html_text = f"""
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(note)}</p>
            {state_grid_html(state, self.rows, self.cols)}
            {legend}
            """

            display(HTML(html_text))

    def clear_tables(self):
        with self.solution_output:
            clear_output()

        with self.trace_output:
            clear_output()

    def load_matrix(self, button=None):
        self.stop_animation()

        try:
            matrix = parse_matrix(self.matrix_input.value)
            state, rows, cols = matrix_to_state(matrix)

            self.rows = rows
            self.cols = cols
            self.start_state = state
            self.current_state = state

            self.solution_states = []
            self.solution_actions = []
            self.solution_rows = []
            self.trace_rows = []

            self.clear_tables()

            self.render_board(
                self.current_state,
                "Đã load ma trận",
                f"Số ô bẩn hiện tại: {dirty_count(self.current_state)}"
            )

            self.status.value = f"""
            <b>Đã load thành công.</b><br>
            Kích thước ma trận: {self.rows} x {self.cols}<br>
            Số ô bẩn: {dirty_count(self.current_state)}
            """

        except Exception as e:
            self.status.value = f"<b style='color:#b00020;'>Lỗi:</b> {html.escape(str(e))}"

    def random_again(self, button=None):
        self.stop_animation()

        try:
            matrix = make_random_matrix(3, 3)
            self.matrix_input.value = matrix_to_text(matrix)

            state, rows, cols = matrix_to_state(matrix)

            self.rows = rows
            self.cols = cols
            self.start_state = state
            self.current_state = state

            self.solution_states = []
            self.solution_actions = []
            self.solution_rows = []
            self.trace_rows = []

            self.clear_tables()

            self.render_board(
                self.current_state,
                "Random Matrix",
                "Ma trận random hợp lệ, có robot và có ô bẩn."
            )

            self.status.value = f"""
            <b>Đã tạo ma trận random.</b><br>
            Kích thước: {self.rows} x {self.cols}<br>
            Số ô bẩn: {dirty_count(self.current_state)}<br>
            Bấm <b>Solve BFS</b> để giải.
            """

        except Exception as e:
            self.status.value = f"<b style='color:#b00020;'>Lỗi:</b> {html.escape(str(e))}"

    def build_solution_table(self):
        self.solution_rows = []

        for i, state in enumerate(self.solution_states):
            if i == 0:
                move = "Start"
            else:
                move = self.solution_actions[i - 1]

            robot_row, robot_col, dirty = state

            self.solution_rows.append({
                "Step": i,
                "Move": move,
                "Robot Position": f"({robot_row}, {robot_col})",
                "Dirty Left": sum(dirty),
                "State Grid": state_grid_html(state, self.rows, self.cols),
                "State Compact": state_compact(state, self.rows, self.cols)
            })

    def show_solution_table(self):
        with self.solution_output:
            clear_output(wait=True)

            if not self.solution_rows:
                display(HTML("<p>Chưa có Solution Steps. Hãy bấm <b>Solve BFS</b>.</p>"))
                return

            display(HTML(make_html_table(
                self.solution_rows,
                "Solution Steps"
            )))

    def show_trace_table(self):
        with self.trace_output:
            clear_output(wait=True)

            if not self.trace_rows:
                display(HTML("<p>Chưa có BFS Table. Hãy bấm <b>Solve BFS</b>.</p>"))
                return

            display(HTML(make_html_table(
                self.trace_rows,
                "BFS Table: Node / State / Frontier / Reached"
            )))

    def show_tables(self, button=None):
        self.show_solution_table()
        self.show_trace_table()

    def solve_bfs(self, button=None):
        self.stop_animation()

        try:
            matrix = parse_matrix(self.matrix_input.value)
            state, rows, cols = matrix_to_state(matrix)

            self.rows = rows
            self.cols = cols
            self.start_state = state
            self.current_state = state

            selected = self.algorithm_select.value

            if selected == "reached":
                self.algorithm_name = "BFS Cách 1 - Queue + Reached"
                result = bfs_queue_reached(
                    self.start_state,
                    self.rows,
                    self.cols
                )
            else:
                self.algorithm_name = "BFS Cách 2 - Queue + Path Checking"
                result = bfs_queue_path_checking(
                    self.start_state,
                    self.rows,
                    self.cols
                )

            self.solution_states = result["states"]
            self.solution_actions = result["actions"]
            self.trace_rows = result["trace"]
            self.solution_rows = []

            if result["found"]:
                self.build_solution_table()

                self.status.value = f"""
                <b>Thuật toán:</b> {self.algorithm_name}<br>
                <b>Kết quả:</b> Tìm thấy lời giải.<br>
                <b>Số bước:</b> {len(self.solution_actions)}<br>
                <b>Số node đã mở rộng:</b> {result["expanded"]}<br>
                <b>Số trạng thái đã lưu/đã sinh:</b> {result["visited"]}<br>
                <b>Đường đi:</b> {action_path_text(self.solution_actions)}<br>
                <b>Lưu ý:</b> Bảng BFS chỉ hiện {BFS_MAX_TRACE_ROWS} dòng đầu để dễ minh họa.
                """

                self.show_tables()
                self.start_animation()

            else:
                self.status.value = f"""
                <b>Thuật toán:</b> {self.algorithm_name}<br>
                <b>Kết quả:</b> Không tìm thấy lời giải trong BFS_MAX_DEPTH = {BFS_MAX_DEPTH}.<br>
                <b>Gợi ý:</b> Ma trận có thể cần nhiều bước hơn. Hãy random lại hoặc tăng BFS_MAX_DEPTH.
                """

                self.show_trace_table()

        except Exception as e:
            self.status.value = f"<b style='color:#b00020;'>Lỗi:</b> {html.escape(str(e))}"

    def start_animation(self):
        if not self.solution_states:
            return

        self.auto_running = True
        self.run_id += 1
        my_run_id = self.run_id

        thread = threading.Thread(
            target=self.animation_loop,
            args=(my_run_id,),
            daemon=True
        )

        thread.start()

    def animation_loop(self, my_run_id):
        for i, state in enumerate(self.solution_states):
            if not self.auto_running or my_run_id != self.run_id:
                return

            self.current_state = state

            if i == 0:
                move_text = "Bước 0: Start"
            else:
                move_text = f"Bước {i}: {self.solution_actions[i - 1]}"

            self.render_board(
                self.current_state,
                f"Animation BFS - Step {i}",
                move_text
            )

            self.status.value = f"""
            <b>Đang chạy animation...</b><br>
            <b>Thuật toán:</b> {self.algorithm_name}<br>
            <b>Step:</b> {i}/{len(self.solution_states) - 1}<br>
            <b>Move:</b> {move_text}<br>
            <b>Số ô bẩn còn lại:</b> {dirty_count(self.current_state)}
            """

            time.sleep(BFS_AUTO_SPEED)

        self.auto_running = False

        self.status.value = f"""
        <b>Chạy xong.</b><br>
        <b>Thuật toán:</b> {self.algorithm_name}<br>
        <b>Kết quả:</b> Robot đã hút sạch toàn bộ ô bẩn.<br>
        <b>Tổng số bước:</b> {len(self.solution_actions)}
        """

    def stop_animation(self, button=None):
        self.auto_running = False
        self.run_id += 1

        if button is not None:
            self.status.value = "<b>Đã dừng animation.</b>"

    def reset(self, button=None):
        self.stop_animation()

        self.matrix_input.value = DEFAULT_MATRIX_TEXT
        matrix = parse_matrix(DEFAULT_MATRIX_TEXT)
        state, rows, cols = matrix_to_state(matrix)

        self.rows = rows
        self.cols = cols
        self.start_state = state
        self.current_state = state

        self.solution_states = []
        self.solution_actions = []
        self.solution_rows = []
        self.trace_rows = []

        self.clear_tables()

        self.render_board(
            self.current_state,
            "Reset về mặc định",
            "Sẵn sàng chạy BFS."
        )

        self.status.value = "<b>Đã reset bài toán về mặc định.</b>"

    def show(self):
        display(self.app)


# ============================================================
# CHẠY APP
# ============================================================

app = VacuumBFSApp()
app.show()
