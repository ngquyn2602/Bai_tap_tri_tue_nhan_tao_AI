# -*- coding: utf-8 -*-
"""
BÀI TẬP TRÍ TUỆ NHÂN TẠO - 8 PUZZLE LOCAL SEARCH
GitHub tham khảo: https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI

- Chia code thành nhiều class: PuzzleTools, BoardView, LocalSearchSolver, NotebookApp.
- Giao diện Jupyter dùng bố cục tab/khối cấu hình rõ hơn.
- Vẫn giữ đúng yêu cầu bài: Leo đồi khởi tạo ngẫu nhiên và Local Beam Search.
- Có bảng Node / Frontier / Reached để dễ trình bày khi báo cáo.

"""

import random
import re
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display


# ============================================================
# 0. THÔNG TIN BÀI / HẰNG SỐ
# ============================================================

REPO_LINK = "https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI"
ACTION_ORDER = ("L", "R", "U", "D")
ACTION_VI = {
    "L": "Trái",
    "R": "Phải",
    "U": "Lên",
    "D": "Xuống",
}

State = Tuple[int, ...]
ParentMap = Dict[State, Tuple[Optional[State], Optional[str]]]


@dataclass
class SearchResult:
    """Gói kết quả trả về sau khi chạy thuật toán."""

    found: bool
    message: str
    goal_state: Optional[State]
    best_state: State
    best_h: int
    parent: ParentMap
    logs: List[dict]
    labeler: "NodeLabeler"
    expanded_count: int


# ============================================================
# 1. XỬ LÝ 8-PUZZLE
# ============================================================

class PuzzleTools:
    """Nhóm hàm xử lý trạng thái 8-puzzle."""

    @staticmethod
    def parse(text: str) -> State:
        """
        Cho phép nhập nhiều kiểu:
        - 1 2 3/4 5 6/7 8 0
        - 1 2 3\n4 5 6\n7 8 0
        - [1,2,3,4,5,6,7,8,0]
        """
        values = list(map(int, re.findall(r"-?\d+", text)))
        if len(values) != 9:
            raise ValueError("Cần nhập đúng 9 số cho ma trận 3x3.")
        if sorted(values) != list(range(9)):
            raise ValueError("Trạng thái phải chứa đủ các số từ 0 đến 8, không được trùng.")
        return tuple(values)

    @staticmethod
    def to_rows(state: State) -> List[List[int]]:
        return [list(state[i:i + 3]) for i in range(0, 9, 3)]

    @staticmethod
    def to_multiline(state: State) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in PuzzleTools.to_rows(state))

    @staticmethod
    def to_short(state: State) -> str:
        return " / ".join(" ".join(str(x) for x in state[i:i + 3]) for i in range(0, 9, 3))

    @staticmethod
    def goal_index(goal: State) -> Dict[int, Tuple[int, int]]:
        return {value: (idx // 3, idx % 3) for idx, value in enumerate(goal)}

    @staticmethod
    def h_manhattan(state: State, goal: State) -> int:
        """Tổng khoảng cách Manhattan, không tính ô 0."""
        pos = PuzzleTools.goal_index(goal)
        s = 0
        for idx, tile in enumerate(state):
            if tile == 0:
                continue
            r, c = divmod(idx, 3)
            gr, gc = pos[tile]
            s += abs(r - gr) + abs(c - gc)
        return s

    @staticmethod
    def h_misplaced(state: State, goal: State) -> int:
        """Số ô sai vị trí, không tính ô 0."""
        return sum(1 for a, b in zip(state, goal) if a != 0 and a != b)

    @staticmethod
    def choose_heuristic(name: str) -> Callable[[State, State], int]:
        if name == "misplaced":
            return PuzzleTools.h_misplaced
        return PuzzleTools.h_manhattan

    @staticmethod
    def parity_with_goal(state: State, goal: State) -> int:
        """
        Kiểm tra parity theo thứ tự tương đối của goal.
        Với 8-puzzle 3x3: solvable khi số nghịch thế là chẵn.
        """
        rank = {tile: i for i, tile in enumerate(goal) if tile != 0}
        seq = [rank[tile] for tile in state if tile != 0]
        inv = 0
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] > seq[j]:
                    inv += 1
        return inv % 2

    @staticmethod
    def can_solve(start: State, goal: State) -> bool:
        return PuzzleTools.parity_with_goal(start, goal) == 0

    @staticmethod
    def expand(state: State) -> List[Tuple[str, State]]:
        """Sinh hàng xóm theo thứ tự L, R, U, D. Action là hướng đi của ô trống 0."""
        z = state.index(0)
        r, c = divmod(z, 3)
        children: List[Tuple[str, State]] = []

        def make(nr: int, nc: int, action: str) -> None:
            nz = nr * 3 + nc
            arr = list(state)
            arr[z], arr[nz] = arr[nz], arr[z]
            children.append((action, tuple(arr)))

        if c > 0:
            make(r, c - 1, "L")
        if c < 2:
            make(r, c + 1, "R")
        if r > 0:
            make(r - 1, c, "U")
        if r < 2:
            make(r + 1, c, "D")
        return children

    @staticmethod
    def random_valid(goal: State, min_h: int = 7) -> State:
        """Tạo trạng thái ngẫu nhiên có thể giải được, tránh quá gần goal."""
        numbers = list(range(9))
        while True:
            random.shuffle(numbers)
            state = tuple(numbers)
            if PuzzleTools.can_solve(state, goal) and PuzzleTools.h_manhattan(state, goal) >= min_h:
                return state


# ============================================================
# 2. ĐẶT TÊN NODE A, B, C, ...
# ============================================================

class NodeLabeler:
    def __init__(self) -> None:
        self._map: Dict[State, str] = {}
        self._count = 0

    @staticmethod
    def _name(index: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if index < 26:
            return alphabet[index]
        return alphabet[index % 26] + str(index // 26)

    def get(self, state: State) -> str:
        if state not in self._map:
            self._map[state] = self._name(self._count)
            self._count += 1
        return self._map[state]


# ============================================================
# 3. HIỂN THỊ HTML TRONG NOTEBOOK
# ============================================================

class BoardView:
    @staticmethod
    def board(state: State, goal: Optional[State] = None, title: str = "", note: str = "") -> str:
        cells = []
        for i, value in enumerate(state):
            if value == 0:
                background = "#f8fafc"
                color = "#94a3b8"
                text = "□"
                border = "#cbd5e1"
            elif goal is not None and value == goal[i]:
                background = "#dcfce7"
                color = "#166534"
                text = str(value)
                border = "#86efac"
            else:
                background = "#e0f2fe"
                color = "#0f172a"
                text = str(value)
                border = "#bae6fd"

            cells.append(f"""
            <div style="
                width:58px;height:58px;border-radius:12px;
                display:flex;align-items:center;justify-content:center;
                background:{background};color:{color};
                border:1px solid {border};font-weight:800;font-size:24px;
                box-shadow:0 1px 3px rgba(15,23,42,.10);
            ">{text}</div>
            """)

        return f"""
        <div style="
            display:inline-block;vertical-align:top;margin:8px;padding:13px 15px;
            background:white;border:1px solid #e2e8f0;border-radius:18px;
            box-shadow:0 5px 14px rgba(15,23,42,.08);
        ">
            <div style="font-size:15px;font-weight:800;color:#0f172a;margin-bottom:3px;">{title}</div>
            <div style="font-size:12px;color:#475569;white-space:pre-wrap;margin-bottom:9px;">{note}</div>
            <div style="display:grid;grid-template-columns:repeat(3,58px);gap:7px;">
                {''.join(cells)}
            </div>
        </div>
        """

    @staticmethod
    def many(states: List[State], goal: State, titles: List[str], notes: List[str], limit: int = 24) -> None:
        html = ""
        for state, title, note in zip(states[:limit], titles[:limit], notes[:limit]):
            html += BoardView.board(state, goal, title, note)
        display(HTML(html))
        if len(states) > limit:
            display(HTML(f"<i>Đường đi có {len(states)} trạng thái, chỉ hiển thị {limit} trạng thái đầu.</i>"))

    @staticmethod
    def box(message: str, kind: str = "info") -> HTML:
        palette = {
            "info": ("#eff6ff", "#1d4ed8", "#bfdbfe"),
            "ok": ("#dcfce7", "#166534", "#86efac"),
            "warn": ("#fef9c3", "#92400e", "#fde68a"),
            "bad": ("#fef2f2", "#991b1b", "#fecaca"),
        }
        bg, color, border = palette.get(kind, palette["info"])
        return HTML(f"""
        <div style="padding:12px 14px;margin:8px 0;border-radius:14px;
                    background:{bg};color:{color};border:1px solid {border};">
            {message}
        </div>
        """)

    @staticmethod
    def top_banner() -> HTML:
        return HTML(f"""
        <div style="
            padding:18px 22px;margin-bottom:12px;border-radius:22px;
            background:linear-gradient(135deg,#f8fafc,#eff6ff,#eef2ff);
            border:1px solid #dbeafe;box-shadow:0 8px 20px rgba(15,23,42,.06);
        ">
            <h2 style="margin:0;color:#0f172a;">8-Puzzle Local Search</h2>
            <p style="margin:6px 0 0;color:#475569;font-size:14px;line-height:1.6;">
                Bản code viết lại theo cấu trúc khác: có class xử lý puzzle, class thuật toán,
                class giao diện, bảng log Node / Frontier / Reached.<br>
                Link GitHub: <b>{REPO_LINK}</b>
            </p>
        </div>
        """)


# ============================================================
# 4. TRUY VẾT ĐƯỜNG ĐI
# ============================================================

def build_path(parent: ParentMap, target: State) -> List[Tuple[State, Optional[str]]]:
    route: List[Tuple[State, Optional[str]]] = []
    cur: Optional[State] = target
    while cur is not None:
        prev, act = parent.get(cur, (None, None))
        route.append((cur, act))
        cur = prev
    route.reverse()
    return route


# ============================================================
# 5. THUẬT TOÁN LOCAL SEARCH
# ============================================================

class LocalSearchSolver:
    def __init__(self, goal: State, heuristic_name: str = "manhattan", seed: Optional[int] = None) -> None:
        self.goal = goal
        self.h = PuzzleTools.choose_heuristic(heuristic_name)
        self.heuristic_name = heuristic_name
        self.labeler = NodeLabeler()
        if seed is not None:
            random.seed(seed)

    def _neighbor_summary(self, rows: List[Tuple[int, str, State]]) -> str:
        parts = []
        for h_value, action, state in rows:
            parts.append(f"{self.labeler.get(state)}:{ACTION_VI[action]}:h={h_value}")
        return " | ".join(parts)

    def random_restart_hill_climbing(
        self,
        start: State,
        max_restart: int = 30,
        step_limit: int = 60,
        first_use_input: bool = True,
        allow_sideway: bool = False,
        choose_style: str = "steepest",
    ) -> SearchResult:
        """
        Leo đồi khởi tạo ngẫu nhiên.
        choose_style:
        - steepest: chọn hàng xóm có h nhỏ nhất.
        - random_better: chọn ngẫu nhiên trong nhóm hàng xóm tốt hơn hiện tại.
        """
        parent: ParentMap = {}
        logs: List[dict] = []
        reached_global = set()
        expanded = 0
        best_state = start
        best_h = self.h(start, self.goal)

        for restart_id in range(max_restart + 1):
            if restart_id == 0 and first_use_input:
                current = start
                start_note = "Start người dùng nhập"
            else:
                current = PuzzleTools.random_valid(self.goal)
                start_note = "Start random mới"

            parent.setdefault(current, (None, None))
            reached_local = {current}
            reached_global.add(current)
            self.labeler.get(current)

            for step in range(step_limit + 1):
                current_h = self.h(current, self.goal)
                expanded += 1

                if current_h < best_h:
                    best_h = current_h
                    best_state = current

                if current == self.goal:
                    logs.append({
                        "Lần restart": restart_id,
                        "Bước": step,
                        "Node đang xét": self.labeler.get(current),
                        "Ma trận": PuzzleTools.to_short(current),
                        "h(n)": current_h,
                        "Frontier / Hàng xóm": "Goal",
                        "Reached": len(reached_global),
                        "Chọn": "Dừng",
                        "Nhận xét": "Tìm thấy Goal",
                    })
                    return SearchResult(
                        found=True,
                        message="Đã tìm thấy Goal bằng Leo đồi khởi tạo ngẫu nhiên.",
                        goal_state=current,
                        best_state=best_state,
                        best_h=best_h,
                        parent=parent,
                        logs=logs,
                        labeler=self.labeler,
                        expanded_count=expanded,
                    )

                neighbor_rows: List[Tuple[int, str, State]] = []
                for action, nxt in PuzzleTools.expand(current):
                    self.labeler.get(nxt)
                    neighbor_rows.append((self.h(nxt, self.goal), action, nxt))

                neighbor_rows.sort(key=lambda item: (item[0], ACTION_ORDER.index(item[1])))

                # Cách chọn khác nhau để code có form linh hoạt hơn.
                if choose_style == "random_better":
                    better = [x for x in neighbor_rows if x[0] < current_h]
                    equal = [x for x in neighbor_rows if x[0] == current_h and x[2] not in reached_local]
                    if better:
                        chosen_h, chosen_action, chosen_state = random.choice(better)
                    elif allow_sideway and equal:
                        chosen_h, chosen_action, chosen_state = random.choice(equal)
                    else:
                        chosen_h, chosen_action, chosen_state = neighbor_rows[0]
                else:
                    min_h = neighbor_rows[0][0]
                    best_group = [x for x in neighbor_rows if x[0] == min_h]
                    chosen_h, chosen_action, chosen_state = random.choice(best_group)

                can_go_down = chosen_h < current_h
                can_go_sideway = allow_sideway and chosen_h == current_h and chosen_state not in reached_local
                can_move = can_go_down or can_go_sideway

                if can_move:
                    parent.setdefault(chosen_state, (current, chosen_action))
                    reached_local.add(chosen_state)
                    reached_global.add(chosen_state)
                    decision = f"{self.labeler.get(chosen_state)} qua {ACTION_VI[chosen_action]}"
                    comment = "Đi tới trạng thái tốt hơn" if can_go_down else "Đi ngang plateau"
                else:
                    decision = "Không đi tiếp"
                    comment = "Kẹt local minimum/plateau nên restart"

                logs.append({
                    "Lần restart": restart_id,
                    "Bước": step,
                    "Node đang xét": self.labeler.get(current),
                    "Ma trận": PuzzleTools.to_short(current),
                    "h(n)": current_h,
                    "Frontier / Hàng xóm": self._neighbor_summary(neighbor_rows),
                    "Reached": len(reached_global),
                    "Chọn": decision,
                    "Nhận xét": f"{start_note}; {comment}",
                })

                if can_move:
                    current = chosen_state
                else:
                    break

        return SearchResult(
            found=False,
            message=f"Chưa tìm thấy Goal. Trạng thái tốt nhất có h={best_h}.",
            goal_state=None,
            best_state=best_state,
            best_h=best_h,
            parent=parent,
            logs=logs,
            labeler=self.labeler,
            expanded_count=expanded,
        )

    def local_beam_search(
        self,
        start: State,
        beam_width: int = 5,
        max_loop: int = 100,
        include_input_start: bool = True,
    ) -> SearchResult:
        """
        Local Beam Search:
        - Giữ k trạng thái tốt nhất.
        - Sinh hàng xóm của toàn bộ beam.
        - Lấy k ứng viên có h nhỏ nhất để làm beam mới.
        """
        parent: ParentMap = {}
        logs: List[dict] = []
        reached = set()
        expanded = 0

        beam: List[State] = []
        if include_input_start:
            beam.append(start)
            parent[start] = (None, None)
            reached.add(start)

        while len(beam) < beam_width:
            state = PuzzleTools.random_valid(self.goal)
            if state not in reached:
                beam.append(state)
                parent[state] = (None, None)
                reached.add(state)

        for state in beam:
            self.labeler.get(state)

        best_state = min(beam, key=lambda s: self.h(s, self.goal))
        best_h = self.h(best_state, self.goal)

        for loop in range(max_loop + 1):
            beam.sort(key=lambda s: (self.h(s, self.goal), self.labeler.get(s)))
            expanded += len(beam)

            # Kiểm tra Goal trong beam hiện tại
            for state in beam:
                h_value = self.h(state, self.goal)
                if h_value < best_h:
                    best_h = h_value
                    best_state = state
                if state == self.goal:
                    logs.append({
                        "Vòng lặp": loop,
                        "Beam hiện tại": " | ".join(f"{self.labeler.get(x)}(h={self.h(x, self.goal)})" for x in beam),
                        "Frontier / Ứng viên": "Có Goal trong beam",
                        "Reached": len(reached),
                        "Chọn beam mới": self.labeler.get(state),
                        "h tốt nhất": h_value,
                        "Nhận xét": "Tìm thấy Goal",
                    })
                    return SearchResult(
                        found=True,
                        message="Đã tìm thấy Goal bằng Local Beam Search.",
                        goal_state=state,
                        best_state=best_state,
                        best_h=best_h,
                        parent=parent,
                        logs=logs,
                        labeler=self.labeler,
                        expanded_count=expanded,
                    )

            candidates: List[Tuple[int, str, State, State]] = []
            seen_in_this_round = set()
            for source in beam:
                for action, nxt in PuzzleTools.expand(source):
                    if nxt in reached or nxt in seen_in_this_round:
                        continue
                    self.labeler.get(nxt)
                    seen_in_this_round.add(nxt)
                    candidates.append((self.h(nxt, self.goal), action, source, nxt))
                    parent.setdefault(nxt, (source, action))

            candidates.sort(key=lambda item: (item[0], self.labeler.get(item[2]), ACTION_ORDER.index(item[1])))

            if not candidates:
                logs.append({
                    "Vòng lặp": loop,
                    "Beam hiện tại": " | ".join(f"{self.labeler.get(x)}(h={self.h(x, self.goal)})" for x in beam),
                    "Frontier / Ứng viên": "Rỗng",
                    "Reached": len(reached),
                    "Chọn beam mới": "Không có",
                    "h tốt nhất": best_h,
                    "Nhận xét": "Không còn trạng thái mới để mở rộng",
                })
                break

            chosen = candidates[:beam_width]
            new_beam = [nxt for _, _, _, nxt in chosen]
            for state in new_beam:
                reached.add(state)
                h_value = self.h(state, self.goal)
                if h_value < best_h:
                    best_state = state
                    best_h = h_value

            candidate_text = " | ".join(
                f"{self.labeler.get(src)}--{ACTION_VI[act]}->{self.labeler.get(nxt)}(h={h})"
                for h, act, src, nxt in candidates[:18]
            )
            if len(candidates) > 18:
                candidate_text += f" | ... còn {len(candidates) - 18} ứng viên"

            chosen_text = " | ".join(f"{self.labeler.get(nxt)}(h={h})" for h, _, _, nxt in chosen)

            logs.append({
                "Vòng lặp": loop,
                "Beam hiện tại": " | ".join(f"{self.labeler.get(x)}(h={self.h(x, self.goal)})" for x in beam),
                "Frontier / Ứng viên": candidate_text,
                "Reached": len(reached),
                "Chọn beam mới": chosen_text,
                "h tốt nhất": best_h,
                "Nhận xét": f"Giữ {beam_width} node có h nhỏ nhất",
            })
            beam = new_beam

        return SearchResult(
            found=False,
            message=f"Chưa tìm thấy Goal. Trạng thái tốt nhất có h={best_h}.",
            goal_state=None,
            best_state=best_state,
            best_h=best_h,
            parent=parent,
            logs=logs,
            labeler=self.labeler,
            expanded_count=expanded,
        )


# ============================================================
# 6. GIAO DIỆN JUPYTER
# ============================================================

class NotebookApp:
    def __init__(self) -> None:
        self.start_input = widgets.Textarea(
            value="2 8 3\n1 6 4\n7 0 5",
            description="Start",
            layout=widgets.Layout(width="360px", height="95px"),
        )
        self.goal_input = widgets.Textarea(
            value="1 2 3\n8 0 4\n7 6 5",
            description="Goal",
            layout=widgets.Layout(width="360px", height="95px"),
        )

        self.algorithm = widgets.ToggleButtons(
            options=[
                ("Hill Climbing Restart", "hill"),
                ("Local Beam Search", "beam"),
            ],
            value="hill",
            description="Thuật toán",
            layout=widgets.Layout(width="520px"),
        )

        self.heuristic = widgets.RadioButtons(
            options=[("Manhattan", "manhattan"), ("Số ô sai", "misplaced")],
            value="manhattan",
            description="h(n)",
        )

        self.hill_style = widgets.Dropdown(
            options=[
                ("Chọn node có h nhỏ nhất", "steepest"),
                ("Random trong nhóm tốt hơn", "random_better"),
            ],
            value="steepest",
            description="Kiểu leo",
            layout=widgets.Layout(width="360px"),
        )

        self.restart_count = widgets.IntSlider(
            value=30, min=0, max=200, step=1,
            description="Restart",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.step_count = widgets.IntSlider(
            value=60, min=1, max=300, step=1,
            description="Step",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.allow_sideway = widgets.Checkbox(
            value=False,
            description="Cho phép đi ngang nếu h bằng nhau",
            indent=False,
        )

        self.beam_k = widgets.IntSlider(
            value=5, min=2, max=20, step=1,
            description="Beam k",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.beam_loop = widgets.IntSlider(
            value=100, min=1, max=500, step=1,
            description="Loop",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.use_input_start = widgets.Checkbox(
            value=True,
            description="Dùng Start người nhập làm trạng thái đầu tiên",
            indent=False,
        )
        self.seed = widgets.IntText(
            value=42,
            description="Seed",
            layout=widgets.Layout(width="190px"),
        )

        self.run_btn = widgets.Button(
            description="Chạy",
            icon="play",
            button_style="success",
            layout=widgets.Layout(width="125px", height="40px"),
        )
        self.random_btn = widgets.Button(
            description="Tạo Start random",
            icon="random",
            button_style="info",
            layout=widgets.Layout(width="170px", height="40px"),
        )
        self.clear_btn = widgets.Button(
            description="Xóa",
            icon="trash",
            button_style="warning",
            layout=widgets.Layout(width="110px", height="40px"),
        )
        self.output = widgets.Output()

        self.random_btn.on_click(self.random_start)
        self.clear_btn.on_click(self.clear_output)
        self.run_btn.on_click(self.run)

    def random_start(self, _=None) -> None:
        with self.output:
            try:
                goal = PuzzleTools.parse(self.goal_input.value)
                state = PuzzleTools.random_valid(goal)
                self.start_input.value = PuzzleTools.to_multiline(state)
                clear_output()
                display(BoardView.box("Đã tạo Start ngẫu nhiên có thể giải được.", "ok"))
                BoardView.many([state, goal], goal, ["Start random", "Goal"], ["Mới tạo", "Đích"])
            except Exception as exc:
                clear_output()
                display(BoardView.box(f"<b>Lỗi:</b> {exc}", "bad"))

    def clear_output(self, _=None) -> None:
        with self.output:
            clear_output()

    def _show_start_goal(self, start: State, goal: State) -> None:
        h_m = PuzzleTools.h_manhattan(start, goal)
        h_s = PuzzleTools.h_misplaced(start, goal)
        display(BoardView.box(f"<b>Start / Goal</b> &nbsp; | &nbsp; Manhattan = {h_m} &nbsp; | &nbsp; Số ô sai = {h_s}", "info"))
        BoardView.many(
            [start, goal],
            goal,
            ["Start", "Goal"],
            [f"hM={h_m}, hSai={h_s}", "Trạng thái đích"],
        )

    def _show_result_path(self, result: SearchResult, goal: State) -> None:
        if result.found and result.goal_state is not None:
            route = build_path(result.parent, result.goal_state)
            states = [state for state, _ in route]
            titles = []
            notes = []
            for i, (state, action) in enumerate(route):
                label = result.labeler.get(state)
                if i == 0:
                    titles.append(f"{label} - Start")
                    notes.append(f"h={PuzzleTools.h_manhattan(state, goal)}")
                else:
                    titles.append(f"{label} - Bước {i}")
                    notes.append(f"Đi {ACTION_VI.get(action, action)}, h={PuzzleTools.h_manhattan(state, goal)}")
            display(HTML("<h3 style='color:#0f172a'>Đường đi nghiệm</h3>"))
            display(BoardView.box(f"Số bước nghiệm: <b>{len(states) - 1}</b>", "ok"))
            BoardView.many(states, goal, titles, notes, limit=30)
        else:
            display(HTML("<h3 style='color:#0f172a'>Trạng thái tốt nhất tìm được</h3>"))
            BoardView.many(
                [result.best_state, goal],
                goal,
                ["Best state", "Goal"],
                [f"h={result.best_h}", "Đích"],
            )

    def _show_table(self, result: SearchResult) -> None:
        display(HTML("<h3 style='color:#0f172a'>Bảng Node / Frontier / Reached</h3>"))
        if not result.logs:
            display(HTML("<i>Không có log để hiển thị.</i>"))
            return
        df = pd.DataFrame(result.logs)
        display(df.head(160))
        if len(df) > 160:
            display(HTML(f"<i>Bảng có {len(df)} dòng, chỉ hiển thị 160 dòng đầu.</i>"))

    def run(self, _=None) -> None:
        with self.output:
            clear_output()
            try:
                start = PuzzleTools.parse(self.start_input.value)
                goal = PuzzleTools.parse(self.goal_input.value)
            except Exception as exc:
                display(BoardView.box(f"<b>Lỗi nhập dữ liệu:</b> {exc}", "bad"))
                return

            self._show_start_goal(start, goal)

            if not PuzzleTools.can_solve(start, goal):
                display(BoardView.box(
                    "<b>Start không thể biến đổi về Goal.</b><br>"
                    "Với 8-puzzle 3x3, parity số nghịch thế không phù hợp.",
                    "bad",
                ))
                return

            solver = LocalSearchSolver(
                goal=goal,
                heuristic_name=self.heuristic.value,
                seed=self.seed.value,
            )

            if self.algorithm.value == "hill":
                result = solver.random_restart_hill_climbing(
                    start=start,
                    max_restart=self.restart_count.value,
                    step_limit=self.step_count.value,
                    first_use_input=self.use_input_start.value,
                    allow_sideway=self.allow_sideway.value,
                    choose_style=self.hill_style.value,
                )
            else:
                result = solver.local_beam_search(
                    start=start,
                    beam_width=self.beam_k.value,
                    max_loop=self.beam_loop.value,
                    include_input_start=self.use_input_start.value,
                )

            kind = "ok" if result.found else "warn"
            display(BoardView.box(
                f"<b>Kết quả:</b> {result.message}<br>"
                f"Node đã mở rộng/xét: <b>{result.expanded_count}</b> | h tốt nhất: <b>{result.best_h}</b>",
                kind,
            ))
            self._show_result_path(result, goal)
            self._show_table(result)
            display(BoardView.box(
                "<b>Ghi nhớ lý thuyết:</b><br>"
                "- Hill Climbing chỉ chọn dựa trên hàng xóm hiện tại nên dễ kẹt local minimum/plateau.<br>"
                "- Random Restart thử nhiều điểm bắt đầu để tăng khả năng thoát kẹt.<br>"
                "- Local Beam Search giữ k trạng thái cùng lúc, mở rộng rộng hơn nhưng vẫn không đảm bảo tối ưu.",
                "info",
            ))

    def display(self) -> None:
        left = widgets.VBox([
            self.start_input,
            self.goal_input,
            widgets.HBox([self.random_btn, self.clear_btn]),
        ])
        general_box = widgets.VBox([
            self.algorithm,
            self.heuristic,
            self.use_input_start,
            widgets.HBox([self.seed, self.run_btn]),
        ])
        hill_box = widgets.VBox([
            self.hill_style,
            self.restart_count,
            self.step_count,
            self.allow_sideway,
        ])
        beam_box = widgets.VBox([
            self.beam_k,
            self.beam_loop,
        ])
        tabs = widgets.Tab(children=[general_box, hill_box, beam_box])
        tabs.set_title(0, "Cấu hình chung")
        tabs.set_title(1, "Hill Climbing")
        tabs.set_title(2, "Beam Search")

        display(BoardView.top_banner())
        display(widgets.HBox([left, tabs], layout=widgets.Layout(gap="22px", align_items="flex-start")))
        display(self.output)

        with self.output:
            clear_output()
            try:
                start = PuzzleTools.parse(self.start_input.value)
                goal = PuzzleTools.parse(self.goal_input.value)
                display(BoardView.box("Nhấn nút <b>Chạy</b> để bắt đầu thuật toán.", "info"))
                BoardView.many([start, goal], goal, ["Start mặc định", "Goal mặc định"], ["Có thể sửa", "Đích"])
            except Exception:
                pass


# ============================================================
# 7. KHỞI ĐỘNG APP
# ============================================================

app = NotebookApp()
app.display()
