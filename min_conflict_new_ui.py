


#https://github.com/ngquyn2602/Bai_tap_tri_tue_nhan_tao_AI
import random
import time
import gradio as gr


# ============================================================
# 1. CÁC HÀM KIỂM TRA RÀNG BUỘC N-QUEENS
# assignment[cột] = hàng
# ============================================================

def conflicts_for(assignment, col, row):
    """Đếm số quân hậu xung đột với vị trí (col, row)."""
    total = 0

    for other_col, other_row in enumerate(assignment):
        if other_col == col or other_row is None:
            continue

        same_row = other_row == row
        same_diagonal = abs(other_col - col) == abs(other_row - row)

        if same_row or same_diagonal:
            total += 1

    return total


def total_conflicts(assignment):
    """Đếm tổng số cặp quân hậu đang xung đột."""
    total = 0
    n = len(assignment)

    for col1 in range(n):
        row1 = assignment[col1]

        if row1 is None:
            continue

        for col2 in range(col1 + 1, n):
            row2 = assignment[col2]

            if row2 is None:
                continue

            same_row = row1 == row2
            same_diagonal = abs(col1 - col2) == abs(row1 - row2)

            if same_row or same_diagonal:
                total += 1

    return total


def conflict_columns(assignment):
    """Trả về tập hợp các cột đang có quân hậu xung đột."""
    bad_columns = set()
    n = len(assignment)

    for col1 in range(n):
        row1 = assignment[col1]

        if row1 is None:
            continue

        for col2 in range(col1 + 1, n):
            row2 = assignment[col2]

            if row2 is None:
                continue

            same_row = row1 == row2
            same_diagonal = abs(col1 - col2) == abs(row1 - row2)

            if same_row or same_diagonal:
                bad_columns.add(col1)
                bad_columns.add(col2)

    return bad_columns


# ============================================================
# 2. THUẬT TOÁN MIN-CONFLICTS
# ============================================================

def solve_min_conflicts(n, max_steps=5000, max_restarts=20):
    """
    Min-Conflicts:

    1. Tạo một phép gán đầy đủ.
    2. Chọn một quân hậu đang xung đột.
    3. Chuyển hậu tới hàng tạo ít xung đột nhất.
    4. Nếu chưa tìm thấy nghiệm thì khởi động lại.
    """

    start_time = time.perf_counter()

    logs = []
    total_steps = 0
    assignment = [None] * n

    for restart in range(max_restarts + 1):

        # Tạo phép gán đầy đủ ban đầu.
        assignment = [None] * n
        columns = list(range(n))
        random.shuffle(columns)

        for col in columns:
            scores = [
                conflicts_for(assignment, col, row)
                for row in range(n)
            ]

            minimum_score = min(scores)

            best_rows = [
                row
                for row, score in enumerate(scores)
                if score == minimum_score
            ]

            assignment[col] = random.choice(best_rows)

        current_conflicts = total_conflicts(assignment)

        logs.append(
            f"Khởi tạo lần {restart + 1}: "
            f"{current_conflicts} cặp xung đột."
        )

        for _ in range(max_steps):

            conflicted_columns = [
                col
                for col, row in enumerate(assignment)
                if conflicts_for(assignment, col, row) > 0
            ]

            # Không còn cột xung đột nghĩa là đã có nghiệm.
            if not conflicted_columns:
                elapsed = time.perf_counter() - start_time

                return {
                    "algorithm": "Min-Conflicts",
                    "success": True,
                    "assignment": assignment.copy(),
                    "steps": total_steps,
                    "conflicts": 0,
                    "backtracks": 0,
                    "restarts": restart,
                    "elapsed": elapsed,
                    "logs": logs,
                }

            # Chọn ngẫu nhiên một cột đang xung đột.
            col = random.choice(conflicted_columns)

            row_scores = [
                conflicts_for(assignment, col, row)
                for row in range(n)
            ]

            minimum_score = min(row_scores)

            best_rows = [
                row
                for row, score in enumerate(row_scores)
                if score == minimum_score
            ]

            # Nếu nhiều hàng cùng tốt nhất, chọn ngẫu nhiên.
            assignment[col] = random.choice(best_rows)
            total_steps += 1

            current_conflicts = total_conflicts(assignment)

            # Giới hạn nhật ký để giao diện không quá dài.
            if total_steps <= 200:
                logs.append(
                    f"Bước {total_steps}: chuyển hậu cột {col + 1} "
                    f"đến hàng {assignment[col] + 1}; "
                    f"còn {current_conflicts} cặp xung đột."
                )

    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": "Min-Conflicts",
        "success": False,
        "assignment": assignment.copy(),
        "steps": total_steps,
        "conflicts": total_conflicts(assignment),
        "backtracks": 0,
        "restarts": max_restarts,
        "elapsed": elapsed,
        "logs": logs,
    }


# ============================================================
# 3. FORWARD CHECKING + MRV + LCV
# ============================================================

def solve_forward_checking(n):
    """
    Forward Checking kết hợp:

    - Backtracking.
    - MRV: chọn biến có miền nhỏ nhất.
    - LCV: thử giá trị ít ảnh hưởng nhất.
    - Trail: phục hồi miền khi quay lui.
    """

    start_time = time.perf_counter()

    assignment = [None] * n
    domains = [set(range(n)) for _ in range(n)]

    steps = 0
    backtracks = 0
    logs = []

    def choose_column():
        """MRV: chọn cột chưa gán có miền nhỏ nhất."""

        unassigned_columns = [
            col
            for col in range(n)
            if assignment[col] is None
        ]

        return min(
            unassigned_columns,
            key=lambda col: (len(domains[col]), col),
        )

    def value_impact(col, row):
        """
        LCV: đếm số giá trị của các biến khác
        sẽ bị loại nếu đặt hậu tại (col, row).
        """

        impact = 0

        for other_col in range(n):

            if other_col == col:
                continue

            if assignment[other_col] is not None:
                continue

            distance = abs(other_col - col)

            attacked_rows = {
                row,
                row - distance,
                row + distance,
            }

            impact += sum(
                attacked_row in domains[other_col]
                for attacked_row in attacked_rows
            )

        return impact

    def ordered_rows(col):
        """Sắp xếp miền theo LCV."""

        rows = list(domains[col])

        # Xáo trộn trước để các giá trị bằng nhau không luôn cố định.
        random.shuffle(rows)

        rows.sort(
            key=lambda row: value_impact(col, row)
        )

        return rows

    def search():
        nonlocal steps, backtracks

        # Đã gán hết tất cả các cột.
        if all(row is not None for row in assignment):
            return True

        col = choose_column()

        for row in ordered_rows(col):

            # Kiểm tra với các quân hậu đã được gán.
            if conflicts_for(assignment, col, row) != 0:
                continue

            assignment[col] = row
            steps += 1

            # Lưu những giá trị bị xóa để có thể phục hồi.
            trail = []
            consistent = True

            for other_col in range(n):

                if other_col == col:
                    continue

                if assignment[other_col] is not None:
                    continue

                distance = abs(other_col - col)

                attacked_rows = (
                    row,
                    row - distance,
                    row + distance,
                )

                for attacked_row in attacked_rows:

                    if attacked_row in domains[other_col]:
                        domains[other_col].remove(attacked_row)

                        trail.append(
                            (other_col, attacked_row)
                        )

                # Một miền bị rỗng nghĩa là nhánh này thất bại.
                if not domains[other_col]:
                    consistent = False
                    break

            if steps <= 200:
                logs.append(
                    f"Gán cột {col + 1} = hàng {row + 1}; "
                    f"kiểm tra trước miền của các cột còn lại."
                )

            if consistent:

                if search():
                    return True

            # Phục hồi lại các giá trị đã xóa.
            for changed_col, changed_row in reversed(trail):
                domains[changed_col].add(changed_row)

            # Hủy phép gán hiện tại.
            assignment[col] = None
            backtracks += 1

            if backtracks <= 200:
                logs.append(
                    f"Quay lui khỏi cột {col + 1}; "
                    f"tổng số lần quay lui = {backtracks}."
                )

        return False

    success = search()
    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": "Forward Checking",
        "success": success,
        "assignment": assignment.copy(),
        "steps": steps,
        "conflicts": total_conflicts(assignment),
        "backtracks": backtracks,
        "restarts": 0,
        "elapsed": elapsed,
        "logs": logs,
    }


# ============================================================
# 4. VẼ BÀN CỜ BẰNG HTML
# Không cần Tkinter, Pillow hoặc Matplotlib.
# ============================================================

def board_html(assignment):
    n = len(assignment)
    bad_columns = conflict_columns(assignment)

    cells = []

    for row in range(n):

        for col in range(n):

            light_square = (row + col) % 2 == 0

            background = (
                "linear-gradient(135deg,#F8FAFC,#E0F2FE)"
                if light_square
                else "linear-gradient(135deg,#1E3A8A,#2563EB)"
            )

            has_queen = assignment[col] == row

            queen_color = (
                "#DC2626"
                if col in bad_columns
                else ("#0F172A" if light_square else "#FFFFFF")
            )

            if has_queen:
                queen = (
                    f"<div style='"
                    f"width:72%;height:72%;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"border-radius:18px;"
                    f"background:rgba(255,255,255,0.72);"
                    f"box-shadow:0 10px 22px rgba(15,23,42,0.18);"
                    f"font-size:clamp(22px,5vw,56px);"
                    f"color:{queen_color};"
                    f"line-height:1;'>♛</div>"
                )
            else:
                queen = ""

            cell = f"""
            <div style="
                aspect-ratio:1/1;
                display:flex;
                align-items:center;
                justify-content:center;
                background:{background};
                border:1px solid rgba(255,255,255,0.45);
                box-sizing:border-box;
            ">
                {queen}
            </div>
            """

            cells.append(cell)

    return f"""
    <div style="
        width:min(720px,96vw);
        margin:12px auto 4px auto;
        padding:14px;
        background:linear-gradient(135deg,#FFFFFF,#EFF6FF);
        border:1px solid #DBEAFE;
        border-radius:24px;
        box-shadow:0 24px 60px rgba(15,23,42,0.16);
        box-sizing:border-box;
    ">
        <div style="
            display:grid;
            grid-template-columns:repeat({n},1fr);
            overflow:hidden;
            border-radius:18px;
            border:4px solid #0F172A;
            box-sizing:border-box;
        ">
            {''.join(cells)}
        </div>
    </div>

    <div style="
        text-align:center;
        color:#475569;
        margin-top:10px;
        font-weight:600;
    ">
        Quân hậu màu đỏ là quân đang xung đột
    </div>
    """


def status_html(message, success=None):

    if success is True:
        text_color = "#065F46"
        background = "linear-gradient(135deg,#ECFDF5,#D1FAE5)"
        border = "#A7F3D0"

    elif success is False:
        text_color = "#991B1B"
        background = "linear-gradient(135deg,#FEF2F2,#FEE2E2)"
        border = "#FECACA"

    else:
        text_color = "#1E3A8A"
        background = "linear-gradient(135deg,#EFF6FF,#DBEAFE)"
        border = "#BFDBFE"

    return f"""
    <div style="
        padding:16px 18px;
        border-radius:18px;
        color:{text_color};
        background:{background};
        border:1px solid {border};
        font-weight:800;
        box-shadow:0 12px 30px rgba(15,23,42,0.08);
    ">
        {message}
    </div>
    """


# ============================================================
# 5. HÀM CHẠY GIAO DIỆN
# ============================================================

RESULT_HEADERS = [
    "Thuật toán",
    "Kết quả",
    "Bước",
    "Xung đột",
    "Quay lui",
    "Khởi động lại",
    "Thời gian (giây)",
]


def run_application(
    n,
    algorithm,
    max_steps,
    max_restarts,
):
    n = int(n)
    max_steps = int(max_steps)
    max_restarts = int(max_restarts)

    # Chạy Min-Conflicts.
    if algorithm == "Min-Conflicts":

        results = [
            solve_min_conflicts(
                n=n,
                max_steps=max_steps,
                max_restarts=max_restarts,
            )
        ]

    # Chạy Forward Checking.
    elif algorithm == "Forward Checking":

        results = [
            solve_forward_checking(n=n)
        ]

    # Chạy và so sánh cả hai.
    else:

        results = [
            solve_min_conflicts(
                n=n,
                max_steps=max_steps,
                max_restarts=max_restarts,
            ),
            solve_forward_checking(n=n),
        ]

    result_table = []
    all_logs = []

    for result in results:

        result_table.append(
            [
                result["algorithm"],
                (
                    "Thành công"
                    if result["success"]
                    else "Chưa thành công"
                ),
                result["steps"],
                result["conflicts"],
                result["backtracks"],
                result["restarts"],
                round(result["elapsed"], 6),
            ]
        )

        all_logs.append(
            f'========== {result["algorithm"]} =========='
        )

        all_logs.extend(result["logs"])

        all_logs.append(
            f'Kết quả: '
            f'{"Thành công" if result["success"] else "Chưa thành công"}'
        )

        all_logs.append(
            f'Số bước: {result["steps"]}'
        )

        all_logs.append(
            f'Số lần quay lui: {result["backtracks"]}'
        )

        all_logs.append(
            f'Thời gian: {result["elapsed"]:.6f} giây'
        )

        all_logs.append("")

    # Khi so sánh cả hai, hiển thị bàn cờ của thuật toán cuối cùng.
    final_result = results[-1]

    if final_result["success"]:

        message = (
            f'{final_result["algorithm"]}: '
            f'đã tìm thấy nghiệm hợp lệ.'
        )

    else:

        message = (
            f'{final_result["algorithm"]}: '
            f'chưa tìm thấy nghiệm.'
        )

    return (
        board_html(final_result["assignment"]),
        status_html(
            message,
            final_result["success"],
        ),
        result_table,
        "\n".join(all_logs[-350:]),
    )


def reset_application(n):
    n = int(n)

    return (
        board_html([None] * n),
        status_html("Đã đặt lại."),
        [],
        "",
    )


def update_empty_board(n):
    n = int(n)

    return board_html([None] * n)


# ============================================================
# 6. TẠO GIAO DIỆN GRADIO
# ============================================================

APP_CSS = """
.gradio-container {
    background: radial-gradient(circle at top left,#DBEAFE 0,#F8FAFC 34%,#EEF2FF 100%) !important;
}
.main-card {
    padding: 26px;
    border-radius: 28px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(191,219,254,0.95);
    box-shadow: 0 24px 80px rgba(15,23,42,0.12);
}
.header-card {
    padding: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg,#0F172A,#1D4ED8);
    color: white;
    box-shadow: 0 24px 60px rgba(30,64,175,0.28);
}
.header-card h1 {
    margin: 0 0 10px 0;
    font-size: 38px;
    line-height: 1.1;
}
.header-card p {
    margin: 0;
    color: #DBEAFE;
    font-size: 17px;
}
.panel-card {
    padding: 20px;
    border-radius: 22px;
    background: rgba(255,255,255,0.92);
    border: 1px solid #DBEAFE;
    box-shadow: 0 14px 34px rgba(15,23,42,0.08);
}
.section-title h2 {
    color: #0F172A;
    margin-bottom: 4px;
}
button.primary {
    border-radius: 14px !important;
}
textarea, input, select {
    border-radius: 14px !important;
}
"""

with gr.Blocks(
    title="N-Queens AI Solver",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css=APP_CSS,
) as demo:

    with gr.Column(elem_classes="main-card"):

        gr.HTML(
            """
            <div class="header-card">
                <h1>N-Queens Solver</h1>
                <p>Trực quan hóa Min-Conflicts và Forward Checking trên bàn cờ N-Queens.</p>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(scale=1, elem_classes="panel-card"):

                gr.Markdown("### Thiết lập")

                n_input = gr.Slider(
                    minimum=4,
                    maximum=20,
                    value=8,
                    step=1,
                    label="Kích thước bàn cờ N",
                )

                algorithm_input = gr.Dropdown(
                    choices=[
                        "Min-Conflicts",
                        "Forward Checking",
                        "So sánh cả hai",
                    ],
                    value="Min-Conflicts",
                    label="Thuật toán",
                )

                with gr.Row():

                    max_steps_input = gr.Number(
                        value=5000,
                        precision=0,
                        label="Số bước tối đa",
                    )

                    max_restarts_input = gr.Number(
                        value=20,
                        precision=0,
                        label="Số lần khởi động lại",
                    )

                with gr.Row():

                    start_button = gr.Button(
                        "▶ Chạy thuật toán",
                        variant="primary",
                    )

                    reset_button = gr.Button(
                        "↻ Làm mới"
                    )

            with gr.Column(scale=1, elem_classes="panel-card"):

                gr.Markdown("### Trạng thái")

                status_output = gr.HTML(
                    status_html("Sẵn sàng.")
                )

                result_output = gr.Dataframe(
                    headers=RESULT_HEADERS,
                    value=[],
                    interactive=False,
                    label="Bảng kết quả",
                )

        gr.Markdown("## Bàn cờ", elem_classes="section-title")

        board_output = gr.HTML(
            board_html([None] * 8)
        )

        log_output = gr.Textbox(
            value="",
            lines=16,
            interactive=False,
            label="Nhật ký thuật toán",
        )

    start_button.click(
        fn=run_application,
        inputs=[
            n_input,
            algorithm_input,
            max_steps_input,
            max_restarts_input,
        ],
        outputs=[
            board_output,
            status_output,
            result_output,
            log_output,
        ],
    )

    reset_button.click(
        fn=reset_application,
        inputs=n_input,
        outputs=[
            board_output,
            status_output,
            result_output,
            log_output,
        ],
        queue=False,
    )

    n_input.change(
        fn=update_empty_board,
        inputs=n_input,
        outputs=board_output,
        queue=False,
    )


# Quan trọng:
# Không có tham số css trong launch().
demo.launch(
    share=True
)