"""生成 ArUco 标记点 0/1/2/3：一张 A4 上一组完整四个。"""

from pathlib import Path

import cv2
import numpy as np

# 打印分辨率与物理尺寸
DPI = 300
A4_MM = (210.0, 297.0)  # 宽 x 高
MARKER_MM = 100.0  # 单个标记边长 10 cm
MARGIN_MM = 5.0
GAP_MM = 5.0  # 组内间距（宽度紧张时会自动缩小）
LABEL_MM = 5.0
HEADER_MM = 8.0
DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)  # 2x2：左上0 右上1 / 左下2 右下3


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def get_dictionary(dict_id: int = DICT_ID):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def generate_marker(marker_id: int, size_px: int, dictionary=None) -> np.ndarray:
    dictionary = dictionary or get_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker, 1)
    return marker


def create_a4_sheet(
    marker_ids: tuple[int, ...] = MARKER_IDS,
    marker_mm: float = MARKER_MM,
    dpi: int = DPI,
) -> np.ndarray:
    """在 A4 居中放置一组 2x2 的 0/1/2/3。"""
    width = mm_to_px(A4_MM[0], dpi)
    height = mm_to_px(A4_MM[1], dpi)
    marker_px = mm_to_px(marker_mm, dpi)
    margin = mm_to_px(MARGIN_MM, dpi)
    label_h = mm_to_px(LABEL_MM, dpi)
    header = mm_to_px(HEADER_MM, dpi)

    # 宽度放不下时压缩组内水平间距
    avail_w = width - 2 * margin
    gap_x_mm = min(GAP_MM, max(0.0, (A4_MM[0] - 2 * MARGIN_MM - 2 * marker_mm)))
    gap_x = mm_to_px(gap_x_mm, dpi)
    gap_y = mm_to_px(GAP_MM, dpi)

    cell_h = marker_px + label_h
    group_w = 2 * marker_px + gap_x
    group_h = 2 * cell_h + gap_y
    if group_w > avail_w:
        raise RuntimeError("A4 宽度放不下 2 个 10cm 标记，请减小 MARKER_MM 或 MARGIN_MM")

    origin_x = (width - group_w) // 2
    origin_y = margin + header + (height - 2 * margin - header - group_h) // 2

    sheet = np.full((height, width), 255, dtype=np.uint8)
    dictionary = get_dictionary()
    cache = {mid: generate_marker(mid, marker_px, dictionary) for mid in marker_ids}
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for marker_id, (col, row) in zip(marker_ids, positions):
        x = origin_x + col * (marker_px + gap_x)
        y = origin_y + row * (cell_h + gap_y)
        sheet[y : y + marker_px, x : x + marker_px] = cache[marker_id]

        label = f"ID {marker_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x + (marker_px - tw) // 2
        ty = y + marker_px + th + mm_to_px(1.0, dpi)
        cv2.putText(sheet, label, (tx, ty), font, scale, 0, thickness, cv2.LINE_AA)

    note = f"ArUco DICT_4X4_50 | {marker_mm:.0f}mm (10cm) | 1 group 0-3 | print 100%"
    cv2.putText(
        sheet,
        note,
        (margin, margin + mm_to_px(4.0, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        1,
        cv2.LINE_AA,
    )
    return sheet


def main():
    out_dir = Path(__file__).resolve().parent
    sheet = create_a4_sheet()
    sheet_path = out_dir / "aruco_0_1_2_3_a4.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"已生成 A4 排版图: {sheet_path}")
    print("布局: 2x2 一组完整标记 0/1/2/3")

    dictionary = get_dictionary()
    marker_px = mm_to_px(MARKER_MM)
    for marker_id in MARKER_IDS:
        marker = generate_marker(marker_id, marker_px, dictionary)
        path = out_dir / f"aruco_{marker_id}.png"
        cv2.imwrite(str(path), marker)
        print(f"已生成单独标记: {path}")

    print(
        f"\n打印提示: 使用 A4、实际大小/100% 缩放打印；"
        f"单个标记边长约 {MARKER_MM:.0f} mm（10 cm）。"
    )


if __name__ == "__main__":
    main()
