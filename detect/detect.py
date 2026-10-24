"""ArUco 标记点 0/1/2/3 识别逻辑。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_IDS = frozenset({0, 1, 2, 3})

# corners 顺序：0左上 / 1右上 / 2右下 / 3左下
# 各 ID 选用的角点作为业务坐标
CORNER_INDEX_BY_ID = {
    0: 2,  # 右下角
    1: 3,  # 左下角
    2: 1,  # 右上角
    3: 0,  # 左上角
}


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray  # shape (4, 2), 顺时针：左上/右上/右下/左下
    point: tuple[float, float]  # 按 ID 规则选取的角点坐标
    center: tuple[float, float]  # 矩形几何中心（仅供参考）


def get_dictionary(dict_id: int = DICT_ID):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def get_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


class ArucoDetector:
    """封装 OpenCV ArUco 检测，兼容新旧 API。"""

    def __init__(self, dict_id: int = DICT_ID, target_ids: frozenset[int] | None = None):
        self.dictionary = get_dictionary(dict_id)
        self.parameters = get_detector_parameters()
        self.target_ids = TARGET_IDS if target_ids is None else frozenset(target_ids)
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr.ndim == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )

        results: list[Detection] = []
        if ids is None:
            return results

        for corner, marker_id in zip(corners, ids.flatten()):
            mid = int(marker_id)
            if mid not in self.target_ids:
                continue
            pts = corner.reshape(4, 2).astype(np.float32)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            corner_idx = CORNER_INDEX_BY_ID.get(mid, 0)
            px, py = float(pts[corner_idx, 0]), float(pts[corner_idx, 1])
            results.append(
                Detection(
                    marker_id=mid,
                    corners=pts,
                    point=(px, py),
                    center=(cx, cy),
                )
            )
        return results

    def draw(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame_bgr.copy()
        if not detections:
            return out

        corners = [d.corners.reshape(1, 4, 2) for d in detections]
        ids = np.array([[d.marker_id] for d in detections], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

        for det in detections:
            px, py = int(det.point[0]), int(det.point[1])
            cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(
                out,
                f"ID {det.marker_id}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out


def detect_markers(frame_bgr: np.ndarray) -> list[Detection]:
    """便捷函数：检测画面中的 0/1/2/3。"""
    return ArucoDetector().detect(frame_bgr)


def main():
    """本地摄像头实时检测；也可改 URL 对接手机推流。"""
    # 与 main.py 一致时可改为: "http://172.20.10.3:8080/video"
    source = "http://172.20.10.3:8080/video"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = ArucoDetector()
    print("ArUco 检测中，按 q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("无法读取画面")
            break

        detections = detector.detect(frame)
        vis = detector.draw(frame, detections)

        if detections:
            found = ", ".join(f"{d.marker_id}@({d.point[0]:.0f},{d.point[1]:.0f})" for d in detections)
            cv2.putText(
                vis,
                f"found: {found}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("ArUco Detect", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
