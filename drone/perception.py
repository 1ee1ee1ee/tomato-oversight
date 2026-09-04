"""온보드 인식 계층.

백엔드 세 개가 같은 인터페이스를 구현한다.

- ``MockSource``  : 하드웨어 없이 도는 방 시뮬레이션. 테스트와 데모용.
- ``OakDSource``  : OAK-D Lite. 카메라 안에서 추론이 끝나고 3D 좌표까지 나온다.
- ``OnnxSource``  : Pi 카메라 + ONNX Runtime(또는 Hailo). 깊이는 추정값이다.

셋 다 ``read(telem) -> Perception | None`` 하나만 노출한다. 상위 계층은
어떤 카메라가 달렸는지 몰라도 된다.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

from . import rangefinders
from .state import Detection, Perception, Telemetry


class PerceptionSource(Protocol):
    def read(self, telem: Telemetry | None) -> Perception | None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Mock — 방 하나를 시뮬레이션한다. 폐루프 테스트가 가능해진다.
# ---------------------------------------------------------------------------


class MockSource:
    """직육면체 방 안에 대상 하나를 놓고, 기체 위치에서 본 모습을 계산한다.

    실제 카메라처럼 화각 밖이면 안 보이고, 벽에 가까우면 거리계 값이 줄어든다.
    덕분에 ArduPilot 없이도 판단·안전 계층 전체를 돌려볼 수 있다.
    """

    def __init__(
        self,
        target_ned: tuple[float, float] = (-2.2, 1.8),
        label: str = "person",
        room_half_m: float = 5.0,
        fov_rad: float = 1.2,       # 약 69°
        max_range_m: float = 8.0,
    ) -> None:
        self.target_ned = target_ned
        self.label = label
        self.room_half_m = room_half_m
        self.fov_rad = fov_rad
        self.max_range_m = max_range_m

    def read(self, telem: Telemetry | None) -> Perception | None:
        if telem is None:
            return None

        detections: list[Detection] = []
        tn, te = self.target_ned
        dn, de = tn - telem.north_m, te - telem.east_m
        dist = math.hypot(dn, de)

        # 월드 상대 벡터를 기수 방향 기준으로 회전
        c, s = math.cos(telem.yaw_rad), math.sin(telem.yaw_rad)
        fwd = dn * c + de * s        # 전방 성분
        right = -dn * s + de * c     # 우측 성분

        if dist <= self.max_range_m and fwd > 0:
            bearing = math.atan2(right, fwd)
            if abs(bearing) <= self.fov_rad / 2:
                # 가까울수록 확신도가 오르는, 그럴듯한 모델 거동
                conf = max(0.4, min(0.95, 1.0 - dist / (self.max_range_m * 1.5)))
                detections.append(
                    Detection(
                        label=self.label,
                        confidence=round(conf, 3),
                        x_m=round(fwd, 3),
                        y_m=round(right, 3),
                        z_m=0.0,
                        has_depth=True,
                    )
                )

        return Perception(
            detections=tuple(detections),
            front_m=self._wall_distance(telem, 0.0),
            left_m=self._wall_distance(telem, -math.pi / 2),
            right_m=self._wall_distance(telem, math.pi / 2),
        )

    def _wall_distance(self, telem: Telemetry, offset: float) -> float:
        """기수 기준 offset 방향으로 벽까지의 거리."""
        heading = telem.yaw_rad + offset
        dn, de = math.cos(heading), math.sin(heading)
        best = self.max_range_m
        half = self.room_half_m
        for pos, direction, bound in (
            (telem.north_m, dn, half),
            (telem.east_m, de, half),
        ):
            if abs(direction) < 1e-9:
                continue
            for wall in (bound, -bound):
                t = (wall - pos) / direction
                if 0 < t < best:
                    best = t
        return round(best, 3)

    def close(self) -> None:  # 대칭성을 위해 존재
        return None


# ---------------------------------------------------------------------------
# OAK-D Lite — 추론이 카메라 안에서 끝난다. 실내 온보드 자율에 가장 잘 맞는다.
# ---------------------------------------------------------------------------


class OakDSource:
    """DepthAI SpatialDetectionNetwork 래퍼.

    이 카메라를 고르는 이유: 호스트(Pi)의 CPU/NPU를 거의 쓰지 않고
    ``(라벨, 확신도, X, Y, Z[mm])``가 바로 나온다. 스테레오 깊이가 실측이라
    단안 추정처럼 거리를 틀리지 않는다.

    주의: 이 클래스는 실기체 검증을 하지 않았다. 첫 비행 전에
    기체를 손에 들고 좌표축 방향(전/우/하)부터 확인할 것.
    """

    #: DepthAI 카메라 좌표계 → 기체 좌표계. X=우, Y=하, Z=전방.
    def __init__(self, blob_path: str, labels: list[str], confidence: float = 0.5) -> None:
        import depthai as dai  # 지연 임포트: mock 실행 시 설치 불필요

        self._dai = dai
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(300, 300)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        net = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        net.setBlobPath(blob_path)
        net.setConfidenceThreshold(confidence)
        net.setBoundingBoxScaleFactor(0.5)
        net.setDepthLowerThreshold(200)     # 20cm
        net.setDepthUpperThreshold(8000)    # 8m
        cam.preview.link(net.input)
        stereo.depth.link(net.inputDepth)

        out = pipeline.create(dai.node.XLinkOut)
        out.setStreamName("det")
        net.out.link(out.input)

        # 뎁스맵을 따로 뺀다. 이것이 전방 거리계 역할을 한다 —
        # 별도 ToF 센서를 달지 않는 이유다.
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        net.passthroughDepth.link(depth_out.input)

        self._device = dai.Device(pipeline)
        self._queue = self._device.getOutputQueue("det", maxSize=4, blocking=False)
        self._depth_queue = self._device.getOutputQueue("depth", maxSize=4, blocking=False)
        self._front_m: float | None = None
        self._blind = True
        self.labels = labels

    #: 뎁스가 유효하다고 볼 범위(mm). 밖의 값은 노이즈로 버린다.
    MIN_VALID_MM = 200
    MAX_VALID_MM = 8000
    #: 이보다 유효 픽셀이 적으면 '전방을 못 본다'로 판단한다.
    MIN_VALID_PIXELS = 400

    @classmethod
    def _forward_clearance(cls, depth_mm) -> tuple[float | None, bool]:
        """스테레오 뎁스맵에서 전방 최소 거리를 뽑는다.

        검출 결과(bounding box)가 아니라 뎁스 원본을 쓴다. 모델이 아무것도
        못 알아봐도 벽은 벽이기 때문이다.

        반환값은 ``(거리_m, 눈을 감았는가)``. 유효 픽셀이 부족하면 거리를
        지어내지 않고 blind 를 세운다 — 무늬 없는 흰 벽에 가까이 붙으면
        스테레오가 통째로 실패하는데, 그때가 바로 멈춰야 할 순간이다.
        """
        import numpy as np

        h, w = depth_mm.shape
        # 중앙 영역만 본다. 가장자리에는 프롭과 기체 프레임이 걸린다.
        band = depth_mm[int(h * 0.30):int(h * 0.70), int(w * 0.25):int(w * 0.75)]
        valid = band[(band >= cls.MIN_VALID_MM) & (band <= cls.MAX_VALID_MM)]

        if valid.size < cls.MIN_VALID_PIXELS:
            return None, True

        # 최솟값이 아니라 하위 5퍼센타일. 튀는 픽셀 하나에 기체가 멈추지 않게 한다.
        return float(np.percentile(valid, 5)) / 1000.0, False

    def read(self, telem: Telemetry | None) -> Perception | None:
        depth_packet = self._depth_queue.tryGet()
        if depth_packet is not None:
            self._front_m, self._blind = self._forward_clearance(
                depth_packet.getFrame()
            )

        packet = self._queue.tryGet()
        if packet is None:
            return None

        detections = []
        for d in packet.detections:
            name = self.labels[d.label] if d.label < len(self.labels) else str(d.label)
            detections.append(
                Detection(
                    label=name,
                    confidence=float(d.confidence),
                    # mm → m, 카메라축(X=우, Y=하, Z=전) → 기체축(x=전, y=우, z=하)
                    x_m=d.spatialCoordinates.z / 1000.0,
                    y_m=d.spatialCoordinates.x / 1000.0,
                    z_m=d.spatialCoordinates.y / 1000.0,
                    has_depth=True,
                )
            )
        # 좌/우는 None 으로 남긴다. OAK-D 는 전방만 본다 — 옆과 뒤에는
        # 눈이 없다. Policy 가 기수를 먼저 돌린 뒤에만 전진하는 이유다.
        return Perception(
            detections=tuple(detections),
            front_m=self._front_m,
            forward_blind=self._blind,
        )

    def close(self) -> None:
        self._device.close()


# ---------------------------------------------------------------------------
# ONNX / Hailo — 단안 카메라. 깊이는 추정이므로 신뢰도를 낮게 다뤄야 한다.
# ---------------------------------------------------------------------------


class OnnxSource:
    """Pi 카메라 + ONNX Runtime.

    단안이라 거리를 직접 못 잰다. 대상의 실제 높이를 안다고 가정하고
    바운딩박스 높이로 역산한다. 오차가 20~30%는 나므로
    ``has_depth=False``로 표시하고, 접근 정지는 반드시 ToF 거리계로 판단할 것.
    """

    def __init__(
        self,
        model_path: str,
        labels: list[str],
        focal_px: float = 500.0,
        assumed_height_m: float = 1.7,
        input_size: int = 640,
    ) -> None:
        import onnxruntime as ort  # 지연 임포트

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.labels = labels
        self.focal_px = focal_px
        self.assumed_height_m = assumed_height_m
        self.input_size = input_size
        self._camera = None

    def _frame(self):
        if self._camera is None:
            from picamera2 import Picamera2  # 지연 임포트

            self._camera = Picamera2()
            self._camera.configure(
                self._camera.create_preview_configuration(
                    main={"size": (self.input_size, self.input_size), "format": "RGB888"}
                )
            )
            self._camera.start()
        return self._camera.capture_array()

    def read(self, telem: Telemetry | None) -> Perception | None:
        import numpy as np

        frame = self._frame()
        blob = frame.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        raw = self._session.run(None, {self._input_name: blob})[0]

        detections = []
        for row in self._decode(raw):
            cls_id, conf, _x1, y1, _x2, y2 = row
            box_h = max(1.0, y2 - y1)
            depth = (self.assumed_height_m * self.focal_px) / box_h
            name = self.labels[int(cls_id)] if int(cls_id) < len(self.labels) else str(int(cls_id))
            detections.append(
                Detection(
                    label=name,
                    confidence=float(conf),
                    x_m=round(depth, 2),
                    y_m=0.0,
                    z_m=0.0,
                    has_depth=False,   # 추정값임을 명시한다
                )
            )
        return Perception(detections=tuple(detections))

    @staticmethod
    def _decode(raw):
        """모델 출력 후처리. 사용하는 모델의 출력 형식에 맞게 교체할 것."""
        for row in raw.reshape(-1, raw.shape[-1]):
            if row[4] >= 0.25:
                yield row[5], row[4], row[0], row[1], row[2], row[3]

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()


class WithRangefinders:
    """인식 백엔드에 좌·우·후방 ToF 값을 덧붙인다.

    카메라와 거리계는 서로 다른 하드웨어이고 고장도 따로 난다. 한쪽을
    다른 쪽 클래스 안에 넣는 대신 합성해서, 어느 쪽을 갈아끼워도 나머지가
    그대로 남게 한다. 전방(``front_m``)은 건드리지 않는다 — 그건 OAK-D
    스테레오 뎁스의 몫이다.
    """

    def __init__(self, inner: PerceptionSource, array: rangefinders.RangefinderArray) -> None:
        self.inner = inner
        self.array = array

    def read(self, telem: Telemetry | None) -> Perception | None:
        percep = self.inner.read(telem)
        if percep is None:
            return None
        clear = self.array.read()
        return dataclasses.replace(
            percep,
            left_m=clear.left_m,
            right_m=clear.right_m,
            back_m=clear.back_m,
        )

    def close(self) -> None:
        self.array.close()
        self.inner.close()


def build(cfg) -> PerceptionSource:
    """설정에 따라 백엔드를 고른다."""
    backend = cfg.runtime.perception_backend
    if backend == "mock":
        # 모의 방 시뮬레이션이 이미 좌·우 벽 거리를 계산하므로 덧씌우지 않는다.
        return MockSource(label=cfg.mission.target_label)
    if backend == "oakd":
        source = OakDSource(
            blob_path=cfg.runtime.onnx_model_path, labels=[cfg.mission.target_label]
        )
    elif backend == "onnx":
        source = OnnxSource(
            model_path=cfg.runtime.onnx_model_path, labels=[cfg.mission.target_label]
        )
    else:
        raise ValueError(f"알 수 없는 perception 백엔드: {backend}")

    array = rangefinders.build(cfg.runtime.rangefinder_backend)
    return WithRangefinders(source, array)
