"""실내 온보드 자율 드론 스택.

계층은 넷이고, 아래로 갈수록 빠르고 단순하며 신뢰도가 높다.

    policy   판단   0.5~10Hz   상태기계 (또는 학습 정책)
    guard    검증   루프마다   결정론적 if 문. 여기를 통과해야 나간다.
    link     통신   10~20Hz    MAVLink
    ArduPilot 제어  400Hz~1kHz 자세·위치. 이 코드는 여기 손대지 않는다.
"""

__all__ = ["config", "state", "perception", "guard", "policy", "link"]
