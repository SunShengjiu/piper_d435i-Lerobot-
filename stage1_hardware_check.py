#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段1：SDK 与硬件包接入验证（库导入 + 硬件初始化 + 硬件读取测试）。

本阶段范围（严格限定，不做超范围实现）
------------------------------------------------------------------
  [√] 导入 pyrealsense2 / lerobot / piper-sdk / opencv / numpy
  [√] 封装 PiperArm          ：CAN 连接、使能、读取 8 维状态、失能释放
  [√] 封装 D435iCamera       ：按序列号初始化 RGB 流、读帧、停止释放
  [√] 封装 DualD435iCamera   ：同时管理俯视 + 腕部两台 D435i
  [√] 主测试逻辑             ：硬件预检 -> 循环读取打印 -> Ctrl+C 优雅退出
  [×] 不实现 episode 采集 / 不写 LeRobotDataset / 不做多线程时序对齐
  [×] 不直接 import 或调用【树彩】目录内的任何驱动源码（仅参考其调用思路）

8 维机械臂状态定义（本阶段数据契约）
------------------------------------------------------------------
  PiPER 硬件实际为「6 个关节 + 1 个夹爪」，SDK 只有 joint_1..joint_6，不存在 joint_7。
  经确认为下列 8 维（全部取自 SDK 真实反馈，无虚构值）：
     索引 0..5 : joint1..joint6 关节角        [单位 rad]
     索引 6    : 夹爪位置 grippers_angle      [SDK 原始单位 0.001 mm]
     索引 7    : 夹爪力矩 grippers_effort     [SDK 原始单位 0.001 N*m]
  说明：夹爪位置/力矩保留 SDK 原始整数值，不在此阶段换算为毫米或牛顿米
        （换算需要夹爪标定，未标定时一律标注为无效，不伪造物理量）。

时间戳口径
------------------------------------------------------------------
  PiPER 的 CAN 反馈协议不带设备侧时间戳，SDK 暴露的 time_stamp 实为 python-can
  的报文接收时刻（主机时间）。因此本脚本只记录 host_recv_ns，不声称设备时间。
  RealSense 反馈的是设备时间戳(ms)，与主机时间不同域，读取时两者都保留。

依赖安装（全部 pip）
------------------------------------------------------------------
  注意1：lerobot 0.6.1 要求 Python >= 3.12；系统 /usr/bin/python3 为 3.10，装不了 lerobot。
  注意2：必须使用带 GUI 的 opencv-python。若环境中存在 opencv-python-headless，
        它会和 opencv-python 争抢同一个 cv2 模块，导致 cv2.imshow 报
        "The function is not implemented"，必须卸载 headless 版。

  # 1) 本机 uv 自带 3.12 解释器；用 uv 建环境可复用已有 wheel 缓存，无需重下 torch
  cd /home/robot/piper_d435
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python \
      pyrealsense2 piper-sdk opencv-python numpy "lerobot[dataset]==0.6.1"

  # 2) 关键一步：卸载 headless 版 opencv，确保 imshow 可用
  uv pip uninstall --python .venv/bin/python opencv-python-headless
  uv pip install  --python .venv/bin/python --reinstall opencv-python

  # 3) 自检（应打印 GUI: QT5 或 GTK3，而不是 GUI: NONE）
  .venv/bin/python -c "import cv2;print(cv2.__version__);print([l for l in cv2.getBuildInformation().splitlines() if 'GUI:' in l])"

  # 若不用 uv，也可用原生 venv + pip（但 torch 需重新下载，体积较大）：
  #   /home/robot/.local/bin/python3.12 -m venv .venv && source .venv/bin/activate
  #   pip install pyrealsense2 piper-sdk opencv-python numpy "lerobot[dataset]==0.6.1"
  #   pip uninstall -y opencv-python-headless && pip install --force-reinstall opencv-python

CAN 前置条件（本程序不激活/不修改 CAN，沿用既有配置）
------------------------------------------------------------------
  can0 需已 UP、bitrate 1000000。检查：  ip -details link show can0

运行说明（前置：机械臂上电并接入 can0，两台 D435i 已插好 USB3）
------------------------------------------------------------------
  cd /home/robot/piper_d435
  PY=.venv/bin/python            # 或先 source .venv/bin/activate 再直接用 python

  # 默认：使能机械臂 + 连接两台 D435i + 循环打印状态 + 预览画面
  $PY stage1_hardware_check.py

  # 只读模式（不使能机械臂，仅读取反馈）
  $PY stage1_hardware_check.py --no-enable

  # 干跑：只检查包导入与设备枚举，不连接硬件
  $PY stage1_hardware_check.py --dry-run

  # 仅列出已连接的 RealSense 设备序列号
  $PY stage1_hardware_check.py --list-devices

  # 无显示器环境（SSH/无 X）时关闭预览
  $PY stage1_hardware_check.py --no-preview

  # 退出：终端按 Ctrl+C；或预览窗口中按 q / ESC
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 依赖导入：本阶段的核心目标之一就是验证这些包能被正确导入
# ---------------------------------------------------------------------------
IMPORT_ERRORS: List[str] = []

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore
    IMPORT_ERRORS.append(f"numpy 导入失败: {exc}")

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    IMPORT_ERRORS.append(f"opencv-python(cv2) 导入失败: {exc}")

try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover
    rs = None  # type: ignore
    IMPORT_ERRORS.append(f"pyrealsense2 导入失败: {exc}")

try:
    import piper_sdk
    from piper_sdk import C_PiperInterface_V2
except Exception as exc:  # pragma: no cover
    piper_sdk = None  # type: ignore
    C_PiperInterface_V2 = None  # type: ignore
    IMPORT_ERRORS.append(f"piper-sdk 导入失败: {exc}")

# lerobot 本阶段只做导入验证与 LeRobotDataset 定义查看入口，不写任何数据集
try:
    import lerobot
except Exception as exc:  # pragma: no cover
    lerobot = None  # type: ignore
    IMPORT_ERRORS.append(f"lerobot 导入失败: {exc}")


# ---------------------------------------------------------------------------
# 全局默认配置
# ---------------------------------------------------------------------------
DEFAULT_CAN_INTERFACE = "can0"
DEFAULT_WRIST_SERIAL = "241122071942"        # 腕部相机
DEFAULT_THIRD_PERSON_SERIAL = "042222071742"  # 俯视（第三人称）相机
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 640, 480, 30

# 8 维状态向量各维名称（供打印与后续阶段复用）
STATE_DIM_NAMES: Tuple[str, ...] = (
    "joint1_rad", "joint2_rad", "joint3_rad", "joint4_rad",
    "joint5_rad", "joint6_rad",
    "gripper_pos_raw_0p001mm", "gripper_effort_raw_0p001Nm",
)

_MDEG_TO_RAD = 3.141592653589793 / 180.0 / 1000.0  # SDK 关节反馈 0.001 度 -> rad


def log(msg: str) -> None:
    """统一带时间戳的日志输出。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 一、依赖与版本检查
# ===========================================================================
def check_imports() -> bool:
    """导入检查：打印各包版本，并验证 LeRobotDataset 可导入（只导入，不使用）。"""
    log("=" * 68)
    log("步骤1  依赖包导入检查")
    log("=" * 68)
    ok = True

    def _v(mod: Any) -> str:
        try:
            import importlib.metadata as md
            return md.version(mod.__name__.split(".")[0])
        except Exception:
            return getattr(mod, "__version__", "未知版本")

    for name, mod in (("numpy", np), ("opencv(cv2)", cv2),
                      ("realsense(pyrealsense2)", rs), ("piper-sdk", piper_sdk)):
        if mod is None:
            log(f"  [失败] {name} 未导入")
            ok = False
        else:
            log(f"  [成功] {name:24s} 版本 {_v(mod)}")

    if lerobot is None:
        log("  [失败] lerobot 未导入")
        ok = False
    else:
        log(f"  [成功] {'lerobot':24s} 版本 {_v(lerobot)}")
        # 学习 LeRobotDataset 定义的入口：仅验证可导入，本阶段绝不写入数据集
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
            log("  [成功] lerobot.datasets.lerobot_dataset.LeRobotDataset 可导入（本阶段仅验证）")
        except Exception as exc:
            log(f"  [失败] LeRobotDataset 导入失败: {exc}")
            ok = False

    if C_PiperInterface_V2 is None:
        log("  [失败] piper_sdk.C_PiperInterface_V2 不可用")
        ok = False
    else:
        log("  [成功] piper_sdk.C_PiperInterface_V2 可导入")

    if IMPORT_ERRORS:
        log("  导入错误明细：")
        for e in IMPORT_ERRORS:
            log(f"    - {e}")

    log(f"依赖导入检查结果：{'全部通过' if ok else '存在失败'}")
    return ok


# ===========================================================================
# 二、Piper 机械臂封装
# ===========================================================================
@dataclass
class ArmState:
    """机械臂一帧状态快照（8 维 + 有效性标记）。

    joints_valid / gripper_valid 表示"是否真的收到了对应 CAN 反馈帧"。
    SDK 用 time_stamp=0 表示该路反馈从未收到过，因此未收到时数据不可信，
    一律置为无效并标注，绝不用全 0 冒充真实读数。
    """

    joints_rad: List[float]                 # 长度 6，单位 rad
    joints_raw_mdeg: List[int]              # 长度 6，SDK 原始 0.001 度
    gripper_pos_raw: Optional[int]          # SDK 原始 0.001 mm
    gripper_effort_raw: Optional[int]       # SDK 原始 0.001 N*m
    gripper_status_code: Optional[int]
    joints_valid: bool
    gripper_valid: bool
    host_recv_ns: int                       # 主机接收时刻（非设备时间）
    enable: Optional[List[bool]] = None

    def as_vector(self) -> List[float]:
        """返回 8 维状态向量；无效维度以 float('nan') 标记，不伪造数值。"""
        vec: List[float] = []
        for i, v in enumerate(self.joints_rad):
            vec.append(float(v) if self.joints_valid else float("nan"))
        vec.append(float(self.gripper_pos_raw) if (self.gripper_valid and self.gripper_pos_raw is not None) else float("nan"))
        vec.append(float(self.gripper_effort_raw) if (self.gripper_valid and self.gripper_effort_raw is not None) else float("nan"))
        return vec

    def pretty(self) -> str:
        """可读的一行打印文本。"""
        if self.joints_valid:
            j = " ".join(f"{v:+.4f}" for v in self.joints_rad)
            j_str = f"[{j}] rad"
        else:
            j_str = "[无关节反馈]"
        if self.gripper_valid:
            g_str = f"pos={self.gripper_pos_raw:6d}(0.001mm) eff={self.gripper_effort_raw:6d}(0.001N*m) status={self.gripper_status_code}"
        else:
            g_str = "[无夹爪反馈]"
        en = "-" if self.enable is None else ("已使能" if all(self.enable) else f"{self.enable}")
        return f"关节 {j_str}\n        夹爪 {g_str}\n        使能 {en}"


class PiperArm:
    """PiPER 机械臂 CAN 通信封装。

    仅使用官方 piper_sdk 的公开接口：
      ConnectPort / DisconnectPort / EnableArm / DisableArm /
      GetArmJointMsgs / GetArmGripperMsgs / GetArmEnableStatus / GetArmStatus

    参数:
        can_interface   : socketcan 接口名（默认 can0）
        enable          : 连接成功后是否使能机械臂（False 为只读模式）
        dh_is_offset    : piper_sdk DH 2° 偏置开关，需与固件代际对应
        logger_level    : piper_sdk 内部日志级别（40=ERROR，减少刷屏）
    """

    def __init__(
        self,
        can_interface: str = DEFAULT_CAN_INTERFACE,
        *,
        enable: bool = True,
        dh_is_offset: int = 0x01,
        logger_level: int = 40,
    ) -> None:
        self.can_interface = can_interface
        self.enable_requested = bool(enable)
        self.dh_is_offset = int(dh_is_offset)
        self.logger_level = int(logger_level)

        self._piper: Any = None
        self.versions: Dict[str, Any] = {}
        self.open_error: Optional[str] = None
        # 反馈帧统计（用于预检判断"是否真的收到 CAN 反馈"）
        self._counters = {"joint_frames": 0, "gripper_frames": 0, "read_errors": 0}
        self._last_joint_stamp: Any = None
        self._last_gripper_stamp: Any = None

    # ----------------------------------------------------------- 生命周期
    @property
    def is_connected(self) -> bool:
        return self._piper is not None

    def connect(self, timeout_s: float = 5.0) -> bool:
        """建立 CAN 连接并使能（可选），等待首帧关节反馈。返回是否成功。"""
        if C_PiperInterface_V2 is None:
            self.open_error = "piper_sdk 未导入，无法连接机械臂"
            return False
        try:
            # can_auto_init=True：由 SDK 打开 socketcan（沿用既有 can0 配置，rate 1M）
            # start_sdk_joint_limit / start_sdk_gripper_limit=False：不在连接期劫持限位逻辑
            self._piper = C_PiperInterface_V2(
                can_name=self.can_interface,
                judge_flag=True,
                can_auto_init=True,
                dh_is_offset=self.dh_is_offset,
                start_sdk_joint_limit=False,
                start_sdk_gripper_limit=False,
                logger_level=self.logger_level,
            )
        except Exception as exc:
            self.open_error = f"创建 CAN 接口失败({self.can_interface}): {exc}"
            self._piper = None
            return False

        try:
            # piper_init=False：不在连接时下发初始化/查询帧，避免任何非必要下发
            self._piper.ConnectPort(can_init=True, piper_init=False, start_thread=True)
        except Exception as exc:
            self.open_error = f"ConnectPort 失败: {exc}"
            self._piper = None
            return False

        self.versions = {
            "sdk_version": self._safe(lambda: str(self._piper.GetCurrentSDKVersion())),
            "interface_version": self._safe(lambda: str(self._piper.GetCurrentInterfaceVersion())),
            "protocol_version": self._safe(lambda: str(self._piper.GetCurrentProtocolVersion())),
            "can_interface": self.can_interface,
        }

        # 等待首帧真实关节反馈，确认 CAN 通信真的通了（而不只是打开了接口）
        if not self._wait_first_feedback(timeout_s):
            self.open_error = (
                f"{timeout_s:.0f}s 内未收到关节反馈帧。请检查机械臂是否上电、"
                f"{self.can_interface} 是否正确接线且 bitrate=1000000。"
            )
            self.close()
            return False

        if self.enable_requested:
            self._safe(lambda: self._piper.EnableArm(7))  # 7 = 全部电机（6 关节 + 夹爪）
            time.sleep(0.2)
        return True

    def _wait_first_feedback(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            st = self.read_state()
            if st.joints_valid:
                return True
            time.sleep(0.05)
        return False

    def _safe(self, fn) -> Any:
        try:
            return fn()
        except Exception:
            return None

    def close(self) -> None:
        """安全释放：先失能再断开 CAN。可重复调用。"""
        if self._piper is None:
            return
        if self.enable_requested:
            self._safe(lambda: self._piper.DisableArm(7))
            time.sleep(0.1)
        self._safe(lambda: self._piper.DisconnectPort())
        self._piper = None

    # ----------------------------------------------------------- 读取
    def read_state(self) -> ArmState:
        """读取一帧 8 维状态。无反馈时对应维度标记为无效。"""
        now_ns = time.time_ns()
        if self._piper is None:
            return ArmState([0.0] * 6, [0] * 6, None, None, None, False, False, now_ns)

        joints_rad = [0.0] * 6
        joints_raw = [0] * 6
        joints_valid = False
        try:
            jmsg = self._piper.GetArmJointMsgs()
            js = jmsg.joint_state
            # SDK 用 time_stamp=报文接收时间填该字段，默认 0 表示"从未收到过该反馈"
            stamp = getattr(jmsg, "time_stamp", None)
            joints_raw = [int(js.joint_1), int(js.joint_2), int(js.joint_3),
                          int(js.joint_4), int(js.joint_5), int(js.joint_6)]
            joints_rad = [v * _MDEG_TO_RAD for v in joints_raw]
            if stamp not in (None, 0):
                joints_valid = True
                if stamp != self._last_joint_stamp:
                    self._counters["joint_frames"] += 1
                    self._last_joint_stamp = stamp
        except Exception:
            self._counters["read_errors"] += 1

        gpos = geff = gstatus = None
        gripper_valid = False
        try:
            gmsg = self._piper.GetArmGripperMsgs()
            gs = gmsg.gripper_state
            stamp = getattr(gmsg, "time_stamp", None)
            gpos = int(gs.grippers_angle)
            geff = int(gs.grippers_effort)
            gstatus = int(gs.status_code)
            if stamp not in (None, 0):
                gripper_valid = True
                if stamp != self._last_gripper_stamp:
                    self._counters["gripper_frames"] += 1
                    self._last_gripper_stamp = stamp
        except Exception:
            self._counters["read_errors"] += 1

        enable = self._safe(lambda: [bool(v) for v in self._piper.GetArmEnableStatus()])

        return ArmState(
            joints_rad=joints_rad,
            joints_raw_mdeg=joints_raw,
            gripper_pos_raw=gpos,
            gripper_effort_raw=geff,
            gripper_status_code=gstatus,
            joints_valid=joints_valid,
            gripper_valid=gripper_valid,
            host_recv_ns=now_ns,
            enable=enable if isinstance(enable, list) else None,
        )

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._counters)


# ===========================================================================
# 三、单台 / 双台 D435i 相机封装
# ===========================================================================
class CameraSpecError(RuntimeError):
    """请求的流规格无法满足（不做静默降级）。"""


class D435iCamera:
    """Intel RealSense D435i 单目 RGB 流封装（按序列号绑定设备）。

    参数:
        serial        : 相机序列号（多相机场景必须指定，否则会互相抢占设备）
        width/height  : RGB 分辨率，默认 640x480
        fps           : RGB 帧率，默认 30
        warmup_frames : 启动后预热丢弃帧数（首帧常有曝光/白平衡未稳定）
        timeout_ms    : 等待单帧超时
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        name: str = "camera",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        warmup_frames: int = 30,
        timeout_ms: int = 5000,
    ) -> None:
        self.serial = serial
        self.name = name
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.warmup_frames = int(warmup_frames)
        self.timeout_ms = int(timeout_ms)

        self._pipeline: Any = None
        self._profile: Any = None
        self.firmware_version: Optional[str] = None
        self.open_error: Optional[str] = None
        self.counters = {"frames": 0, "timeouts": 0}

    # ----------------------------------------------------------- 设备枚举
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """列出当前系统上所有 RealSense 设备（含序列号/固件/USB 类型）。"""
        if rs is None:
            return []
        out: List[Dict[str, Any]] = []
        for dev in rs.context().query_devices():
            info: Dict[str, Any] = {}
            for key in ("name", "serial_number", "firmware_version", "usb_type_descriptor"):
                try:
                    value = dev.get_info(getattr(rs.camera_info, key))
                except Exception:
                    # 不能为了"看起来完整"而编造字段，读不到就明确标为未知
                    value = "N/A(读不到)"
                info[key] = value
            out.append(info)
        return out

    # ----------------------------------------------------------- 生命周期
    @property
    def is_open(self) -> bool:
        return self._pipeline is not None

    def start(self) -> None:
        """初始化 RGB 流并核对实际生效规格。失败抛异常，由上层统一处理。"""
        if rs is None:
            raise RuntimeError("pyrealsense2 未导入")
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("未发现任何 RealSense 设备")

        target = None
        for dev in devices:
            if self.serial is None or dev.get_info(rs.camera_info.serial_number) == self.serial:
                target = dev
                break
        if target is None:
            raise RuntimeError(
                f"未找到序列号为 {self.serial} 的相机；当前在线: "
                f"{[d.get_info(rs.camera_info.serial_number) for d in devices]}"
            )

        self.serial = target.get_info(rs.camera_info.serial_number)
        try:
            self.firmware_version = target.get_info(rs.camera_info.firmware_version)
        except Exception:
            self.firmware_version = "N/A(读不到)"

        cfg = rs.config()
        # 关键：把 pipeline 显式绑定到该序列号，否则多相机时后启动的会抢占同一台设备
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(cfg)

        # 核对实际生效 profile，Requested 与 Actual 不一致则报错（不静默降级）
        prof = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        actual = (prof.width(), prof.height(), prof.fps(), str(prof.format()).replace("format.", ""))
        expect = (self.width, self.height, self.fps, "bgr8")
        if actual != expect:
            self.stop()
            raise CameraSpecError(f"[{self.name}] 实际流 {actual} 与请求 {expect} 不一致")

        for _ in range(max(0, self.warmup_frames)):
            try:
                self._pipeline.wait_for_frames(self.timeout_ms)
            except Exception:
                break

    def read(self) -> Optional[Tuple[Any, float, int]]:
        """读取一帧 RGB。返回 (bgr 图像 ndarray, 设备时间戳 ms, 主机接收 ns)；超时返回 None。"""
        if self._pipeline is None:
            raise RuntimeError(f"[{self.name}] 相机尚未打开")
        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
        except Exception:
            self.counters["timeouts"] += 1
            return None
        host_ns = time.time_ns()
        color = frames.get_color_frame()
        if not color:
            return None
        img = np.asanyarray(color.get_data())
        self.counters["frames"] += 1
        return img, float(color.get_timestamp()), host_ns

    def stop(self) -> None:
        """停止并释放该相机（可重复调用）。"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._profile = None


class DualD435iCamera:
    """两台 D435i（俯视 + 腕部）统一管理。

    启动顺序与失败处理：任一相机启动失败时，回滚已启动的相机并抛异常，
    由主流程判定"硬件预检失败 -> 直接退出"。
    """

    def __init__(
        self,
        wrist_serial: str = DEFAULT_WRIST_SERIAL,
        third_person_serial: str = DEFAULT_THIRD_PERSON_SERIAL,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        warmup_frames: int = 30,
    ) -> None:
        self.wrist = D435iCamera(wrist_serial, name="wrist(腕部)",
                                 width=width, height=height, fps=fps, warmup_frames=warmup_frames)
        self.top = D435iCamera(third_person_serial, name="third_person(俯视)",
                               width=width, height=height, fps=fps, warmup_frames=warmup_frames)

    @property
    def is_open(self) -> bool:
        return self.wrist.is_open and self.top.is_open

    def start(self) -> None:
        """依次启动两台相机；任一失败则回滚并抛出异常。"""
        self.wrist.start()
        try:
            self.top.start()
        except Exception:
            self.wrist.stop()
            raise

    def read_pair(self) -> Tuple[Optional[Any], Optional[Any]]:
        """读取一对 RGB 图像，返回 (腕部图, 俯视图)。单路超时为 None。"""
        w = self.wrist.read()
        t = self.top.read()
        return (w[0] if w else None), (t[0] if t else None)

    def stop(self) -> None:
        """停止并释放两台相机。"""
        self.wrist.stop()
        self.top.stop()

    def diagnostics(self) -> Dict[str, Any]:
        return {"wrist": dict(self.wrist.counters), "third_person": dict(self.top.counters)}


# ===========================================================================
# 四、主测试逻辑
# ===========================================================================
def _render_preview(wrist_img: Any, top_img: Any) -> Any:
    """把两路画面横向拼接并加标签，便于单窗口预览。"""
    panels = []
    for title, img in (("wrist", wrist_img), ("third_person", top_img)):
        if img is None:
            panel = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
            cv2.putText(panel, f"{title}: no frame", (20, DEFAULT_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            panel = img.copy()
        cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        panels.append(panel)
    return np.hstack(panels)


def run(args: argparse.Namespace) -> int:
    """主流程：预检 -> 循环读取 -> 优雅退出。返回进程退出码。"""
    # ---------------- 步骤1：依赖导入检查 ----------------
    if not check_imports():
        log("依赖导入检查未通过，终止。请按文件头注释安装依赖后重试。")
        return 2

    # ---------------- 步骤2：RealSense 设备枚举 ----------------
    log("=" * 68)
    log("步骤2  硬件预检 - RealSense 设备枚举")
    log("=" * 68)
    devices = D435iCamera.list_devices()
    if not devices:
        log("  [失败] 未发现任何 RealSense 设备")
        return 1
    for d in devices:
        log(f"  - {d['name']}  serial={d['serial_number']}  fw={d['firmware_version']}  usb={d['usb_type_descriptor']}")
    log(f"  目标腕部相机 serial={args.wrist_serial}；目标俯视相机 serial={args.third_person_serial}")
    online = {d["serial_number"] for d in devices}
    missing = [s for s in (args.wrist_serial, args.third_person_serial) if s not in online]
    if missing:
        log(f"  [失败] 目标相机不在线: {missing}")
        return 1

    # ---------------- 步骤3：连接 Piper 机械臂 ----------------
    log("=" * 68)
    log("步骤3  硬件预检 - 连接 Piper 机械臂")
    log("=" * 68)
    arm = PiperArm(can_interface=args.can, enable=not args.no_enable)
    if not arm.connect(timeout_s=args.connect_timeout):
        log(f"  [失败] 机械臂连接失败：{arm.open_error}")
        log("任一硬件失败直接退出程序。")
        return 1
    log(f"  [成功] 机械臂已连接  interface={args.can}  sdk={arm.versions.get('sdk_version')}  "
        f"protocol={arm.versions.get('protocol_version')}")
    log(f"  [成功] {'已使能机械臂' if not args.no_enable else '只读模式（未使能）'}")

    # ---------------- 步骤4：启动两台 D435i ----------------
    log("=" * 68)
    log("步骤4  硬件预检 - 启动两台 D435i RGB 流")
    log("=" * 68)
    cams = DualD435iCamera(args.wrist_serial, args.third_person_serial,
                           width=args.width, height=args.height, fps=args.fps)
    try:
        cams.start()
    except Exception as exc:
        log(f"  [失败] 相机启动失败：{exc}")
        cams.stop()
        arm.close()
        log("任一硬件失败直接退出程序。")
        return 1
    log(f"  [成功] 腕部相机  serial={cams.wrist.serial}  fw={cams.wrist.firmware_version}  "
        f"{args.width}x{args.height}@{args.fps} bgr8")
    log(f"  [成功] 俯视相机  serial={cams.top.serial}   fw={cams.top.firmware_version}  "
        f"{args.width}x{args.height}@{args.fps} bgr8")
    log(f"  [成功] 8 维状态顺序: {list(STATE_DIM_NAMES)}")

    log("=" * 68)
    log("步骤5  循环读取（Ctrl+C 退出，或预览窗口按 q / ESC）")
    log("=" * 68)

    stop_flag = threading.Event()

    def _on_sigint(_signum, _frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _on_sigint)

    interval = max(0.01, 1.0 / max(0.1, args.rate))
    loop_count = 0
    t_start = time.monotonic()
    exit_code = 0

    try:
        while not stop_flag.is_set():
            t_loop = time.monotonic()

            # --- 机械臂 8 维状态 ---
            st = arm.read_state()
            vec = st.as_vector()
            txt = "[" + ", ".join(("nan" if v != v else f"{v:+.4f}") for v in vec) + "]"
            log(f"#{loop_count:05d} 8维状态 {txt}")
            log(f"         {st.pretty()}")

            # --- 两路相机画面 ---
            wrist_img, top_img = cams.read_pair()
            if wrist_img is not None:
                log(f"        相机 wrist {wrist_img.shape[1]}x{wrist_img.shape[0]}  "
                    f"top {'无帧' if top_img is None else f'{top_img.shape[1]}x{top_img.shape[0]}'}")

            # --- 预览 ---
            if not args.no_preview and cv2 is not None:
                cv2.imshow("piper_d435 stage1 preview  [wrist | third_person]", _render_preview(wrist_img, top_img))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log("检测到退出按键，准备关闭。")
                    break

            loop_count += 1
            if args.max_loops and loop_count >= args.max_loops:
                log(f"已达到 --max-loops={args.max_loops}，正常结束。")
                break

            dt = interval - (time.monotonic() - t_loop)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:  # 兜底：未走 signal 处理器的 Ctrl+C
        log("收到 Ctrl+C，准备关闭。")
    finally:
        elapsed = time.monotonic() - t_start
        log("=" * 68)
        log("步骤6  优雅关闭，释放硬件资源")
        log("=" * 68)
        cams.stop()
        log(f"  [完成] 两台相机已停止释放  {cams.diagnostics()}")
        arm.close()
        log(f"  [完成] 机械臂已失能并断开 CAN  {arm.diagnostics()}")
        if cv2 is not None:
            cv2.destroyAllWindows()
        if elapsed > 0:
            log(f"本次共循环 {loop_count} 次，耗时 {elapsed:.1f}s，"
                f"平均机械臂状态速率 {loop_count / elapsed:.1f} 次/秒")
        log("阶段1 硬件读取验证结束。")

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="阶段1：SDK 与硬件包接入验证（库导入 / 硬件初始化 / 硬件读取）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--can", default=DEFAULT_CAN_INTERFACE, help="CAN 接口名")
    p.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL, help="腕部相机序列号")
    p.add_argument("--third-person-serial", default=DEFAULT_THIRD_PERSON_SERIAL, help="俯视相机序列号")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="RGB 宽度")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="RGB 高度")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="RGB 帧率")
    p.add_argument("--rate", type=float, default=10.0, help="主循环读取频率 Hz")
    p.add_argument("--no-enable", action="store_true", help="只读模式，不使能机械臂")
    p.add_argument("--no-preview", action="store_true", help="不显示 cv2 预览窗口")
    p.add_argument("--max-loops", type=int, default=0, help="循环次数上限，0 表示无限")
    p.add_argument("--connect-timeout", type=float, default=5.0, help="机械臂连接超时秒数")
    p.add_argument("--dry-run", action="store_true", help="只做依赖导入与设备枚举，不连接硬件")
    p.add_argument("--list-devices", action="store_true", help="仅列出 RealSense 设备")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list_devices:
        for d in D435iCamera.list_devices():
            print(d)
        return 0

    if args.dry_run:
        ok = check_imports()
        print("\nRealSense 设备:")
        for d in D435iCamera.list_devices():
            print(" ", d)
        return 0 if ok else 2

    return run(args)


if __name__ == "__main__":
    sys.exit(main())