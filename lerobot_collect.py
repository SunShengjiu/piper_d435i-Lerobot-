#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper 机械臂 + 双 D435i —— LeRobot 拖拽示教数据采集 CLI

用途
    阶段2 交付物：一个可直接运行的采集入口。完成硬件自检 -> 自动回位/就位 ->
    等空格 -> 拖拽示教同时录制双路 RGB + 关节状态 -> 写入 LeRobotDataset(use_videos=True)
    -> 三次空格后复位回位 -> 循环下一个 episode。Ctrl+C 安全退出。

业务流程（与需求条目一一对应，代码中标注【步骤N】）

    ⚠ 空格语义速查 —— 每一轮 episode 固定按 3 次空格，程序"接管运动"只发生在第 3 次：
        【第 1 次空格】开始录制（程序只读反馈，绝不接管运动）
        【第 2 次空格】停止录制并保存 episode（程序仍然不接管运动，人工拖动机械臂）
        【第 3 次空格】程序开始接管：【按下即直接下电复位，无二次确认】
                       （示教模式 0x02 无法用指令切出，下电是唯一恢复控制的方式）
                       之后切 CAN 模式 + 回 HOME_BASE + 到 START_POSE

    【步骤1】CLI 启动
            1a 硬件初始化 + 全设备状态自检（只读：不创建数据集、不下发任何指令）
            1b 驱动机械臂回【复位基准位姿】HOME_BASE -> 停留 1s
               -> 平滑运动到【采集初始位姿】START_POSE
    【步骤2】进入空闲等待：程序【不下发任何指令、不录制、不接管运动】，等待第 1 次空格
    【步骤3】第 1 次空格 -> 此时才创建/校验 LeRobotDataset，正式开始 episode 录制
    【步骤4】操作员按机械臂本体绿色按键进入零重力拖拽模式，手动拖动示教；
            程序同步记录两路相机 RGB + 机械臂关节状态 + 夹爪 + 末端位姿，写入缓存
            ⚠ 本阶段【程序绝不向机械臂下发任何运动指令】（只读反馈）
    【步骤5】第 2 次空格 -> 仅【停止录制】；episode 帧交接给后台保存队列
            （常驻 worker 线程串行编码），与下一轮采集【真正并行、互不阻塞】
            ⚠ 此时程序【仍然不接管机械臂运动】，由操作员手动把机械臂拖到安全位置并托稳
    【步骤6】第 3 次空格 -> 程序【开始接管，立即执行】：强制下电复位(ResetPiper)
            （示教模式 0x02 无法用指令切出，下电是唯一恢复控制的方式；
              按下即执行，无二次确认，务必先托稳）-> 使能 -> 切 CAN 控制模式
            -> 回 HOME_BASE -> 停 1s -> 平滑到 START_POSE
            （后台保存全程并行；Ctrl+C 退出时会先等保存队列排空再 finalize）
    【步骤7】回到空闲等待，循环下一轮
    【步骤8】Ctrl+C -> 停线程 -> 关相机 -> finalize 数据集 -> 断开 CAN，释放资源

数据契约（已与需求方确认）
    observation.image.top      俯视 D435i    RGB 1280x720 uint8  -> H.264 视频编码
    observation.image.wrist    腕部 D435i    RGB 1280x720 uint8  -> H.264 视频编码
    observation.depth.top      俯视 D435i    深度 uint16 mm 1280x720
                               -> 12-bit log 量化 + HEVC(gray12le, lossless) 视频编码，
                                  加载时原生 LeRobotDataset 自动反量化为 float32 米制
    observation.depth.wrist    腕部 D435i    同上
    observation.state          7 维 float32
                               [j1..j6 rad, 夹爪位置 raw(0.001mm)]
    observation.gripper_effort 1 维 float32  [夹爪力矩 raw(0.001N·m)]
    observation.ee_pose        6 维 float32  [x,y,z 单位 m, rx,ry,rz 单位 rad]
    action                     7 维 float32  示教采集：action = 当前 observation.state
    next.done                  1 维 bool     LeRobot v3 强制字段：每 episode 最后一帧
                               True，其余 False（保存前统一回写 buffer 末帧）

    深度量化说明（与"uint8 方案A"的差异）：
        lerobot 0.6.1 官方深度路径为 12-bit log 量化（QMAX=4095 写死，不可配 8-bit），
        且 h264 不支持 12-bit 像素格式 —— uint8+H.264 深度视频无法走官方反量化路径。
        为满足"原生 LeRobotDataset 直接加载、反量化回米"这一硬约束，深度采用官方
        DepthEncoderConfig 默认（HEVC Main12 + gray12le + lossless + log 量化），
        量化精度（4096 级）优于 uint8（256 级）。RGB 已按要求全部切换 H.264。

    说明：夹爪位置/力矩没有标定，因此保留 SDK 原始整数量纲并在字段名里写明单位，
          不做 mm / N·m 的物理换算，避免伪造未标定数值。

安全设计（重要）
    1. ResetPiper() 会【立即失电落下】，程序在执行前必须获得人工确认（按 y）。
    2. 退出时【不做失能】(不调用 DisableArm)，避免机械臂因失电跌落；
       需要下电请人工按急停/断电。
    3. 一旦进入示教/拖拽阶段，程序不再下发任何 CAN 运动指令，避免与人工拖拽对抗。
    4. 到达判定：关节误差 < ARRIVE_TOL_RAD 才认为到位，超时或使能丢失即报错停机。

数据集目录策略（自动续录）
    目标目录 <dataset_root>/<repo_id> 的三种情况：
    1. 不存在 / 为空            -> 新建 LeRobotDataset
    2. 存在且 total_episodes==0  -> 上次失败的空数据集，自动清理重建
    3. 存在且有已有 episode      -> 【自动续录】：校验 fps 与 features 一致后，
       以可写模式(LeRobotDataset.resume)加载，新 episode 追加到同一目录，
       episode 编号接续；fps 或字段不一致则报错拒绝，防止写坏已有数据。

依赖
    pyrealsense2 / piper-sdk / lerobot[dataset] / opencv-python（--preview 用）/ numpy
    全部 pip 安装，直接 import。

运行前置
    - can0 已 UP 且 bitrate 1000000（本程序不修改 CAN 配置）
    - 机械臂上电、两台 D435i 插好 USB3、无其他程序占用 /dev/video*

运行示例
    .venv/bin/python lerobot_collect.py --check-only          # 只做硬件自检，不运动
    .venv/bin/python lerobot_collect.py                       # 完整采集流程
    .venv/bin/python lerobot_collect.py --preview             # 附带预览：单窗口 RGB+深度伪彩
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import traceback
import tty
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 依赖导入
# ---------------------------------------------------------------------------
try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover
    sys.exit(f"[FATAL] pyrealsense2 导入失败: {exc}\n请执行: uv pip install pyrealsense2")

try:
    from piper_sdk import C_PiperInterface_V2
except Exception as exc:  # pragma: no cover
    sys.exit(f"[FATAL] piper-sdk 导入失败: {exc}\n请执行: uv pip install piper-sdk")

# ⚠ cv2 的导入顺序至关重要：必须放在 lerobot(torch) 之前！
# 实测复现：torch 先加载、进程内已有 CAN/相机多线程时，首次 cv2.imshow 会触发
# OpenMP/Qt 初始化死锁 —— 表现为 --preview 下程序卡死且 Ctrl+C 无效；
# cv2 抢在 torch 之前导入则完全正常（已实测验证）。
# 未安装 GUI 版 opencv 时仅 --preview 不可用，不影响采集。
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

# 终端净化(1/2)：屏蔽 datasets.map 进度条(Map: 100%|...)与 Qt 字体警告(QFontDatabase...)。
# ⚠ 必须放在 lerobot/cv2(Qt) 导入【之前】—— tqdm 在模块导入时固化 TQDM_* 环境变量，
# Qt 在首次初始化时读取 QT_LOGGING_RULES
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.feature_utils import features_equal_for_merge
    from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
    import pyarrow.parquet as pq  # lerobot 硬依赖；仅用于续录前 footer 完整性预检
except Exception as exc:  # pragma: no cover
    sys.exit(f"[FATAL] lerobot 导入失败: {exc}\n请执行: uv pip install 'lerobot[dataset]==0.6.1'")

# 终端净化(2/2)：放在 lerobot 导入【之后】，覆盖库导入期间的任何 logging 配置。
# 目标：终端只显示带时间戳的程序日志（log/warn/step），屏蔽库噪音 ——
#   - root logger 提到 ERROR 级：屏蔽 WARNING（如 torchcodec 回退提示），保留真正的 ERROR
#   - ffmpeg 日志提到 ERROR 级：屏蔽 libx264/x265 编码统计与 "encoded N frames" 总结行，保留编码错误
logging.getLogger().setLevel(logging.ERROR)
# datasets 的 tqdm 包装在 HF_DATASETS_DISABLE_PROGRESS_BARS=1 下会发 UserWarning（stderr），
# 属预期的进度条禁用提示，过滤掉
warnings.filterwarnings("ignore", message="Cannot enable progress bars.*")
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:  # pragma: no cover
    pass


# ===========================================================================
# 一、全局固定配置区（所有可调参数集中在这里，方便后续修改）
# ===========================================================================

# ---- 两个固定位姿，单位 rad -------------------------------------------------
HOME_BASE: List[float] = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0]           # 复位基准位姿
START_POSE: List[float] = [1.57, 0.8987, -1.2139, 0.0, 1.1546, 0.0]  # 采集初始位姿

# ---- 关节软限位（rad，取自 piper_sdk JointCtrl 官方表格）---------------------
JOINT_LIMITS: List[Tuple[float, float]] = [
    (-2.6179, 2.6179),      # joint1
    (0.0, 3.14),            # joint2
    (-2.967, 0.0),          # joint3
    (-1.745, 1.745),        # joint4
    (-1.22, 1.22),          # joint5
    (-2.09439, 2.09439),    # joint6
]

# ---- 运动控制参数 -----------------------------------------------------------
MOVE_SPEED_PCT = 20         # MotionCtrl_2 运动速度百分比 0~100
ARRIVE_TOL_RAD = 0.02       # 到位判定阈值：每关节误差绝对值 < 该值(rad) 视为到位
ARRIVE_TIMEOUT_S = 30.0     # 单次运动超时(s)
CTRL_HZ_MOVE = 200          # 运动时指令下发频率(Hz)
CTRL_HZ_HOLD = 20           # 空闲保持(心跳保活)频率(Hz)
HOME_HOLD_S = 1.0           # 在 HOME_BASE 停留时间(s)
MODE_SWITCH_TIMEOUT_S = 15.0  # 切换到 CAN 指令控制模式的超时(s)

# 空闲(按空格前)是否下发保持指令。
#   False(默认)：空闲期间【完全不下发任何指令】，做到"按空格前程序绝不接管运动"，
#                与需求一致；代价是空闲超过约 60s 机械臂可能因无报文而掉使能/掉模式，
#                此时第 3 次空格执行复位前会自动重新切 CAN 模式（必要时提示下电复位）。
#   True：空闲持续下发保持指令保活，但机械臂会一直处于被指令控制的状态。
# 需要长时间空转保活的场景可用命令行 --idle-heartbeat 打开。
HEARTBEAT_IDLE = False

# ---- 单位换算 ---------------------------------------------------------------
RAD_TO_SDK = 57295.7795             # rad -> SDK 整数(0.001度)
SDK_TO_RAD = 1.0 / RAD_TO_SDK       # SDK 整数(0.001度) -> rad
END_POSE_XYZ_TO_M = 1e-6            # 末端位置反馈 0.001mm -> m
END_POSE_RPY_TO_RAD = math.pi / 180.0 / 1000.0  # 末端姿态反馈 0.001度 -> rad

# ---- CAN --------------------------------------------------------------------
DEFAULT_CAN = "can0"

# ---- 相机（固定参数，两台仅 white_balance 不同）-----------------------------
CAM_TOP_SERIAL = "042222071742"      # 俯视（第三人称）D435i
CAM_WRIST_SERIAL = "241122071942"    # 腕部 D435i
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
WARMUP_FRAMES = 30                   # 启动后丢弃的预热帧数
FRAME_TIMEOUT_MS = 5000              # wait_for_frames 超时(ms)

# ---- 预览（--preview）：单窗口同时展示两路 RGB 与深度伪彩 ---------------------
PREVIEW_WINDOW = "piper-collect:preview (RGB + depth)"
PREVIEW_TILE_W, PREVIEW_TILE_H = 640, 360   # 每块瓦片尺寸 -> 整窗 1280x720
PREVIEW_DEPTH_SCALE = 0.06           # 深度伪彩映射系数：65535*0.06≈3.9m 满量程（仅预览展示用）

COLOR_OPTIONS_COMMON: Dict[str, float] = {
    "enable_auto_exposure": 0.0,
    "exposure": 500.0,
    "gain": 0.0,
    "enable_auto_white_balance": 0.0,
    "brightness": 0.0,
    "contrast": 50.0,
    "saturation": 64.0,
    "sharpness": 50.0,
    "gamma": 300.0,
    "hue": 0.0,
    "backlight_compensation": 0.0,
    "power_line_frequency": 3.0,
    "auto_exposure_priority": 0.0,
}
COLOR_WHITE_BALANCE: Dict[str, float] = {"top": 6200.0, "wrist": 6150.0}

# ---- 数据集 -----------------------------------------------------------------
DEFAULT_REPO_ID = "local/piper_d435_teach"
DEFAULT_DATASET_ROOT = "./lerobot_data"
DEFAULT_TASK = "piper drag-teach pick and place"
RECORD_FPS = 30                     # 录制频率 & 数据集 fps

# ---- 数据集字段名（与上面的数据契约严格对应）--------------------------------
STATE_NAMES: Tuple[str, ...] = (
    "joint1_rad", "joint2_rad", "joint3_rad", "joint4_rad", "joint5_rad", "joint6_rad",
    "gripper_pos_raw_0p001mm",
)
EFFORT_NAMES: Tuple[str, ...] = ("gripper_effort_raw_0p001Nm",)
EE_POSE_NAMES: Tuple[str, ...] = (
    "ee_x_m", "ee_y_m", "ee_z_m", "ee_rx_rad", "ee_ry_rad", "ee_rz_rad",
)
ACTION_NAMES: Tuple[str, ...] = STATE_NAMES

IMG_TOP_KEY = "observation.image.top"
IMG_WRIST_KEY = "observation.image.wrist"
DEPTH_TOP_KEY = "observation.depth.top"
DEPTH_WRIST_KEY = "observation.depth.wrist"
STATE_KEY = "observation.state"
EFFORT_KEY = "observation.gripper_effort"
EE_POSE_KEY = "observation.ee_pose"
ACTION_KEY = "action"
NEXT_DONE_KEY = "next.done"

_VERBOSE = True


def log(msg: str) -> None:
    """统一带时间戳的日志输出。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step(msg: str) -> None:
    """业务步骤日志，前缀醒目，便于终端跟踪状态机。"""
    print(f"\n[{time.strftime('%H:%M:%S')}] ===== {msg} =====", flush=True)


def warn(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [警告] {msg}", flush=True)


# ===========================================================================
# 二、机械臂封装
# ===========================================================================
@dataclass
class ArmSnapshot:
    """机械臂一帧状态快照（全部来自 SDK 真实反馈）。"""

    joints_rad: List[float]                 # 6 维，单位 rad
    joints_valid: bool                      # 是否收到过关节反馈
    gripper_pos_raw: Optional[int]          # 0.001 mm
    gripper_effort_raw: Optional[int]       # 0.001 N·m
    gripper_valid: bool
    ee_pose_si: Optional[List[float]]       # [x,y,z m, rx,ry,rz rad]
    ctrl_mode: int
    enable: List[bool]
    host_recv_ns: int

    def state_vector(self) -> Optional[np.ndarray]:
        """7 维 observation.state：[j1..j6 rad, 夹爪位置 raw]。数据无效返回 None。"""
        if not self.joints_valid or not self.gripper_valid or self.gripper_pos_raw is None:
            return None
        return np.array([*self.joints_rad, float(self.gripper_pos_raw)], dtype=np.float32)

    def effort_vector(self) -> Optional[np.ndarray]:
        """1 维 observation.gripper_effort：[夹爪力矩 raw]。"""
        if not self.gripper_valid or self.gripper_effort_raw is None:
            return None
        return np.array([float(self.gripper_effort_raw)], dtype=np.float32)

    def pretty(self) -> str:
        if self.joints_valid:
            joints = " ".join(f"{v:+.4f}" for v in self.joints_rad)
        else:
            joints = "[无关节反馈]"
        g = (f"pos={self.gripper_pos_raw}(0.001mm) eff={self.gripper_effort_raw}(0.001N*m)"
             if self.gripper_valid else "[无夹爪反馈]")
        en = "全部使能" if (self.enable and all(self.enable)) else f"使能异常{self.enable}"
        return f"ctrl_mode=0x{self.ctrl_mode:02X} {en}\n         关节(rad): [{joints}]\n         夹爪: {g}"


class PiperArm:
    """PiPER 机械臂 CAN 通信封装（仅使用 piper_sdk 官方公开接口）。"""

    def __init__(self, can_name: str) -> None:
        self.can_name = can_name
        self.piper: Optional[Any] = None
        self.connected = False

    # ---------------- 连接 / 断开 ----------------
    def connect(self) -> None:
        self.piper = C_PiperInterface_V2(can_name=self.can_name)
        # can_init=True: 复用已有 can0（不修改 CAN 配置）；piper_init=False: 不做零点/初始化写入
        self.piper.ConnectPort(can_init=True, piper_init=False, start_thread=True)
        self.connected = True
        log(f"机械臂 CAN 已连接: {self.can_name}")

    def disconnect(self) -> None:
        """断开 CAN。注意：不断电、不失能，机械臂保持最后指令姿态。"""
        if self.piper is not None and self.connected:
            try:
                self.piper.DisconnectPort()
            except Exception as exc:
                warn(f"DisconnectPort 异常: {exc}")
            self.connected = False
            log("机械臂 CAN 已断开（未做失能，机械臂保持当前姿态）")

    # ---------------- 状态读取 ----------------
    def ctrl_mode(self) -> int:
        return int(self.piper.GetArmStatus().arm_status.ctrl_mode)

    def read(self) -> ArmSnapshot:
        """读取一帧完整状态。SDK 用 time_stamp==0 表示该路反馈从未收到，此时标记为无效。"""
        joints_raw, joints_valid = self._read_joints()
        gripper_pos, gripper_eff, gripper_valid = self._read_gripper()
        ee_pose = self._read_end_pose()
        return ArmSnapshot(
            joints_rad=[float(v) for v in joints_raw] if joints_valid else [float("nan")] * 6,
            joints_valid=joints_valid,
            gripper_pos_raw=gripper_pos,
            gripper_effort_raw=gripper_eff,
            gripper_valid=gripper_valid,
            ee_pose_si=ee_pose,
            ctrl_mode=self.ctrl_mode(),
            enable=self.enable_state(),
            host_recv_ns=time.time_ns(),
        )

    def _read_joints(self) -> Tuple[List[float], bool]:
        msgs = self.piper.GetArmJointMsgs()
        if getattr(msgs, "time_stamp", 0) == 0:
            return [0.0] * 6, False
        js = msgs.joint_state
        raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        return [int(v) * SDK_TO_RAD for v in raw], True

    def _read_gripper(self) -> Tuple[Optional[int], Optional[int], bool]:
        msgs = self.piper.GetArmGripperMsgs()
        if getattr(msgs, "time_stamp", 0) == 0:
            return None, None, False
        gs = msgs.gripper_state
        return int(gs.grippers_angle), int(gs.grippers_effort), True

    def _read_end_pose(self) -> Optional[List[float]]:
        """末端位姿 -> SI 单位 [x,y,z (m), rx,ry,rz (rad)]。"""
        msgs = self.piper.GetArmEndPoseMsgs()
        if getattr(msgs, "time_stamp", 0) == 0:
            return None
        ep = msgs.end_pose
        return [
            float(ep.X_axis) * END_POSE_XYZ_TO_M,
            float(ep.Y_axis) * END_POSE_XYZ_TO_M,
            float(ep.Z_axis) * END_POSE_XYZ_TO_M,
            float(ep.RX_axis) * END_POSE_RPY_TO_RAD,
            float(ep.RY_axis) * END_POSE_RPY_TO_RAD,
            float(ep.RZ_axis) * END_POSE_RPY_TO_RAD,
        ]

    # ---------------- 指令下发 ----------------
    def send_motion_ctrl(self, ctrl_mode: int = 0x01, move_mode: int = 0x01, speed: int = MOVE_SPEED_PCT) -> None:
        """0x151：ctrl_mode=0x01(CAN指令控制) / move_mode=0x01(MOVE J 关节插值) / 速度百分比。"""
        self.piper.MotionCtrl_2(ctrl_mode, move_mode, speed, 0x00)

    def send_joint_ctrl(self, joints_rad: List[float]) -> None:
        """0x155~0x157：关节角指令，rad -> SDK 整数(0.001度)。"""
        self.piper.JointCtrl(*[int(round(v * RAD_TO_SDK)) for v in joints_rad])

    def enable(self) -> None:
        """使能 6 关节 + 夹爪。"""
        self.piper.EnableArm(7)

    def reset_piper(self) -> None:
        """0x150 强制下电复位。⚠ 机械臂会立即失电落下，调用前必须有人工确认。"""
        self.piper.ResetPiper()

    def enable_state(self) -> List[bool]:
        """读取 6 个电机的使能状态（走低速信息反馈帧）。"""
        try:
            return [bool(v) for v in self.piper.GetArmEnableStatus()]
        except Exception:
            return []

    def arm_status_msg(self) -> Any:
        """读取 0x2A1 机械臂状态（含状态码 / 模式反馈 / 运动状态 / 关节超限与通信故障位）。

        运动指令被静默忽略时，真正的原因（超限 / 碰撞 / 抱闸未开 / 通信异常）
        都在这个帧里，必须读出来才能定位问题。
        """
        try:
            return self.piper.GetArmStatus().arm_status
        except Exception:
            return None

    def arm_status_text(self) -> str:
        """把 0x2A1 状态整理成单行文本，便于打进运动诊断日志。"""
        st = self.arm_status_msg()
        if st is None:
            return "arm_status=[读取失败]"
        extra = ""
        try:
            err = st.err_status
            limit = [i for i in range(1, 7) if getattr(err, f"joint_{i}_angle_limit")]
            comm = [i for i in range(1, 7) if getattr(err, f"communication_status_joint_{i}")]
            if limit:
                extra += f" 关节角度超限={limit}"
            if comm:
                extra += f" 关节通信异常={comm}"
        except Exception:
            pass
        return (f"arm_status={st.arm_status} mode_feed={st.mode_feed} "
                f"teach_status={st.teach_status} motion_status={st.motion_status}{extra}")

    def blocking_arm_status(self) -> Optional[str]:
        """返回会【阻止运动】的 arm_status 描述；正常/可忽略的状态返回 None。

        0x04 目标角度超限、0x07 碰撞、0x06 抱闸未打开等状态下，
        重新切 CAN 模式是无用的，必须直接把原因报出来。
        """
        st = self.arm_status_msg()
        if st is None:
            return None
        name = str(st.arm_status)
        blocking = ("TARGET_POS_EXCEEDS_LIMIT", "COLLISION_OCCURRED", "EMERGENCY_STOP",
                    "JOINT_BRAKE_NOT_RELEASED", "JOINT_COMMUNICATION_ERR",
                    "JOINT_STATUS_ERR", "NO_SOLUTION", "SINGULARITY_POINT")
        if any(b in name for b in blocking):
            return f"{name}（0x2A1 反馈：{self.arm_status_text()}）"
        return None

    def status_ok(self) -> Tuple[bool, str]:
        """运动过程中的安全校验：模式须为 CAN 指令控制，且所有电机处于使能状态。"""
        mode = self.ctrl_mode()
        if mode != 0x01:
            return False, f"ctrl_mode=0x{mode:02X} 不是 CAN 指令控制模式(0x01)"
        enable = self.enable_state()
        if not enable or not all(enable):
            return False, f"电机未全部使能: {enable}"
        return True, "ok"


class ArmCommander(threading.Thread):
    """按固定频率持续下发 0x151 + 0x155~0x157 的指令线程。

    - 运动模式：以 CTRL_HZ_MOVE(200Hz) 连续下发目标关节角；
      Piper 有超时保护，只发一次会被中断，必须连续下发。
    - 保持模式：以 CTRL_HZ_HOLD(20Hz) 下发固定目标，用于空闲心跳保活，
      避免长时间无 CAN 报文导致 ctrl_mode 掉 0 / 使能丢失。
    - 暂停模式：一个字节都不发。拖拽示教期间必须处于该模式。
    """

    def __init__(self, arm: PiperArm, speed: int = MOVE_SPEED_PCT) -> None:
        super().__init__(daemon=True, name="ArmCommander")
        self.arm = arm
        self.speed = speed
        self._lock = threading.Lock()
        self._target: Optional[List[float]] = None
        self._rate = CTRL_HZ_HOLD
        self._paused = True
        self._stop = threading.Event()
        self._sent = 0

    # ---- 外部控制接口 ----
    def start_move(self, target: List[float]) -> None:
        with self._lock:
            self._target = list(target)
            self._rate = CTRL_HZ_MOVE
            self._paused = False

    def start_hold(self, pose: List[float]) -> None:
        with self._lock:
            self._target = list(pose)
            self._rate = CTRL_HZ_HOLD
            self._paused = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def sent_count(self) -> int:
        """已成功下发的指令组数，用于确认指令线程确实在发报文。"""
        return self._sent

    def stop(self) -> None:
        self.pause()
        self._stop.set()

    # ---- 线程体 ----
    def run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                paused, target, rate = self._paused, self._target, self._rate
            if paused or target is None:
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            try:
                self.arm.send_motion_ctrl(0x01, 0x01, self.speed)
                self.arm.send_joint_ctrl(target)
                self._sent += 1
            except Exception as exc:
                warn(f"指令下发异常: {exc}")
            next_t += 1.0 / rate
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()   # 落后则重新对齐，避免时间堆积


def validate_joints(joints_rad: List[float], label: str) -> None:
    """下发前按硬件限位校验，越界直接报错，而不是让 SDK 静默截断。"""
    if len(joints_rad) != 6:
        raise ValueError(f"{label}: 关节数必须为 6，实际 {len(joints_rad)}")
    bad = []
    for i, (v, (lo, hi)) in enumerate(zip(joints_rad, JOINT_LIMITS), start=1):
        if not (lo <= v <= hi):
            bad.append(f"j{i}={v:+.4f} 超出 [{lo}, {hi}]")
    if bad:
        raise ValueError(f"{label} 超出关节限位: " + "; ".join(bad))


# ===========================================================================
# 三、相机封装
# ===========================================================================
class CameraStream:
    """单台 D435i：固定参数开启 RGB 流，后台线程持续抓帧，主线程随时取最新帧。"""

    def __init__(self, name: str, serial: str, white_balance: float,
                 enable_depth: bool = False) -> None:
        self.name = name
        self.serial = serial
        self.white_balance = white_balance
        self.enable_depth = enable_depth   # --preview 时才开深度流（仅预览展示，不入数据集）
        self.pipeline: Optional[Any] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_no = -1
        self._ts_ns = 0
        self._latest_depth: Optional[np.ndarray] = None   # HWC uint16 (mm)
        self._depth_no = -1
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.read_errors = 0

    # ---- 设备枚举 ----
    @staticmethod
    def list_serials() -> List[str]:
        return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]

    # ---- 生命周期 ----
    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_device(self.serial)   # 必须显式绑定序列号，否则 rs 会自行挑设备
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        if self.enable_depth:
            # 深度流仅在 --preview 时开启，供预览伪彩展示；数据集录制仍只存 RGB
            cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(cfg)

        # 实际协商结果校验，防止静默降级
        sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        got = (sp.width(), sp.height(), sp.fps(), str(sp.format()).split(".")[-1])
        want = (CAM_W, CAM_H, CAM_FPS, "bgr8")
        log(f"  相机 {self.name:<5} 序列号={self.serial} color 流协商 {got[0]}x{got[1]}@{got[2]} {got[3]}"
            f"  [{'OK' if got == want else '!! 与请求不符 ' + str(want)}]")

        self._apply_color_options(profile)

        # 预热：丢弃前 WARMUP_FRAMES 帧，让自动曝光/白平衡收敛（已关自动，仍需要稳定帧）
        for _ in range(WARMUP_FRAMES):
            try:
                self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError:
                break

        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"Cam-{self.name}")
        self._thread.start()

    def _apply_color_options(self, profile: Any) -> None:
        """写入固定彩色传感器参数，并回读校验。"""
        dev = profile.get_device()
        sensor = dev.first_color_sensor() if dev.sensors else None
        if sensor is None:
            warn(f"相机 {self.name}: 未找到彩色传感器，跳过参数写入")
            return
        opts = dict(COLOR_OPTIONS_COMMON)
        opts["white_balance"] = self.white_balance
        ok = warn_n = bad = 0
        for name, want in opts.items():
            opt = getattr(rs.option, name, None)
            if opt is None:
                warn_n += 1
                continue
            try:
                sensor.set_option(opt, want)
            except Exception as exc:
                # 部分固件(实测 5.13.0.55)对"写入等于当前值的冗余写"会报 hwmon -9，回读确认即可
                try:
                    cur = float(sensor.get_option(opt))
                except Exception:
                    cur = None
                if cur is not None and abs(cur - want) < 1e-3:
                    warn_n += 1
                else:
                    bad += 1
                    warn(f"相机 {self.name}: {name} 写入失败({exc})")
                continue
            try:
                got = float(sensor.get_option(opt))
            except Exception as exc:
                bad += 1
                warn(f"相机 {self.name}: {name} 回读失败({exc})")
                continue
            if abs(got - want) < 1e-3:
                ok += 1
            else:
                bad += 1
                warn(f"相机 {self.name}: {name} 写 {want:g} 读到 {got:g}")
        log(f"  相机 {self.name:<5} 彩色参数: 一致 {ok} / 提示 {warn_n} / 异常 {bad}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError:
                self.read_errors += 1
                continue
            cf = frames.get_color_frame()
            if not cf:
                continue
            img = np.asanyarray(cf.get_data())   # HWC uint8 BGR
            depth_img: Optional[np.ndarray] = None
            depth_no = -1
            if self.enable_depth:
                df = frames.get_depth_frame()
                if df:
                    depth_img = np.asanyarray(df.get_data())   # HWC uint16 (mm)
                    depth_no = df.get_frame_number()
            with self._lock:
                self._latest = img
                self._frame_no = cf.get_frame_number()
                self._ts_ns = time.time_ns()
                if depth_img is not None:
                    self._latest_depth = depth_img
                    self._depth_no = depth_no

    def latest(self) -> Tuple[Optional[np.ndarray], int, int]:
        """返回 (最新帧 HWC uint8, 帧号, 主机时间戳 ns)。未拿到帧时图像为 None。"""
        with self._lock:
            return self._latest, self._frame_no, self._ts_ns

    def latest_depth(self) -> Tuple[Optional[np.ndarray], int]:
        """返回 (最新深度帧 HWC uint16 mm, 帧号)。未开启深度流/未拿到帧时图像为 None。"""
        with self._lock:
            return self._latest_depth, self._depth_no

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


# ===========================================================================
# 四、键盘输入（非阻塞单键）
# ===========================================================================
class KeyInput:
    """cbreak 模式单键读取：空格立即返回，Ctrl+C 仍产生 SIGINT（ISIG 保留）。"""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin else 0
        self.is_tty = sys.stdin.isatty()
        self._saved: Optional[List[Any]] = None

    def __enter__(self) -> "KeyInput":
        if self.is_tty:
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)   # 保留 ISIG，Ctrl+C 仍能中断
        else:
            warn("标准输入不是终端，退化为整行输入模式（每行回车视为一次空格）")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def poll(self, timeout: Optional[float] = None) -> Optional[str]:
        """非阻塞/限时读取一个字符；超时返回 None。"""
        if not self.is_tty:
            line = sys.stdin.readline()
            if line == "":
                raise KeyboardInterrupt   # 输入流关闭，按 Ctrl+C 处理
            return " "
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        return os.read(self.fd, 1).decode(errors="ignore")


# ===========================================================================
# 五、数据集
# ===========================================================================
def build_features() -> Dict[str, Dict[str, Any]]:
    """LeRobotDataset features 定义（与数据契约一一对应）。

    - 深度字段：dtype=video + info.is_depth_map=True，lerobot 官方深度路径，
      save_episode 时自动做 12-bit log 量化编码，加载时自动反量化；
    - next.done：LeRobot v3 强制字段，每 episode 最后一帧 True（保存前设置）。
    """
    depth_names = ["channels", "height", "width"]
    return {
        IMG_TOP_KEY: {"dtype": "video", "shape": (3, CAM_H, CAM_W),
                      "names": ["channels", "height", "width"]},
        IMG_WRIST_KEY: {"dtype": "video", "shape": (3, CAM_H, CAM_W),
                        "names": ["channels", "height", "width"]},
        DEPTH_TOP_KEY: {"dtype": "video", "shape": (1, CAM_H, CAM_W),
                        "names": depth_names, "info": {"is_depth_map": True}},
        DEPTH_WRIST_KEY: {"dtype": "video", "shape": (1, CAM_H, CAM_W),
                          "names": depth_names, "info": {"is_depth_map": True}},
        STATE_KEY: {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)},
        EFFORT_KEY: {"dtype": "float32", "shape": (1,), "names": list(EFFORT_NAMES)},
        EE_POSE_KEY: {"dtype": "float32", "shape": (6,), "names": list(EE_POSE_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": list(ACTION_NAMES)},
        NEXT_DONE_KEY: {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }


def build_rgb_encoder() -> RGBEncoderConfig:
    """RGB 视频编码配置：H.264（libx264），yuv420p，兼顾压缩率与解码兼容性。"""
    return RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=28, preset="faster")


def build_depth_encoder() -> DepthEncoderConfig:
    """深度视频编码配置：lerobot 官方深度路径（非 uint8，原因见模块 docstring 坑点）。

    - 12-bit log 量化（depth_min=0.01m, depth_max=10m, shift=3.5, use_log=True），
      量化级数 4096，近场精度优于 uint8 的 256 级；
    - HEVC Main12 + gray12le + lossless（x265-params=lossless=1），
      保证反量化误差只来自量化本身，无编码损失；
    - 加载端：原生 LeRobotDataset 读到的是自动反量化后的 float32 米制深度。
    """
    return DepthEncoderConfig()


# ===========================================================================
# 六、采集主控制器
# ===========================================================================
class Collector:
    """采集主状态机。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.arm = PiperArm(args.can)
        self.cmd = ArmCommander(self.arm, speed=args.speed)
        # 深度流常开：录制 observation.depth.* 需要（--preview 的深度伪彩同样复用）
        self.cam_top = CameraStream("top", CAM_TOP_SERIAL, COLOR_WHITE_BALANCE["top"],
                                    enable_depth=True)
        self.cam_wrist = CameraStream("wrist", CAM_WRIST_SERIAL, COLOR_WHITE_BALANCE["wrist"],
                                      enable_depth=True)
        self.dataset: Optional[LeRobotDataset] = None
        self.key: Optional[KeyInput] = None
        self.episode_count = 0
        self.recording = False
        self._preview_ok = True   # 预览初始化/运行失败后置 False，自动降级不再弹窗
        # 后台 episode 保存（视频编码耗时约录制时长 1~2 倍）：
        # 保存任务队列 + 常驻单 worker 线程 —— 主线程交接 buffer 后立即继续下一轮录制，
        # 保存与采集【真正并行】，互不阻塞（并发安全性见 _save_worker_loop 注释）
        self._save_queue: "queue.Queue" = queue.Queue()
        self._save_worker: Optional[threading.Thread] = None
        self._next_episode_idx = 0            # 主线程预分配的下一个 episode 编号
        self._save_errors: List[Tuple[int, BaseException]] = []

    # ------------------------------------------------------------------
    # 资源生命周期
    # ------------------------------------------------------------------
    def open_resources(self) -> None:
        step("【步骤1a】硬件初始化 + 全设备状态自检")

        # 1) 相机设备枚举
        serials = CameraStream.list_serials() if not self.args.no_camera else []
        if not self.args.no_camera:
            log(f"RealSense 设备: {serials if serials else '无'}")
            for need in (CAM_TOP_SERIAL, CAM_WRIST_SERIAL):
                if need not in serials:
                    raise RuntimeError(f"未找到必需相机 serial={need}（当前: {serials}）")

        # 2) 机械臂连接
        self.arm.connect()
        self._selfcheck_arm()

        # 2b) 启动指令下发线程（关键！）
        #     ArmCommander 是 Thread 子类，只 __init__ 不 start() 的话线程根本不会运行，
        #     所有 start_move/start_hold 只是改了内部状态，一条 CAN 报文都发不出去，
        #     表现就是"运动指令全部无效、误差一动不动"。
        #     它的 _paused 初值为 True，启动后处于暂停态，不会主动下发任何指令，
        #     必须由 start_move / start_hold 显式唤醒。
        assert not self.cmd.is_alive()
        self.cmd.start()
        log("  指令下发线程已启动（初始为暂停态，不会主动下发任何指令）")

        # 3) 相机启动
        if not self.args.no_camera:
            self.cam_top.start()
            self.cam_wrist.start()

        # 4) 数据集【不在这里创建】：
        #    创建数据集属于"采集"动作，必须等到第 1 次按下空格后才执行，
        #    避免"还没按空格就好像已经开始采集"。见 _ensure_dataset()。
        log("  提示：本阶段未创建数据集、未下发任何运动指令；"
            "按下第 1 次空格后才创建数据集并开始录制")

    def _selfcheck_arm(self) -> None:
        """等待三路 CAN 反馈到齐，并打印状态。"""
        t0 = time.time()
        snap = None
        while time.time() - t0 < 5.0:
            snap = self.arm.read()
            if snap.joints_valid and snap.gripper_valid and snap.ee_pose_si is not None:
                break
            time.sleep(0.1)
        if snap is None or not snap.joints_valid:
            raise RuntimeError("机械臂关节反馈未到齐，请检查 can0 与机械臂上电状态")
        log(f"  机械臂状态自检通过（耗时 {time.time() - t0:.2f}s）:\n         {snap.pretty()}")
        missing = []
        if not snap.gripper_valid:
            missing.append("夹爪")
        if snap.ee_pose_si is None:
            missing.append("末端位姿")
        if missing:
            warn(f"以下反馈未收到，对应字段将按无效处理: {missing}")
        if snap.enable and not all(snap.enable):
            warn(f"电机未全部使能: {snap.enable}")

    def _ensure_dataset(self) -> None:
        """【步骤3】第 1 次按下空格后才创建数据集，且只创建一次。

        把数据集创建从"启动时"推迟到"第 1 次空格后"，保证按空格之前
        程序不产生任何与采集相关的动作（不建目录、不写文件、不下发指令）。
        """
        # 后台保存已不再阻塞采集（保存与录制真正并行），只检查历史保存错误
        self._raise_if_save_failed()
        if self.dataset is not None:
            return
        self._prepare_dataset()
        # 崩溃安全：每个 episode 独立成文件并立即关闭写入器（新建/续录均生效）
        self._enable_crash_safe_saving()

    # ------------------------------------------------------------------
    # 后台 episode 保存（保存与采集真正并行，互不阻塞）
    # ------------------------------------------------------------------
    # 并发安全性（为什么能与主线程 add_frame 并行而不写坏数据集）：
    #   1. meta 只由本 worker 单线程更新（FIFO 串行保存，无并发写 meta）；
    #   2. validate_episode_buffer 校验 episode_index == meta.total_episodes，
    #      FIFO 顺序保证入队顺序=录制顺序，校验恒通过；
    #   3. PNG 临时文件目录按 episode-{idx:06d} 隔离，主线程录新 episode 写的是
    #      预分配的新编号目录，与本线程正在编码的目录不冲突；
    #   4. save_episode 内部 _wait_image_writer() 等待的是共享 PNG 落盘队列排空，
    #      FIFO 保证旧 episode 的帧全部落盘后队列才可能排空，读到的一定完整；
    #      主线程持续提交新帧时最多把保存拖到录制间隙，绝不会死锁或丢帧。
    def _save_worker_loop(self) -> None:
        """常驻保存线程：串行消费保存队列，与主线程采集完全并行。"""
        while True:
            item = self._save_queue.get()
            try:
                if item is None:      # 退出哨兵
                    break
                ep_no, ep_idx, buf = item
                try:
                    t0 = time.time()
                    # LeRobot v3 强制字段：episode 最后一帧 next.done=True。
                    # 已在主线程入队前回写完毕，此处直接保存
                    # 编码窗口静默 stderr（fd 级）：libx264/x265 统计直写 fd2
                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    saved_stderr_fd = os.dup(2)
                    try:
                        os.dup2(devnull_fd, 2)
                        self.dataset.save_episode(episode_data=buf, parallel_encoding=False)
                    finally:
                        os.dup2(saved_stderr_fd, 2)
                        os.close(saved_stderr_fd)
                        os.close(devnull_fd)
                    # 崩溃安全：立即关闭 data / episodes 元数据 parquet 写入器
                    # （close 才写 footer）。不关闭的话，进程被硬中断（断电/
                    # kill -9/误关终端）时，最新 episode 的表格数据会缺 footer
                    # 损坏且无法续录。视频无需处理：每 episode 编码完即完整。
                    self.dataset.writer.close_writer()   # data parquet
                    self.dataset.meta._close_writer()   # episodes 元数据 parquet（先 flush 缓冲再 close）
                    # episode_data 路径不会清理 PNG 临时目录（writer 只在内部 buffer 路径清理），
                    # 手动删除本 episode 的 PNG，避免磁盘持续膨胀
                    w = self.dataset.writer
                    for key in w._meta.image_keys:
                        shutil.rmtree(w._get_image_file_dir(ep_idx, key), ignore_errors=True)
                    self.episode_count += 1
                    log(f"[后台] episode {ep_no} 已保存（视频编码+落盘 耗时 {time.time() - t0:.1f}s）")
                except BaseException as exc:  # noqa: BLE001 —— 记录后在下一安全点上抛
                    self._save_errors.append((ep_no, exc))
                    warn(f"[后台] episode {ep_no} 保存失败: {exc}")
            finally:
                self._save_queue.task_done()

    def _ensure_save_worker(self) -> None:
        if self._save_worker is None or not self._save_worker.is_alive():
            self._save_worker = threading.Thread(
                target=self._save_worker_loop, name="save-worker", daemon=True)
            self._save_worker.start()

    def _raise_if_save_failed(self) -> None:
        """在安全点（下一轮录制开始 / 退出）检查后台保存错误并上抛。"""
        if self._save_errors:
            errs = "; ".join(f"episode {n}: {e}" for n, e in self._save_errors)
            self._save_errors.clear()
            raise RuntimeError(f"后台保存失败，数据集状态可能不完整，请检查: {errs}")

    def _prepare_dataset(self) -> None:
        root = Path(self.args.dataset_root).expanduser().resolve() / self.args.repo_id
        if root.exists() and any(root.iterdir()):
            if self._is_empty_dataset(root):
                # 上一次启动失败(例如运动中止)会留下 0 episode 的空数据集，直接清理重建
                log(f"检测到上次遗留的空数据集(0 episode)，自动清理后重建: {root}")
                shutil.rmtree(root)
            elif self._load_existing_dataset(root):
                # 自动续录：以可写模式加载已有数据集，新 episode 追加到同一目录
                return
            else:
                raise RuntimeError(
                    f"数据集目录已存在且非空，但不是可续录的 LeRobotDataset: {root}\n"
                    f"如需重新采集，请先备份/改名或指定 --dataset-root/--repo-id"
                )
        log(f"创建 LeRobotDataset: repo_id={self.args.repo_id} root={root} fps={self.args.fps} use_videos=True")
        self.dataset = LeRobotDataset.create(
            repo_id=self.args.repo_id,
            fps=self.args.fps,
            features=build_features(),
            root=root,
            robot_type="piper",
            use_videos=True,          # 图像走 MP4 视频编码
            # PNG 落盘异步化：实测 1280x720 单帧同步落盘约 78ms，同步写盘只能跑到 6.4Hz，
            # 无法满足 30Hz 录制；改为 8 线程异步落盘（本机 32 核）后余量充足。
            image_writer_threads=8,
            rgb_encoder=build_rgb_encoder(),     # RGB: H.264
            depth_encoder=build_depth_encoder(), # depth: 12-bit log 量化(官方路径)
        )
        log(f"  数据集已创建，字段: {sorted(build_features().keys())}（尚未写入任何帧）")
        self._next_episode_idx = 0   # 全新数据集：episode 编号从 0 预分配

    def _enable_crash_safe_saving(self) -> None:
        """崩溃安全落盘：每个 episode 独立成文件，保存后立即关闭全部写入器。

        事故背景（2026-09-21，piper_task_stack_boxes 损失 8 个 episode）：
        lerobot 默认 100MB/200MB 阈值才轮转文件，data parquet 与 episodes 元数据
        parquet 的写入器在轮转前一直保持打开 —— parquet 仅在 close 时写 footer。
        进程一旦被硬中断（断电/kill -9/直接关终端），自文件创建以来的全部
        episode 表格数据损坏且无法续录（视频不受影响：每 episode 编码完即完整）。

        方案（两处配合）：
        1) 把轮转阈值调到极小值，令每次 save_episode 内部 data/episodes parquet
           与四路视频都轮转为新文件（data/video 的旧写入器随轮转关闭）；
        2) 保存线程在每次 save_episode 返回后再显式关闭 data 与 episodes 元数据
           写入器（episodes 元数据的 RAM 缓冲只在轮转/finalize 时落盘）。
        硬中断的损失因此收窄到"正在保存中的那一个 episode"。
        代价：每 episode 一组小文件。该设置随 info.json 持久化，长期生效。
        """
        self.dataset.meta.info.data_files_size_in_mb = 0.0001
        self.dataset.meta.info.video_files_size_in_mb = 0.0001

    def _load_existing_dataset(self, root: Path) -> bool:
        """已有数据集时【自动续录】：以可写模式加载，新 episode 追加到同一目录。

        - 必须用 LeRobotDataset.resume()：普通构造函数加载的数据集是只读的，
          add_frame 会抛 "Cannot call 'add_frame()' on a read-only dataset"。
        - 续录前做两项兼容性校验（不一致直接报错拒绝，防止写坏数据集）：
          1) fps 一致；
          2) features 一致（json round-trip 归一化 tuple->list 后，
             复用 lerobot 官方 features_equal_for_merge：忽略 video 编码参数 info 键）。
        - 目录非空但没有 meta/info.json（不是有效数据集）时返回 False，
          由调用方报错提示人工处理。
        """
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            return False
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if "total_episodes" not in info or "fps" not in info:
            return False

        # 落盘完整性预检：拦截上次硬中断留下的损坏数据集（详见 _check_dataset_intact）。
        # 必须在 resume 之前做 —— 损坏数据集强行 resume 会误转 HuggingFace Hub 报 401。
        self._check_dataset_intact(root)

        old_fps = int(info["fps"])
        if old_fps != int(self.args.fps):
            raise RuntimeError(
                f"已有数据集 fps={old_fps} 与当前 --fps={self.args.fps} 不一致，拒绝续录: {root}\n"
                f"请保持相同的 fps，或指定新的 --dataset-root/--repo-id 重新采集"
            )
        old_features = json.loads(json.dumps(info.get("features", {})))   # tuple->list 归一化
        new_features = json.loads(json.dumps(build_features()))
        # 剔除 lerobot 自动附加的索引元字段（dataset_tools.py: required_features），
        # 否则已有数据集(含 index/timestamp 等)永远比不过 build_features()(只有业务字段)
        META_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        old_features = {k: v for k, v in old_features.items() if k not in META_FEATURE_KEYS}

        # video 的 "info" 键两侧来源不同：已有数据集是 save_episode 写入的完整编码参数
        # (codec/crf/pix_fmt/...)，build_features() 只有数据模式标记 —— 编码参数可忽略，
        # 但 is_depth_map 决定加载时是否做深度反量化，必须参与比较。
        # 因此【双边】规范化 info 为只含 is_depth_map 后再比较（实测教训：只剔单边必误报）
        def _norm_video_info(feat: dict) -> dict:
            if isinstance(feat, dict) and feat.get("dtype") == "video":
                info = feat.get("info") or {}
                feat["info"] = {"is_depth_map": bool(info.get("is_depth_map", False))}
            return feat

        old_features = {k: _norm_video_info(v) for k, v in old_features.items()}
        new_features = {k: _norm_video_info(v) for k, v in new_features.items()}
        if not features_equal_for_merge(old_features, new_features):
            raise RuntimeError(
                f"已有数据集字段与当前录制字段不一致，拒绝续录: {root}\n"
                f"已有字段: {sorted(old_features)}\n当前字段: {sorted(new_features)}\n"
                f"常见原因：录制格式升级（如新增 observation.depth.*/next.done）。\n"
                f"请指定新的 --dataset-root/--repo-id 重新采集，旧数据集保持原样"
            )

        total_eps = int(info["total_episodes"])
        total_frames = int(info.get("total_frames", 0))
        log(f"检测到已有数据集，【自动续录】（不清空、不覆盖，新 episode 追加写入）: {root}")
        log(f"  已有 {total_eps} 个 episode / {total_frames} 帧")
        self.dataset = LeRobotDataset.resume(
            repo_id=self.args.repo_id,
            root=root,
            image_writer_threads=8,
            # 深度编码参数(depth_min/max/shift/use_log)自动从已有 info.json 的
            # features[*].info 恢复（DepthEncoderConfig.from_video_info），
            # 保证续录 episode 的量化参数与已有数据一致
            rgb_encoder=build_rgb_encoder(),
            depth_encoder=build_depth_encoder(),
        )
        # 清理上次异常退出遗留的图片临时文件（正常退出会随保存/退出自动清理）。
        # 残留目录的编号与即将录制的新 episode 相同，旧帧会混入新视频，
        # 导致视频帧数 > parquet 行数（数据错位），必须在续录前清掉。
        stale_img_root = root / "images"
        if stale_img_root.is_dir():
            n_stale = sum(1 for p in stale_img_root.rglob("*") if p.is_file())
            if n_stale:
                shutil.rmtree(stale_img_root)
                warn(f"已清理上次异常退出遗留的临时图片 {n_stale} 帧（不影响已保存 episode）")
        # 清理上次硬中断遗留的视频编码临时目录（tempfile.mkdtemp 生成，仅占磁盘
        # 不影响正确性，但残留会一直占用空间）
        for stale_tmp in root.glob("tmp*"):
            if stale_tmp.is_dir():
                shutil.rmtree(stale_tmp, ignore_errors=True)
                warn(f"已清理上次异常退出遗留的视频临时目录 {stale_tmp.name}")
        self.episode_count = total_eps
        self._next_episode_idx = total_eps   # 预分配编号与已有 episode 数对齐
        log(f"  已以可写模式加载(resume)，episode 编号从 {total_eps + 1} 继续")
        return True

    @staticmethod
    def _check_dataset_intact(root: Path) -> None:
        """续录前落盘完整性预检：拦截上次硬中断留下的损坏数据集。

        正常退出(Ctrl+C)会 finalize —— 所有 parquet 带 footer、meta/episodes/
        已完整落盘。硬中断（断电/kill -9/直接关终端）则会留下：
          1) info.json 记录了 N 个 episode，但 meta/episodes/ 下没有任何 parquet
             （episode 元数据仍在 RAM 缓冲，从未写盘）；
          2) data/ 或 meta/episodes/ 下有缺 footer 的 parquet（写入器未关闭）。
        此时强行 resume，lerobot 会误判为本地缺失、转而请求 HuggingFace Hub，
        报出误导性的 401 Repository Not Found —— 实际是本地数据损坏。
        """
        eps_dir = root / "meta" / "episodes"
        eps_parquets = sorted(eps_dir.glob("*/*.parquet")) if eps_dir.is_dir() else []
        if not eps_parquets:
            raise RuntimeError(
                f"数据集已损坏，拒绝续录: {root}\n"
                f"  info.json 声明已有 episode，但 meta/episodes/ 下没有任何 parquet\n"
                f"  —— 上次程序疑似被硬中断（断电/强制结束/直接关闭终端），\n"
                f"     episode 元数据未落盘；此时 resume 会误报 HuggingFace 401。\n"
                f"  处理：将整个目录改名备份（视频通常仍完整，可留待抢救），\n"
                f"     再用原 --repo-id 重新采集。"
            )
        # episodes 元数据 parquet 必须 all-valid（保存线程每 episode flush+close 一次；
        # 缺 footer 说明恰好崩在 flush 中途，无法自愈）
        for f in eps_parquets:
            try:
                pq.read_metadata(f)
            except Exception:
                raise RuntimeError(
                    f"数据集已损坏，拒绝续录: {f}\n"
                    f"  episodes 元数据 parquet 缺 footer（写入器未正常关闭）。\n"
                    f"  处理：将整个数据集目录 {root} 改名备份后重新采集。"
                ) from None
        # data parquet：仅允许【编号最大的一个】缺 footer —— 那是硬中断瞬间
        # 正在保存的 episode，下次 save_episode 会在同编号文件上截断重建，自愈；
        # 更早的文件缺 footer 则是实损坏，必须拦截。
        data_files = sorted((root / "data").glob("*/*.parquet"))
        bad_data = []
        for f in data_files:
            try:
                pq.read_metadata(f)
            except Exception:
                bad_data.append(f)
        if bad_data and (len(bad_data) > 1 or bad_data[0] != data_files[-1]):
            raise RuntimeError(
                f"数据集已损坏，拒绝续录: {bad_data}\n"
                f"  data parquet 缺 footer 且不是最新文件（写入器未正常关闭）。\n"
                f"  处理：将整个数据集目录 {root} 改名备份后重新采集。"
            ) from None

    @staticmethod
    def _is_empty_dataset(root: Path) -> bool:
        """仅当 meta/info.json 明确写着 total_episodes==0 时才认为是可清理的空数据集。"""
        info = root / "meta" / "info.json"
        if not info.is_file():
            return False
        try:
            return int(json.loads(info.read_text(encoding="utf-8")).get("total_episodes", -1)) == 0
        except Exception:
            return False

    def close_resources(self) -> None:
        step("【步骤8】释放硬件资源")
        self.cmd.stop()
        # 后台保存队列必须先排空：worker 可能正在编码/写 meta，
        # 此时 clear_episode_buffer/finalize 会破坏正在进行的保存。
        # Ctrl+C 时会在此等待（正在编码的任务 + 排队任务），请勿再次 Ctrl+C
        if self._save_worker is not None and self._save_queue.unfinished_tasks > 0:
            log(f"后台仍有 {self._save_queue.unfinished_tasks} 个 episode 待保存/保存中，"
                f"等待全部完成（不要再次 Ctrl+C）...")
        self._save_queue.join()
        if self._save_worker is not None:
            self._save_queue.put(None)          # 退出哨兵
            self._save_worker.join(timeout=10)
        for ep_no, exc in self._save_errors:    # 退出路径只告警不抛
            warn(f"episode {ep_no} 后台保存失败: {exc}")
        self._save_errors.clear()
        if self.dataset is not None and self.dataset.has_pending_frames():
            # 录制中途退出：丢弃未完成的 episode，避免写入半截轨迹与残留图片
            try:
                self.dataset.clear_episode_buffer()
                warn("退出时仍有未保存的录制帧，已丢弃该未完成 episode")
            except Exception as exc:
                warn(f"丢弃 episode 失败: {exc}")
        for cam in (self.cam_top, self.cam_wrist):
            try:
                cam.stop()
            except Exception as exc:
                warn(f"相机 {cam.name} 释放异常: {exc}")
        log("两台相机已释放")
        if self.dataset is not None:
            try:
                self.dataset.finalize()
                log(f"数据集已 finalize（共 {self.episode_count} 个 episode）")
            except Exception as exc:
                warn(f"数据集 finalize 异常: {exc}")
        self.arm.disconnect()

    # ------------------------------------------------------------------
    # 运动
    # ------------------------------------------------------------------
    def _enter_can_mode(self, timeout: float) -> bool:
        """使能 + 轮询 0x151，直到【ctrl_mode == 0x01 且 6 个电机全部使能】。

        注意：ResetPiper 之后使能反馈是异步到的实测更慢（低速信息帧），
        因此 EnableArm 不能只发一次，必须与 ctrl_mode 一起轮询确认，
        否则会误判为"电机未使能"。
        """
        t0 = time.time()
        enable_resent = 0
        while time.time() - t0 < timeout:
            try:
                self.arm.send_motion_ctrl(0x01, 0x01, self.args.speed)
            except Exception as exc:
                warn(f"模式切换指令异常: {exc}")
            mode = self.arm.ctrl_mode()
            enabled = self.arm.enable_state()
            all_enabled = bool(enabled) and all(enabled)
            if mode == 0x01 and all_enabled:
                log(f"  已进入 CAN 指令控制模式(0x01) 且 6 个电机全部使能，"
                    f"耗时 {time.time() - t0:.2f}s")
                return True
            if not all_enabled:
                # 未全部使能则重复下发使能指令（单次下发可能落在机械臂未就绪的时刻而丢失）
                try:
                    self.arm.enable()
                    enable_resent += 1
                except Exception as exc:
                    warn(f"使能异常: {exc}")
            if enable_resent == 1 or enable_resent % 40 == 0:
                log(f"  等待使能... ctrl_mode=0x{mode:02X} 使能状态={enabled}")
            time.sleep(0.05)
        return False

    def _ensure_can_mode(self, allow_reset: bool, confirm_reset: bool) -> None:
        """确保处于 CAN 指令控制模式；必要时（经人工确认后）强制下电复位。"""
        mode = self.arm.ctrl_mode()
        log(f"  当前 ctrl_mode=0x{mode:02X}")
        if self._enter_can_mode(timeout=3.0):
            return
        if not allow_reset:
            raise RuntimeError(f"无法进入 CAN 指令控制模式(当前 0x{mode:02X})")
        warn("机械臂不在 CAN 指令控制模式（示教模式 0x02 会静默忽略所有运动指令），"
             "必须执行【强制下电复位】才能切出。")
        warn("⚠ ResetPiper() 会立即失电，机械臂将自由下落，请务必先托住机械臂或确认其处于安全低位！")
        if confirm_reset:
            self._require_confirm("确认已托住/机械臂处于安全位置，继续下电复位？按 y 继续，其他键取消：")
        log("  执行 ResetPiper() 强制下电复位 ...")
        self.arm.reset_piper()
        time.sleep(0.5)
        if not self._enter_can_mode(timeout=MODE_SWITCH_TIMEOUT_S):
            raise RuntimeError("强制下电复位后仍无法进入 CAN 指令控制模式")

    def _takeover_after_teach(self) -> None:
        """【步骤6】第 3 次空格后的接管：【直接】下电复位，不再二次确认。

        拖拽示教后机械臂处于示教模式 0x02，运动与模式切换指令都会被静默忽略，
        无法通过其他指令重新取得控制权 —— 下电复位是唯一出口。
        第 3 次空格本身就是"已托稳"的人工确认，因此：
          - 不再做 3 秒模式切换重试（在 0x02 下注定无效，纯浪费时间）；
          - 不再弹 y 二次确认。
        仅当碰巧仍在 CAN 模式且全部使能（本轮没有真正进入示教）时才跳过复位。
        """
        if self._enter_can_mode(timeout=0.5):
            log("  仍处于 CAN 指令控制模式(0x01)且已使能，无需下电复位")
            return
        warn("⚠ ResetPiper() 立即执行，机械臂瞬时失电！（第 3 次空格已作为托稳确认）")
        self.arm.reset_piper()
        time.sleep(0.5)
        if not self._enter_can_mode(timeout=MODE_SWITCH_TIMEOUT_S):
            raise RuntimeError("强制下电复位后仍无法进入 CAN 指令控制模式(0x01)")

    def _require_confirm(self, prompt: str) -> None:
        log(prompt)
        assert self.key is not None
        while True:
            k = self.key.poll(timeout=0.2)
            if k is None:
                continue
            if k.lower() == "y":
                return
            raise RuntimeError("用户取消操作")

    def move_to(self, target: List[float], label: str) -> None:
        """连续 200Hz 下发目标关节角，直到到位判定通过。"""
        validate_joints(target, label)
        log(f"  运动到 {label}: {['%+.4f' % v for v in target]} rad")
        # 打印实际下发帧里的整数（单位 0.001 度），用于核对指令内容本身是否正确
        sdk_target = [int(round(v * RAD_TO_SDK)) for v in target]
        log(f"    下发帧(0.001度): {sdk_target}")
        snap0 = self.arm.read()
        if snap0.joints_valid:
            log(f"    当前帧(0.001度): {[int(round(v * RAD_TO_SDK)) for v in snap0.joints_rad]}")
        self.cmd.start_move(target)
        try:
            self._wait_arrival(target, label)
        finally:
            # 到位后转为低频保持，而不是完全停发（停发超过超时保护会掉使能）
            self.cmd.start_hold(target)
        log(f"  {label} 到位判定通过")

    def _wait_arrival(self, target: List[float], label: str) -> None:
        t0 = time.time()
        bad_streak = 0
        arm_bad_streak = 0
        seen_enabled = False   # 必须先观察到正常状态，之后失能/掉模式才算真故障
        err: Optional[List[float]] = None
        next_diag = t0 + 1.0
        sent_at_check = self.cmd.sent_count()   # 用于确认指令线程真的在发报文
        checked_sending = False
        while True:
            snap = self.arm.read()
            if snap.joints_valid:
                err = [abs(a - b) for a, b in zip(snap.joints_rad, target)]
                if max(err) < ARRIVE_TOL_RAD:
                    return

            # 自检：运动开始约 0.5s 后，指令计数必须增长。若一条都没发出去，
            # 说明指令线程没在跑（例如忘了 Thread.start()），立刻报错而不是干等超时。
            if not checked_sending and time.time() - t0 > 0.5:
                checked_sending = True
                if self.cmd.sent_count() == sent_at_check:
                    raise RuntimeError(
                        f"运动到 {label} 期间【指令下发线程未发出任何报文】"
                        f"（已下发计数始终为 {sent_at_check}）——机械臂不可能运动。"
                        f"请检查 ArmCommander 线程是否已 start()。"
                    )
            ok, why = self.arm.status_ok()
            if ok:
                seen_enabled, bad_streak = True, 0
            else:
                bad_streak += 1

            # 固件状态字报出会阻止运动的原因（超限/碰撞/抱闸/通信异常）：
            # 这种故障重新切 CAN 模式是无效的，持续约 0.5s 就直接报出真实原因。
            blocking = self.arm.blocking_arm_status()
            arm_bad_streak = arm_bad_streak + 1 if blocking else 0
            now = time.time()
            if arm_bad_streak >= 25:
                raise RuntimeError(
                    f"运动到 {label} 被机械臂固件拒绝: {blocking}\n"
                    f"各关节剩余误差: {['%.4f' % v for v in err] if err else '无反馈'}\n"
                    f"该状态不会因重切 CAN 模式而恢复，请先排除机械原因"
                    f"（目标位姿是否可达/是否被遮挡/抱闸/急停），再重新运行本程序"
                )

            # 运动中掉出 CAN 指令控制模式/失能并持续约 0.5s：先尝试重新夺回控制权，失败才停机
            if seen_enabled and bad_streak >= 25:
                warn(f"运动中状态异常({label}): {why} —— 尝试重新切换到 CAN 指令控制模式 ...")
                if self._enter_can_mode(timeout=2.0):
                    log("  已重新夺回控制权，继续运动")
                    seen_enabled, bad_streak, err = False, 0, None
                    t0 = now          # 重新计时，避免因故障重试直接超时
                    continue
                raise RuntimeError(
                    f"运动过程中硬件状态异常({label}): {why}\n"
                    f"提示：ctrl_mode=0x02 是示教模式，会静默忽略所有运动指令。"
                    f"回位/就位过程中请勿触碰机械臂本体，也不要按绿色按键；"
                    f"若已被切到示教模式，需重新运行本程序（启动时会提示下电复位切口）"
                )
            if now >= next_diag:
                next_diag = now + 1.0
                max_err = max(err) if err else float("nan")
                per_joint = (" ".join(f"{v:.3f}" for v in err)) if err else "无反馈"
                log(f"  运动中... ctrl_mode=0x{snap.ctrl_mode:02X} 使能={snap.enable} "
                    f"已下发={self.cmd.sent_count()} 组 最大误差={max_err:.4f} rad "
                    f"(阈值 {ARRIVE_TOL_RAD}) 已用时 {now - t0:.1f}s")
                log(f"    各关节误差[{per_joint}]")
                log(f"    {self.arm.arm_status_text()}")
            if now - t0 > ARRIVE_TIMEOUT_S:
                snap = self.arm.read()
                errs = ([f"{abs(a - b):.4f}" for a, b in zip(snap.joints_rad, target)]
                        if snap.joints_valid else ["无反馈"])
                raise RuntimeError(
                    f"运动到 {label} 超时({ARRIVE_TIMEOUT_S}s)，未达到到位判定阈值 "
                    f"{ARRIVE_TOL_RAD} rad，各关节剩余误差: {errs}；"
                    f"ctrl_mode=0x{snap.ctrl_mode:02X} 使能={snap.enable} "
                    f"已下发={self.cmd.sent_count()} 组；"
                    f"{self.arm.arm_status_text()}"
                )
            time.sleep(0.02)

    def go_home_then_start(self) -> None:
        """【步骤1b / 步骤6 尾部】回 HOME_BASE -> 停 1s -> 平滑到 START_POSE。"""
        warn("回位/就位运动期间请勿触碰机械臂本体，也不要按绿色按键"
             "（绿色按键会切入示教模式 0x02，运动指令会被静默忽略）")
        self.move_to(HOME_BASE, "复位基准位姿 HOME_BASE")
        log(f"  在 HOME_BASE 停留 {HOME_HOLD_S}s")
        time.sleep(HOME_HOLD_S)
        self.move_to(START_POSE, "采集初始位姿 START_POSE")

    # ------------------------------------------------------------------
    # 录制
    # ------------------------------------------------------------------
    def wait_space(self, prompt: str) -> None:
        assert self.key is not None
        log(prompt)
        while True:
            k = self.key.poll(timeout=0.05)       # 终端按键
            pk = self._maybe_preview()            # 刷新预览；预览窗口有焦点时按键在这里
            if " " in (k, pk):
                log("检测到【空格】")
                return

    def _maybe_preview(self) -> Optional[str]:
        """刷新预览窗口并返回 cv2 窗口捕获的按键（如 ' '）；无则返回 None。

        单窗口布局（1280x720，四块 640x360 瓦片）：
            [ top:   RGB | 深度伪彩 ]
            [ wrist: RGB | 深度伪彩 ]
        - 深度流仅在 --preview 时开启，只用于预览展示；数据集仍只录制 RGB（数据契约不变）。
        - 预览窗口拿到键盘焦点时，空格根本不会送达终端 —— 因此把 cv2.waitKey
          捕获到的按键一并返回，由调用方与终端按键合并判定。
        - cv2 未安装 / 开窗失败时自动降级：告警一次并关闭预览，采集不受影响。
        """
        if not self.args.preview or not self._preview_ok:
            return None
        if cv2 is None:
            self._preview_ok = False
            warn("未安装 GUI 版 opencv（opencv-python），预览不可用；采集不受影响")
            return None
        try:
            rows = [self._preview_row(cam.name, cam.latest()[0], cam.latest_depth()[0])
                    for cam in (self.cam_top, self.cam_wrist)]
            cv2.imshow(PREVIEW_WINDOW, np.vstack(rows))
            k = cv2.waitKey(1)
            if k < 0:          # -1 = 无按键
                return None
            k &= 0xFF
            return chr(k) if 0 < k < 127 else None
        except Exception as exc:
            # 无显示器/SSH 无 X 转发/headless opencv 等场景：关掉预览继续采集，不让流程崩溃
            self._preview_ok = False
            warn(f"预览不可用（{exc}），已自动关闭预览，采集流程继续")
            return None

    @staticmethod
    def _preview_row(name: str, rgb: Optional[np.ndarray],
                     depth: Optional[np.ndarray]) -> np.ndarray:
        """单台相机的一行预览瓦片：[RGB | 深度伪彩]，各 640x360，带文字标注。"""

        def blank(text: str) -> np.ndarray:
            tile = np.zeros((PREVIEW_TILE_H, PREVIEW_TILE_W, 3), np.uint8)
            cv2.putText(tile, text, (16, PREVIEW_TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2, cv2.LINE_AA)
            return tile

        if rgb is not None:
            t_rgb = cv2.resize(rgb, (PREVIEW_TILE_W, PREVIEW_TILE_H))
        else:
            t_rgb = blank(f"{name}: no RGB frame")
        cv2.putText(t_rgb, f"{name} RGB", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if depth is not None:
            # uint16(mm) -> 伪彩：约 3.9m 满量程；无效值 0 显示为深蓝/黑色
            d8 = cv2.convertScaleAbs(depth, alpha=PREVIEW_DEPTH_SCALE)
            t_dep = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
            t_dep = cv2.resize(t_dep, (PREVIEW_TILE_W, PREVIEW_TILE_H))
        else:
            t_dep = blank(f"{name}: no depth frame")
        cv2.putText(t_dep, f"{name} depth", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return np.hstack([t_rgb, t_dep])

    def record_episode(self) -> None:
        """【步骤3~4】录制一个 episode，直到按下【第 2 次空格】。

        本阶段程序只读反馈，绝不下发任何运动指令（避免与人工拖拽对抗）。
        第 2 次空格只负责"停止录制并保存"，程序仍然不接管机械臂运动。
        """
        assert self.dataset is not None and self.key is not None
        self.cmd.pause()          # 关键：录制期间停止一切指令下发
        self.recording = True
        period = 1.0 / self.args.fps
        n_frames = n_dup = 0
        last_no = {"top": -1, "wrist": -1}
        t_start = time.perf_counter()
        next_t = t_start
        next_status = t_start + 1.0
        log(f"开始录制 episode {self._next_episode_idx + 1}（目标 {self.args.fps}Hz，"
            f"两路 RGB+深度 {CAM_W}x{CAM_H} + 关节状态）；再按【第 2 次空格】停止录制")

        try:
            while True:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

                term_key = self.key.poll(timeout=0)   # 终端按键
                gui_key = self._maybe_preview()       # 刷新预览；预览窗口有焦点时按键在这里
                if " " in (term_key, gui_key):
                    log("检测到【第 2 次空格】，停止录制（程序仍然【不接管运动】）")
                    break

                snap = self.arm.read()   # 只读反馈，本阶段不下发任何指令
                state = snap.state_vector()
                effort = snap.effort_vector()
                if state is None or effort is None or snap.ee_pose_si is None:
                    continue

                frames: Dict[str, np.ndarray] = {}
                depths: Dict[str, np.ndarray] = {}
                for cam in (self.cam_top, self.cam_wrist):
                    img, no, _ = cam.latest()
                    if img is None:
                        continue
                    depth_img, _dno = cam.latest_depth()
                    if depth_img is None:      # 深度无效 -> 整帧丢弃（与 RGB 同等校验）
                        continue
                    if no == last_no[cam.name]:
                        n_dup += 1
                    last_no[cam.name] = no
                    # 相机流为 bgr8（OpenCV 顺序），数据契约/lerobot 视频编码按 RGB 解释：
                    # 写入前必须反转通道，否则录出的视频红蓝颠倒（预览走 BGR 不受影响）
                    frames[cam.name] = np.ascontiguousarray(img[..., ::-1])
                    # (H,W) uint16 mm -> (1,H,W)：lerobot 深度特征的通道优先布局
                    depths[cam.name] = depth_img[np.newaxis, :, :]
                if len(frames) != 2 or len(depths) != 2:
                    continue

                frame = {
                    IMG_TOP_KEY: frames["top"],
                    IMG_WRIST_KEY: frames["wrist"],
                    DEPTH_TOP_KEY: depths["top"],
                    DEPTH_WRIST_KEY: depths["wrist"],
                    STATE_KEY: state,
                    EFFORT_KEY: effort,
                    EE_POSE_KEY: np.array(snap.ee_pose_si, dtype=np.float32),
                    ACTION_KEY: state.copy(),   # 示教采集：action = 当前 state
                    NEXT_DONE_KEY: np.array([False], dtype=np.bool_),  # 末帧保存前改 True
                    "task": self.args.task,
                }
                self.dataset.add_frame(frame)
                n_frames += 1

                now = time.perf_counter()
                if now >= next_status:
                    next_status = now + 1.0
                    hz = n_frames / max(now - t_start, 1e-6)
                    js = " ".join(f"{v:+.3f}" for v in snap.joints_rad)
                    msg = f"  录制中: {n_frames} 帧  {hz:5.1f} Hz  重复帧={n_dup}  关节(rad)[{js}]"
                    print(msg + " " * 4, end="\r", flush=True)
                # 预览刷新已提前到循环开头（与终端按键合并判定），这里不再重复调用
        finally:
            self.recording = False
            print(" " * 120, end="\r", flush=True)

        elapsed = time.perf_counter() - t_start
        if n_frames == 0:
            warn("本 episode 未录到任何有效帧，已丢弃")
            self.dataset.clear_episode_buffer()
            # clear 内部重置 buffer 读 meta.total_episodes（后台保存可能尚未更新它），
            # 必须覆盖回预分配编号，否则与后台正在保存的 episode 编号冲突
            w = self.dataset.writer
            w.episode_buffer = w._create_episode_buffer(episode_index=self._next_episode_idx)
            return

        hz = n_frames / max(elapsed, 1e-6)
        log(f"录制结束: {n_frames} 帧 / {elapsed:.1f}s = {hz:.1f} Hz（重复帧 {n_dup}）")
        if hz < self.args.fps * 0.9:
            warn(f"实测录制频率 {hz:.1f}Hz 低于数据集 fps={self.args.fps}，"
                 f"数据集时间轴会被压缩；建议用 --fps {int(round(hz))} 重新采集或降低分辨率")
        # 保存与采集【真正并行】：交接 buffer 给后台保存队列后立即返回，
        # 主线程马上可以开始下一轮录制（不再等待编码完成）。
        # episode 编号由主线程预分配（_next_episode_idx），不依赖 meta 状态
        # 1) 取走当前 buffer（含本轮全部帧），交接给后台保存线程
        buf = self.dataset.writer.episode_buffer
        ep_idx = buf["episode_index"]
        ep_no = ep_idx + 1   # 展示用序号（index 从 0 起）
        # 2) LeRobot v3 强制字段：episode 最后一帧 next.done=True（入队前回写）
        if buf.get(NEXT_DONE_KEY):
            buf[NEXT_DONE_KEY][-1] = np.array([True], dtype=np.bool_)
        # 3) 主线程立即以预分配的新编号开新 buffer，后续 add_frame 全部写入新 buffer
        self._next_episode_idx = ep_idx + 1
        self.dataset.writer.episode_buffer = self.dataset.writer._create_episode_buffer(
            episode_index=self._next_episode_idx)
        # 4) 入队（FIFO：保存顺序=录制顺序），主线程零阻塞返回
        self._ensure_save_worker()
        self._save_queue.put((ep_no, ep_idx, buf))
        log(f"episode {ep_no} 已转入后台保存队列（不阻塞采集；当前待保存 {self._save_queue.qsize()} 个）")
        warn("程序【不接管机械臂运动】，请人工将机械臂拖到安全位置并托稳，再按空格继续")

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.open_resources()

        if self.args.check_only:
            log("--check-only：自检完成，未创建数据集、未下发任何运动指令，直接退出")
            return

        # 【步骤1b】自检后回基准位姿 -> 停 1s -> 到采集初始位姿
        step("【步骤1b】自动回位：HOME_BASE -> 停留 1s -> START_POSE")
        self._ensure_can_mode(allow_reset=True, confirm_reset=not self.args.yes)
        self.go_home_then_start()

        # 【步骤2~7】空闲/录制/复位循环：每一轮需要按 3 次空格
        while True:
            # ------------------------------------------------------------
            # 【步骤2】空闲等待第 1 次空格
            # 本阶段程序【不下发任何指令、不录制、不接管运动】
            # ------------------------------------------------------------
            self.cmd.pause()   # 关键：空闲期间停发一切指令（HEARTBEAT_IDLE=False 时彻底静默）
            if self.args.idle_heartbeat:
                log("  --idle-heartbeat 已开启：空闲期间低频下发保持指令用于保活（不是录制）")
                self.cmd.start_hold(START_POSE)
            else:
                log("  空闲待机：程序不下发任何指令、不录制、不接管运动")
            step(f"【步骤2】空闲待机（已完成 {self.episode_count} 个 episode）"
                 f" —— 未录制、未接管")
            self.wait_space("【第 1 次空格】开始 episode 录制；Ctrl+C 安全退出")
            self.cmd.pause()   # 空格后立即彻底停发，确保录制期间零指令

            # ------------------------------------------------------------
            # 【步骤3/4】第 1 次空格：创建数据集并开始录制（只读，不接管）
            # ------------------------------------------------------------
            step("【步骤3/4】开始 episode 录制 —— 纯记录，程序【不接管运动】")
            self._ensure_dataset()
            log("  请按机械臂本体绿色按键进入零重力拖拽示教，再手动拖动机械臂")
            log("  录制中按【第 2 次空格】停止录制（程序仍不会接管运动）")
            self.record_episode()

            # ------------------------------------------------------------
            # 【步骤5】第 2 次空格：仅停止录制，程序仍不接管运动
            # ------------------------------------------------------------
            self.cmd.pause()   # 明确停发指令：人工拖拽期间程序不下发任何命令
            step("【步骤5】已停止录制（episode 已转入后台保存队列）—— 程序【仍不接管运动】")
            if self._save_queue.unfinished_tasks > 0:
                log(f"  后台保存队列中还有 {self._save_queue.unfinished_tasks} 个 episode"
                    f" 在保存/待保存，与当前操作互不影响")
            warn("本阶段程序不下发任何运动指令；请人工把机械臂拖到安全位置并托稳")

            # ------------------------------------------------------------
            # 【步骤6】第 3 次空格：程序开始接管，【直接】下电复位（无二次确认）
            # ------------------------------------------------------------
            self.wait_space("请人工将机械臂拖到安全位置并【托稳】后按【第 3 次空格】——"
                            "按下后程序【立即下电复位】并接管回位，没有二次确认")
            step("【步骤6】程序接管：直接下电复位 -> 切 CAN 控制模式 -> HOME_BASE -> START_POSE")
            self._takeover_after_teach()
            self.go_home_then_start()

    # ------------------------------------------------------------------
    def cleanup_and_hold(self) -> None:
        """异常/中断后的收尾：尽量把资源释放干净。"""
        try:
            self.close_resources()
        except Exception as exc:
            warn(f"释放资源时异常: {exc}")


# ===========================================================================
# 七、CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Piper + 双 D435i 的 LeRobot 拖拽示教数据采集 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--can", default=DEFAULT_CAN, help="CAN 接口名")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobotDataset repo_id（仅本地，不上传）")
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="数据集根目录")
    p.add_argument("--task", default=DEFAULT_TASK, help="写入每帧的 task 文本")
    p.add_argument("--speed", type=int, default=MOVE_SPEED_PCT, help="运动速度百分比 0~100")
    p.add_argument("--fps", type=int, default=RECORD_FPS, help="录制频率与数据集 fps")
    p.add_argument("--preview", action="store_true",
                   help="预览模式：单窗口同时显示两路 RGB 与深度伪彩（需要 GUI 版 opencv，"
                        "窗口显示在 DISPLAY 对应的屏幕上）；预览窗口拿到键盘焦点时，空格同样有效")
    p.add_argument("--idle-heartbeat", action="store_true", default=HEARTBEAT_IDLE,
                   help="空闲(未按空格)时也低频下发保持指令保活；默认关闭，"
                        "即按空格前程序完全不下发指令")
    p.add_argument("--no-camera", action="store_true", help="跳过相机（仅调试机械臂时使用）")
    p.add_argument("--check-only", action="store_true",
                   help="只做硬件自检后退出：不创建数据集、不运动、不修改任何设备状态")
    p.add_argument("--yes", action="store_true",
                   help="跳过【启动阶段】下电复位的人工确认（第 3 次空格的下电复位"
                        "以空格本身为确认，不受此参数影响）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not (0 <= args.speed <= 100):
        sys.exit(f"[FATAL] --speed 必须在 0~100，当前 {args.speed}")
    if not (1 <= args.fps <= 120):
        sys.exit(f"[FATAL] --fps 必须在 1~120，当前 {args.fps}")

    collector = Collector(args)
    try:
        with KeyInput() as key:
            collector.key = key
            collector.run()
    except KeyboardInterrupt:
        log("收到 Ctrl+C，开始安全退出")
    except Exception as exc:
        warn(f"运行异常终止: {exc}")
        traceback.print_exc()
        return 1
    finally:
        collector.cleanup_and_hold()
    log("程序已安全退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())