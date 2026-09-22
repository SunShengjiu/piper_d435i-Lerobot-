#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D435i 参数设置与深度读取测试

用途
    1) 按固定参数配置两台 D435i（腕部 wrist + 第三人称 third_person）的 color / depth 流
    2) 逐项写入深度传感器 / 彩色传感器设置，并【回读校验】是否真正生效
    3) 读取 RGB-D 帧，打印深度统计（单位：米），验证深度数据可用

设计说明
    - 全部代码独立开发，未 import / 调用【树彩】目录中任何驱动源码，仅参考其调用思路
    - 参数为固定配置（见下方常量），本脚本不做任何自动调参
    - 设备或固件不支持、或写入后回读不符的参数，一律如实标注，绝不臆造数值

运行
    .venv/bin/python d435i_param_depth_test.py                  # 双相机测试
    .venv/bin/python d435i_param_depth_test.py --preview        # 额外弹出预览窗口
    .venv/bin/python d435i_param_depth_test.py --serial 241122071942 --frames 60
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import pyrealsense2 as rs
except Exception as exc:  # 依赖缺失时给出明确提示，而不是抛栈
    sys.exit(f"[FATAL] 依赖导入失败: {exc}\n请先执行: uv pip install pyrealsense2 opencv-python numpy")

# --------------------------------------------------------------------------
# 固定配置区（全部为固定参数，不自动调整）
# --------------------------------------------------------------------------

# 流参数：color 1280x720@30 bgr8 / depth 1280x720@30 z16
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
COLOR_FORMAT_NAME, DEPTH_FORMAT_NAME = "bgr8", "z16"

# 公共相机参数
WARMUP_FRAMES = 30          # 启动后丢弃的预热帧数
FRAME_TIMEOUT_MS = 5000     # wait_for_frames 超时(ms)

# 深度传感器设置（两台完全相同）
# 顺序有意义：先设 visual_preset，再设其余项，避免 preset 覆盖后续写入
DEPTH_OPTIONS: Dict[str, float] = {
    "visual_preset": 0.0,
    "enable_auto_exposure": 1.0,
    "emitter_enabled": 1.0,
    "laser_power": 150.0,
    "auto_exposure_limit_toggle": 0.0,
    "auto_gain_limit_toggle": 0.0,
    "emitter_on_off": 0.0,
    "emitter_always_on": 0.0,
}

# 彩色传感器设置（两台仅 white_balance 不同）
# 顺序有意义：先关自动曝光/自动白平衡，再写手动值，否则手动值会被自动算法覆盖
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

# 两台相机的固定身份：序列号 + 各自的白平衡
CAMERAS: List[Dict[str, Any]] = [
    {"name": "wrist", "serial": "241122071942", "white_balance": 6150.0},
    {"name": "third_person", "serial": "042222071742", "white_balance": 6200.0},
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt(v: Any) -> str:
    return f"{v:.1f}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------
# 相机封装
# --------------------------------------------------------------------------
class D435iCamera:
    """单台 D435i：按序列号绑定设备，开启 color+depth 流，读写传感器参数，释放。"""

    def __init__(self, name: str, serial: str, white_balance: float) -> None:
        self.name = name
        self.serial = serial
        self.white_balance = white_balance
        self.pipeline: Optional[Any] = None
        self.profile: Optional[Any] = None
        self.depth_sensor: Optional[Any] = None
        self.color_sensor: Optional[Any] = None
        self.depth_scale: Optional[float] = None
        self._report: Dict[str, Dict[str, str]] = {"depth": {}, "color": {}}

    # ---------------- 设备发现 ----------------
    @staticmethod
    def list_devices() -> List[Dict[str, str]]:
        """枚举当前所有 RealSense 设备（序列号 / 名称 / 固件 / USB 类型）。"""
        found: List[Dict[str, str]] = []
        for dev in rs.context().query_devices():
            found.append(
                {
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "name": dev.get_info(rs.camera_info.name),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                    "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
                }
            )
        return found

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        """按固定流参数开启 pipeline，并拿到 depth / color 传感器句柄。"""
        cfg = rs.config()
        # 不指定 serial 时 rs 会自动挑选设备，因此必须显式绑定，避免张冠李戴
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
        cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(cfg)

        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        self.color_sensor = dev.first_color_sensor() if dev.sensors else None
        # 深度图原始值 -> 米 的换算系数（D435i 通常为 0.001）
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # 校验实际协商成功的流参数，确认不是降级后的分辨率/帧率
        for stream, want_fmt in ((rs.stream.color, COLOR_FORMAT_NAME), (rs.stream.depth, DEPTH_FORMAT_NAME)):
            is_color = stream == rs.stream.color
            sp = self.profile.get_stream(stream).as_video_stream_profile()
            # str(rs.format.bgr8) 为 "format.bgr8"，取末段再比，否则会把正确的格式误判为"不符"
            got_fmt = str(sp.format()).split(".")[-1]
            want = (
                (COLOR_W, COLOR_H, COLOR_FPS, want_fmt) if is_color
                else (DEPTH_W, DEPTH_H, DEPTH_FPS, want_fmt)
            )
            got = (sp.width(), sp.height(), sp.fps(), got_fmt)
            flag = "OK" if got == want else f"!! 与请求不符(请求 {want})"
            log(f"  {self.name:<12} {'color' if is_color else 'depth'} 流协商: "
                f"{got[0]}x{got[1]}@{got[2]} {got[3]}  [{flag}]")

        log(f"  {self.name:<12} 已启动 serial={self.serial} depth_scale={self.depth_scale}")

    def read(self) -> Optional[Tuple[Any, Any, int, int]]:
        """读取一帧对齐的 frameset，返回 (color_bgr, depth_raw, color_no, depth_no)。"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
        except RuntimeError:
            return None
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return None
        color = np.asanyarray(cf.get_data())
        depth = np.asanyarray(df.get_data())
        return color, depth, cf.get_frame_number(), df.get_frame_number()

    # ---------------- 参数写入 + 回读校验 ----------------
    def _apply(self, sensor: Any, which: str, options: Dict[str, float]) -> None:
        """逐项 set_option 后立即 get_option 回读，记录：实际值 / 是否生效 / 是否支持。"""
        if sensor is None:
            log(f"  {self.name:<12} [{which}] 未找到该传感器，全部参数无法写入")
            return
        # 动态获取该传感器支持的全部选项名，便于判定"设备不支持"
        try:
            supported_names = {o.name for o in sensor.get_supported_options()}
        except Exception:
            supported_names = set()

        for name, want in options.items():
            if name not in supported_names:
                self._report[which][name] = "设备/SDK 不支持"
                continue
            opt = getattr(rs.option, name, None)
            if opt is None:
                self._report[which][name] = "SDK 无此选项枚举"
                continue
            try:
                sensor.set_option(opt, want)
            except Exception as exc:
                # 部分固件（实测 5.13.0.55）对"写入与当前值相同的冗余写"会报 hwmon -9。
                # 此时目标值其实已满足，必须回读确认，不能一律判为失败。
                try:
                    cur = float(sensor.get_option(opt))
                except Exception:
                    cur = None
                if cur is not None and abs(cur - want) < 1e-3:
                    self._report[which][name] = f"已为目标值(写入被固件拒绝: {exc})"
                else:
                    self._report[which][name] = f"写入异常: {exc}"
                continue
            try:
                got = sensor.get_option(opt)
            except Exception as exc:
                self._report[which][name] = f"回读异常: {exc}"
                continue
            # 浮点选项按极小容差比较，避免 149.99999 被误判为不符
            self._report[which][name] = "一致" if abs(float(got) - want) < 1e-3 else f"写{want:g}/读到{got:g}"

    def apply_options(self) -> None:
        """写入深度与彩色全部固定参数（彩色额外注入本机 white_balance）。"""
        color_opts = dict(COLOR_OPTIONS_COMMON)
        color_opts["white_balance"] = self.white_balance
        self._apply(self.depth_sensor, "depth", DEPTH_OPTIONS)
        self._apply(self.color_sensor, "color", color_opts)

    def print_report(self) -> Tuple[int, int, int]:
        """打印参数校验表，返回 (一致数, 需注意数, 异常数)。"""
        print(f"\n--- {self.name} (serial={self.serial}) 参数校验 ---")
        ok = warn = bad = 0
        for which in ("depth", "color"):
            print(f"  [{which}]")
            for name, status in self._report[which].items():
                if status == "一致":
                    ok += 1
                    mark = "OK  "
                elif status.startswith("已为目标值"):
                    warn += 1          # 目标状态已满足，但写入通道被固件拒绝
                    mark = "提示"
                else:
                    bad += 1
                    mark = "异常"
                print(f"    {mark} {name:<30} {status}")
        return ok, warn, bad

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        log(f"  {self.name:<12} 已停止并释放")


# --------------------------------------------------------------------------
# 深度数据检查
# --------------------------------------------------------------------------
def depth_stats(depth: Any, scale: float) -> Dict[str, Any]:
    """统计深度帧（原始 uint16 -> 米）。

    只统计真实有效像素，不做任何填充或猜测：
      0     = 该像素无有效测量（RealSense 约定）
      65535 = uint16 饱和哨兵值(实测存在，约 0.1%)，换算得 65.535 m，非真实距离，一并剔除
    """
    meters = depth.astype(np.float32) * scale
    n_sentinel = int((depth == 65535).sum())
    valid = (depth > 0) & (depth != 65535)
    n_valid = int(valid.sum())
    st: Dict[str, Any] = {
        "shape": depth.shape,
        "dtype": str(depth.dtype),
        "scale": scale,
        "n_valid": n_valid,
        "valid_ratio": n_valid / depth.size,
        "n_zero": int((depth == 0).sum()),
        "n_sentinel": n_sentinel,
        "sentinel_ratio": n_sentinel / depth.size,
    }
    if n_valid == 0:
        st["note"] = "整帧无有效深度"
        return st
    vals = meters[valid]
    st["min_m"] = float(vals.min())
    st["max_m"] = float(vals.max())
    st["median_m"] = float(np.median(vals))
    st["p99_m"] = float(np.percentile(vals, 99))
    h, w = depth.shape
    # 中心 5x5 区域中位数：验证正前方确实有可测深度
    c = meters[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3]
    c_valid = c[(c > 0) & (c * 1.0 < 65.535)]
    st["center_m"] = float(np.median(c_valid)) if c_valid.size else None
    return st


def render_preview(color: Any, depth: Any, scale: float, name: str) -> Any:
    """把彩色图与深度伪彩图横向拼接，便于人眼快速确认两路都正常。"""
    import cv2

    d = np.clip(depth.astype(np.float32) * scale * 1000.0 / 3.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    return np.hstack([color, vis])


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    import cv2

    targets = CAMERAS
    if args.serial:
        targets = [c for c in CAMERAS if c["serial"] == args.serial]
        if not targets:
            log(f"序列号 {args.serial} 不在固定配置表中")
            return 2

    log("枚举 RealSense 设备 ...")
    devices = D435iCamera.list_devices()
    if not devices:
        log("未发现任何 RealSense 设备，退出")
        return 2
    for d in devices:
        log(f"  发现 serial={d['serial']} {d['name']} fw={d['firmware']} usb={d['usb']}")

    available = {d["serial"] for d in devices}
    missing = [c["serial"] for c in targets if c["serial"] not in available]
    if missing:
        log(f"以下序列号未找到，退出: {missing}")
        return 2

    cams: List[D435iCamera] = []
    try:
        # ---- 1) 依次启动 ----
        log("启动相机 ...")
        for c in targets:
            cam = D435iCamera(c["name"], c["serial"], c["white_balance"])
            cam.start()
            cams.append(cam)

        # ---- 2) 写入固定参数并回读校验 ----
        log("写入固定参数并回读校验 ...")
        for cam in cams:
            cam.apply_options()

        total_ok = total_warn = total_bad = 0
        for cam in cams:
            ok, warn, bad = cam.print_report()
            total_ok += ok
            total_warn += warn
            total_bad += bad

        # ---- 3) 预热 ----
        log(f"预热 {WARMUP_FRAMES} 帧 ...")
        for _ in range(WARMUP_FRAMES):
            for cam in cams:
                cam.read()

        # ---- 4) 读取并检查深度 ----
        log(f"读取 {args.frames} 组 RGB-D 帧 ...")
        got = {c.name: 0 for c in cams}
        last: Dict[str, Tuple[Any, Any]] = {}
        # 逐帧累积有效像素占比，用多帧统计代替单帧孤证
        agg: Dict[str, List[float]] = {c.name: [] for c in cams}
        for i in range(args.frames):
            for cam in cams:
                r = cam.read()
                if r is None:
                    continue
                color, depth, cn, dn = r
                got[cam.name] += 1
                last[cam.name] = (color, depth)
                agg[cam.name].append(
                    int(((depth > 0) & (depth != 65535)).sum()) / depth.size
                )
                if i == 0:
                    st = depth_stats(depth, cam.depth_scale)
                    print(f"\n--- {cam.name} 深度统计 (第 1 帧) ---")
                    print(f"  color: {color.shape} {color.dtype} | depth_raw: {st['shape']} {st['dtype']}")
                    print(f"  depth_scale = {st['scale']}")
                    print(f"  有效像素 {st['n_valid']} / {depth.size}  ({st['valid_ratio']*100:.1f}%)"
                          f" | 0值 {st['n_zero']} ({st['n_zero']/depth.size*100:.1f}%)"
                          f" | 65535哨兵 {st['n_sentinel']} ({st['sentinel_ratio']*100:.2f}%)")
                    if "note" in st:
                        print(f"  {st['note']}")
                    else:
                        print(f"  最近 {st['min_m']:.3f} m | 最远 {st['max_m']:.3f} m | 中位 {st['median_m']:.3f} m | p99 {st['p99_m']:.3f} m")
                        ctr = f"{st['center_m']:.3f} m" if st["center_m"] is not None else "中心无有效值"
                        print(f"  画面中心 5x5 中位深度 = {ctr}")
                    print(f"  color/depth 帧号 = {cn} / {dn}")

            # ---- 5) 可选预览 ----
            if args.preview:
                tiles = [render_preview(last[c.name][0], last[c.name][1], c.depth_scale, c.name) for c in cams if c.name in last]
                if tiles:
                    cv2.imshow("D435i  color | depth", np.hstack(tiles) if len(tiles) > 1 else tiles[0])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            time.sleep(1.0 / COLOR_FPS)

        # ---- 6) 汇总 ----
        print("\n================ 结果汇总 ================")
        for cam in cams:
            exp = args.frames
            ratios = agg[cam.name]
            vr = f"{float(np.median(ratios))*100:.1f}%" if ratios else "无数据"
            print(f"  {cam.name:<12} 成功读取 {got[cam.name]}/{exp} 组 RGB-D"
                  f"{'  [OK]' if got[cam.name] == exp else '  [有丢帧]'}"
                  f" | 多帧有效深度占比中位 {vr}")
        print(f"  参数校验: 一致 {total_ok} 项，提示 {total_warn} 项，异常 {total_bad} 项")
        if total_warn:
            print("  「提示」= 目标值已满足但写入通道被固件拒绝（实测 5.13.0.55 对冗余写返回 hwmon -9），无需处理。")
        if total_bad:
            print("  「异常」= 设备不支持或写入后被改写，需人工确认。")
        print("=========================================")
        return 0 if all(got[c.name] == args.frames for c in cams) else 1

    except KeyboardInterrupt:
        log("收到 Ctrl+C，开始优雅释放 ...")
        return 130
    finally:
        for cam in cams:
            cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log("相机资源已全部释放")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D435i 固定参数写入 + 深度读取测试")
    p.add_argument("--serial", default=None, help="只测试指定序列号（默认两台都测）")
    p.add_argument("--frames", type=int, default=30, help="每台相机读取的 RGB-D 帧数（默认 30）")
    p.add_argument("--preview", action="store_true", help="弹出 cv2 预览窗口（需图形界面）")
    return p


def main() -> int:
    args = build_parser().parse_args()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())