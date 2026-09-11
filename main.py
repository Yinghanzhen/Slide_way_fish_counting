import yaml
import argparse
import threading
import time
from collections import defaultdict
from tkinter import Tk, Button, Label, Frame, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class FishCounterApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("鱼苗计数系统")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f0f0")

        # 状态变量
        self.running = False
        self.paused = False
        self.cap = None
        self.current_frame = None
        self.fish_count = 0
        self.counted_ids = set()
        self.history = defaultdict(list)
        self.frame_id = 0

        # 虚拟绊线
        self.line_start = (0, 0)
        self.line_end = (0, 0)

        # ========== 提前加载并预热模型 ==========
        self.model = YOLO(self.config['model_path'])
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.track(
            source=dummy,
            tracker=self.config['tracker_config'],
            device=self.config['device'],
            persist=True,
            verbose=False
        )
        print("模型加载完成，程序已就绪！")

        # 构建界面
        self._build_ui()

    def _build_ui(self):
        """构建界面布局"""
        # 顶部：视频显示区域
        self.video_frame = Frame(self.root, bg="black", width=1000, height=600)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)

        self.video_label = Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # 中部：计数显示
        self.count_label = Label(
            self.root,
            text="当前计数: 0 条",
            font=("微软雅黑", 24, "bold"),
            fg="#2c3e50",
            bg="#f0f0f0"
        )
        self.count_label.pack(pady=5)

        # 底部：按钮区域
        btn_frame = Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        self.start_btn = Button(
            btn_frame, text="开始", font=("微软雅黑", 14),
            width=10, command=self.start,
            bg="#27ae60", fg="white", activebackground="#2ecc71"
        )
        self.start_btn.grid(row=0, column=0, padx=8)

        self.pause_btn = Button(
            btn_frame, text="暂停", font=("微软雅黑", 14),
            width=10, command=self.toggle_pause,
            bg="#f39c12", fg="white", activebackground="#f1c40f",
            state="disabled"
        )
        self.pause_btn.grid(row=0, column=1, padx=8)

        self.stop_btn = Button(
            btn_frame, text="停止", font=("微软雅黑", 14),
            width=10, command=self.stop,
            bg="#c0392b", fg="white", activebackground="#e74c3c",
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=2, padx=8)

        self.reset_btn = Button(
            btn_frame, text="清零", font=("微软雅黑", 14),
            width=10, command=self.reset_count,
            bg="#3498db", fg="white", activebackground="#5dade2"
        )
        self.reset_btn.grid(row=0, column=3, padx=8)

    # ==================== 按钮回调 ====================
    def start(self):
        """开始运行"""
        if self.running:
            return
        self.running = True
        self.paused = False
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="暂停")
        self.stop_btn.config(state="normal")

        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def toggle_pause(self):
        """暂停 / 继续"""
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_btn.config(text="继续" if self.paused else "暂停")

    def stop(self):
        """停止运行"""
        self.running = False
        self.paused = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="暂停")
        self.stop_btn.config(state="disabled")

    def reset_count(self):
        """清零计数（带确认框）"""
        if messagebox.askyesno("确认", "确定要将计数清零吗？"):
            self.fish_count = 0
            self.counted_ids.clear()
            self.history.clear()
            self.count_label.config(text="当前计数: 0 条")

    # 核心处理流程
    def _run_pipeline(self):
        """在后台线程中运行检测与追踪"""
        # 打开视频源（模型已在 __init__ 中加载并预热）
        source = self.config['video_path']
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频源: {source}")
            self.stop()
            return

        # 设置摄像头分辨率（仅对摄像头编号生效）
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])

        while self.running and self.cap.isOpened():
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_id += 1
            h, w = frame.shape[:2]

            # 更新绊线位置
            self.line_start = (int(w * self.config['line_margin']),
                               int(h * self.config['line_position']))
            self.line_end = (int(w * (1 - self.config['line_margin'])),
                             int(h * self.config['line_position']))

            # YOLO + ByteTrack 追踪
            results = self.model.track(
                source=frame,
                tracker=self.config['tracker_config'],
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                device=self.config['device'],
                persist=True,
                verbose=False
            )

            # 处理追踪结果
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    current_pos = (center_x, center_y)

                    cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 255, 0), -1)

                    self.history[track_id].append(current_pos)
                    if len(self.history[track_id]) > 2:
                        self.history[track_id].pop(0)

                    if track_id not in self.counted_ids and len(self.history[track_id]) == 2:
                        prev_pos = self.history[track_id][0]
                        curr_pos = self.history[track_id][1]
                        if self._is_intersect(prev_pos, curr_pos, self.line_start, self.line_end):
                            self.fish_count += 1
                            self.counted_ids.add(track_id)

            # 绘制绊线和计数
            cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {self.fish_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 更新界面
            self.current_frame = frame
            self.root.after(0, self._update_ui)

        self.stop()

    def _update_ui(self):
        """更新视频画面和计数标签（在主线程中执行）"""
        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((1000, 600), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.count_label.config(text=f"当前计数: {self.fish_count} 条")

    @staticmethod
    def _is_intersect(line1_start, line1_end, line2_start, line2_end):
        """判断两条线段是否相交"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        def cross(ox, oy, px, py, qx, qy):
            return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

        d1 = cross(x1, y1, x2, y2, x3, y3)
        d2 = cross(x1, y1, x2, y2, x4, y4)
        d3 = cross(x3, y3, x4, y4, x1, y1)
        d4 = cross(x3, y3, x4, y4, x2, y2)

        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def parse_args():
    parser = argparse.ArgumentParser(description="鱼苗计数系统")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    root = Tk()
    app = FishCounterApp(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()