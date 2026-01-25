import os
import traceback
import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from ttkthemes import ThemedStyle
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import glob
import json
import random
import colorsys
import pickle
import numpy as np
import datetime
import torch
import subprocess
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import shutil
import time
import sam2

# 导入视频分割模块
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# PCA 分析
def pca_analysis(point_cloud):
    if len(point_cloud) < 3:
        return np.eye(3), point_cloud, np.zeros(3)
    centroid = np.mean(point_cloud, axis=0)
    centered = point_cloud - centroid
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    order = eigenvalues.argsort()[::-1]
    return eigenvectors[:, order], centroid, eigenvalues[order]

# 计算 OBB
def calculate_obb(point_cloud):
    if len(point_cloud) < 3:
        return np.zeros(3), np.zeros(3), np.eye(3)

    eigenvectors, centroid, eigenvalues = pca_analysis(point_cloud)
    rotated_points = np.dot(point_cloud - centroid, eigenvectors)
    min_coords = np.min(rotated_points, axis=0)
    max_coords = np.max(rotated_points, axis=0)
    dimensions = max_coords - min_coords
    return dimensions, centroid, eigenvectors

# 统计滤波
def statistical_filter(point_cloud, mean_k=50, std_dev_mul=1.0):
    if len(point_cloud) < mean_k:
        return point_cloud
    distances = np.linalg.norm(point_cloud[:, None] - point_cloud, axis=2)
    mean_dist = np.mean(distances, axis=1)
    std_dist = np.std(mean_dist)
    threshold = mean_dist.mean() + std_dev_mul * std_dist
    return point_cloud[mean_dist < threshold]

# 半径滤波
def radius_filter(point_cloud, radius=0.5, min_neighbors=5):
    """半径滤波"""
    inliers = []
    for point in point_cloud:
        dists = np.linalg.norm(point_cloud - point, axis=1)
        if np.sum(dists < radius) >= min_neighbors:
            inliers.append(point)
    return np.array(inliers)

# 获取感兴趣区域的点云
def get_roi_point_cloud(point_cloud, calib, mask, image):
    height, width = image.shape[:2]
    P = calib['P2']
    R0_rect = calib['R_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    points_homogeneous = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
    points_camera = np.dot(points_homogeneous, Tr_velo_to_cam.T)[:, :3]
    points_camera = np.dot(points_camera, R0_rect.T)

    valid_depth = points_camera[:, 2] > 1e-6
    points_2d = (np.dot(points_camera[valid_depth], P[:, :3].T) + P[:, 3]) / points_camera[valid_depth, 2, np.newaxis]
    points_2d = np.round(points_2d[:, :2]).astype(int)

    valid_mask = np.zeros(valid_depth.sum(), dtype=bool)
    for i, (x, y) in enumerate(points_2d):
        if 0 <= x < width and 0 <= y < height and mask[y, x] == 1:
            valid_mask[i] = True

    roi_indices = np.where(valid_depth)[0][valid_mask]
    return point_cloud[roi_indices][:, :3]

# 计算 ROI 尺寸
def calculate_roi_dimensions(roi_point_cloud):
    if roi_point_cloud.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    filtered_roi = statistical_filter(roi_point_cloud)
    filtered_roi = radius_filter(filtered_roi)

    if len(filtered_roi) < 2:
        x_min = 0.0
        y_min = 0.0
        z_min = 0.0
        x_max = 0.0
        y_max = 0.0
        z_max = 0.0
    else:
        x_min = np.min(filtered_roi[:, 0])
        y_min = np.min(filtered_roi[:, 1])
        z_min = np.min(filtered_roi[:, 2])
        x_max = np.max(filtered_roi[:, 0])
        y_max = np.max(filtered_roi[:, 1])
        z_max = np.max(filtered_roi[:, 2])

    dimensions, centroid, R = calculate_obb(filtered_roi)
    distances = np.linalg.norm(filtered_roi[:, :2], axis=1) if filtered_roi.size > 0 else np.array([])
    depth = np.mean(distances[:int(len(distances) * 0.1)]) if len(distances) > 0 else 0.0

    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    return (
        float(dimensions[0]),
        float(dimensions[1]),
        float(dimensions[2]),
        float(depth),
        float(x_min),
        float(y_min),
        float(z_min),
        float(dx),
        float(dy),
        float(dz)
    )

class SAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 交互式图像分割工具")
        self.root.geometry("1200x830")
        self.root.minsize(900, 700)

        self.style = ThemedStyle(self.root)
        self.style.set_theme("arc")
        self.style.configure("TFrame", background="#f5f6f7")
        self.style.configure("TLabel", background="#f5f6f7")
        self.style.configure("TLabelframe", background="#f5f6f7")
        self.style.configure("TLabelframe.Label", background="#f5f6f7")
        self.style.configure("TButton", font=("Arial", 10), padding=5)
        self.style.configure("TNotebook", background="#f0f0f0")
        self.style.configure("TNotebook.Tab", padding=(10, 5), font=("Arial", 10, "bold"))
        
        # 初始化变量
        self.model_config_path = None
        self.model_weights_path = None
        self.model_config_name = None  # hydra用的config_name
        self.model_weights_relpath = None  # 新增：相对路径
        self.sam2_model = None
        self.image_predictor = None  # 图像预测器
        self.video_predictor = None  # 视频预测器

        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.image_list = []
        self.current_image_index = 0
        self.scale_factor = 1.0

        self.image_annotations = {}
        self.image_masks = {}
        
        # 当前图片的交互数据
        self.positive_points = []
        self.negative_points = []
        self.bboxes = []
        self.temp_bbox = None
        self.bbox_start = None
        self.current_mode = "positive"
        self.operation_history = []

        self.colors = self.generate_distinct_colors(36)
        self.color_index = 0
        
        self.selected_mask_index = None
        
        self.cloud_file = None
        self.imu_file = None
        self.calib_file = None

        self.point_cloud = None
        self.calib_params = {}
        self.imu_data = {}
        
        # 视频相关变量
        self.is_video = False
        self.video_frames_dir = None
        self.video_frame_names = []
        self.video_segments = {}   # 存储视频分割结果
        self.first_frame_cid = ""  # 记录第一帧的CID
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        
        # GPU优化配置
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print("\nMPS support is preliminary. Performance may vary.")
        
        self.setup_ui()
    
    def generate_distinct_colors(self, n):
        """生成一组视觉上可区分的颜色"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + random.random() * 0.3
            value = 0.8 + random.random() * 0.2
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors
    
    def get_next_color(self):
        """获取下一个颜色"""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
        
    def setup_ui(self):
        """设置用户界面布局"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 配置行列权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1) 
        main_frame.rowconfigure(2, weight=0)
        
        # -------------------- 顶部：图片导航和状态区 --------------------
        top_frame = ttk.Frame(main_frame, padding=5)
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_frame.columnconfigure(1, weight=1)  # 中间区域可扩展
        
        # 图片导航按钮
        nav_frame = ttk.Frame(top_frame)
        nav_frame.grid(row=0, column=0, sticky="w")
        
        ttk.Button(nav_frame, text="◀ 上一张", width=10, command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="下一张 ▶", width=10, command=self.next_image).pack(side=tk.LEFT, padx=2)
        
        # 图片信息
        info_frame = ttk.Frame(top_frame)
        info_frame.grid(row=0, column=1, sticky="w", padx=10)
        
        self.image_info_label = ttk.Label(info_frame, text="未加载图片", font=("Arial", 10, "bold"))
        self.image_info_label.pack(anchor=tk.W)
        
        # 标注信息
        self.points_info_label = ttk.Label(info_frame, text="正向点: 0 | 负向点: 0 | 边界框: 0")
        self.points_info_label.pack(anchor=tk.W)
        
        # 顶部模型状态
        status_frame = ttk.Frame(top_frame)
        status_frame.grid(row=0, column=2, sticky="e")
        ttk.Label(status_frame, text="模型状态:").pack(side=tk.LEFT)
        self.model_status_var = tk.StringVar(value="未加载")
        self.model_status_label = ttk.Label(status_frame, textvariable=self.model_status_var, foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=5)
        
        # -------------------- 中部：图片显示区 --------------------
        mid_frame = ttk.Frame(main_frame)
        mid_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        mid_frame.columnconfigure(0, weight=1)
        mid_frame.rowconfigure(0, weight=1)
        
        # 画布 + 滚动条
        canvas_frame = ttk.Frame(mid_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # 使用grid布局滚动条和画布
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # 绑定画布大小变化事件
        canvas_frame.bind("<Configure>", self.on_canvas_resize)
        
        # -------------------- 底部：功能区（横向排列） --------------------
        bottom_frame = ttk.Frame(main_frame, padding=2)
        bottom_frame.grid(row=2, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)
        bottom_frame.columnconfigure(3, weight=1)
        bottom_frame.columnconfigure(4, weight=1)
        
        # ========== 功能区1：媒体管理 ==========
        frame1 = ttk.LabelFrame(bottom_frame, text="配置模块", padding=4)
        frame1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        frame1.columnconfigure(0, weight=1)
        
        photo_frame = ttk.LabelFrame(frame1, text="媒体管理", padding=2)
        photo_frame.pack(fill=tk.X, padx=2, pady=2)

        ttk.Button(photo_frame, text="添加单张图片", command=self.add_single_image).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(photo_frame, text="添加图片文件夹", command=self.load_image_folder).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(photo_frame, text="添加视频文件", command=self.add_video_file).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(photo_frame, text="清空媒体列表", command=self.clear_image_list).pack(fill=tk.X, padx=2, pady=2)
        
        model_get_frame = ttk.LabelFrame(frame1, text="模型管理", padding=2)
        model_get_frame.pack(fill=tk.X, padx=2, pady=2)
        # 模型加载
        ttk.Button(model_get_frame, text="选择配置文件 (.yaml)", command=self.select_config_file).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(model_get_frame, text="选择模型文件 (.pt)", command=self.select_model_file).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(model_get_frame, text="清除模型选择", command=self.clear_model_selection).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(model_get_frame, text="加载SAM2模型", command=self.load_sam2_model).pack(fill=tk.X, padx=2, pady=2)
        # 新增：自动视频分割按钮
        ttk.Button(model_get_frame, text="自动视频分割", command=self.auto_segment_video).pack(fill=tk.X, padx=2, pady=2)
        
        # ========== 功能区2：标注操作 ==========
        frame2 = ttk.LabelFrame(bottom_frame, text="标注操作", padding=4)
        frame2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        frame2.columnconfigure(0, weight=1)
        
        # 交互模式
        mode_frame = ttk.LabelFrame(frame2, text="交互模式", padding=2)
        mode_frame.pack(fill=tk.X, padx=2, pady=2)
        
        self.mode_var = tk.StringVar(value="positive")
        ttk.Radiobutton(mode_frame, text="正向点", variable=self.mode_var, value="positive", command=self.change_mode).pack(anchor=tk.W, padx=2, pady=1)
        ttk.Radiobutton(mode_frame, text="负向点", variable=self.mode_var, value="negative", command=self.change_mode).pack(anchor=tk.W, padx=2, pady=1)
        ttk.Radiobutton(mode_frame, text="边界框", variable=self.mode_var, value="bbox", command=self.change_mode).pack(anchor=tk.W, padx=2, pady=1)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(frame2, text="操作", padding=2)
        action_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Button(action_frame, text="撤销", command=self.undo_operation).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(action_frame, text="清除标注", command=self.clear_current_annotations).pack(fill=tk.X, padx=2, pady=1)
        
        # 推理按钮
        inference_frame = ttk.LabelFrame(frame2, text="推理", padding=2)
        inference_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Button(inference_frame, text="运行推理", command=self.run_current_inference).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(inference_frame, text="批量推理", command=self.run_batch_inference).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(inference_frame, text="分割整个视频", command=self.segment_entire_video).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(inference_frame, text="重置视频状态", command=self.reset_video_state).pack(fill=tk.X, padx=2, pady=1)
        # ========== 功能区3：掩码管理 ==========
        frame3 = ttk.LabelFrame(bottom_frame, text="掩码管理", padding=4)
        frame3.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        frame3.columnconfigure(0, weight=1)
        frame3.rowconfigure(0, weight=1)
        
        # 掩码列表
        mask_frame = ttk.Frame(frame3)
        mask_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        columns = ("id", "center", "cid")
        self.mask_tree = ttk.Treeview(
            mask_frame, 
            columns=columns, 
            show="headings", 
            height=10,
            selectmode="browse"
        )
        
        # 设置列标题
        self.mask_tree.heading("id", text="掩码编号")
        self.mask_tree.heading("center", text="掩码中心")
        self.mask_tree.heading("cid", text="CID")
        
        # 设置列宽
        self.mask_tree.column("id", width=100, anchor=tk.CENTER)
        self.mask_tree.column("center", width=120, anchor=tk.CENTER)
        self.mask_tree.column("cid", width=100, anchor=tk.CENTER)
        
        # 添加滚动条
        tree_scroll = ttk.Scrollbar(mask_frame, orient=tk.VERTICAL, command=self.mask_tree.yview)
        self.mask_tree.configure(yscrollcommand=tree_scroll.set)
        
        # 布局
        self.mask_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        
        # 绑定选择事件
        self.mask_tree.bind('<<TreeviewSelect>>', self.on_mask_selected)
        
        # 绑定CID列的编辑事件
        self.mask_tree.bind("<Double-1>", self.on_cid_double_click)
        
        # 掩码操作按钮
        mask_btn_frame = ttk.Frame(frame3)
        mask_btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(mask_btn_frame, text="删除选中", command=self.delete_selected_mask).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(mask_btn_frame, text="清除所有", command=self.clear_all_masks).pack(side=tk.LEFT, expand=True, padx=2)
        
        # ========== 功能区4：保存选项 ==========
        frame4 = ttk.LabelFrame(bottom_frame, text="保存选项", padding=4)
        frame4.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)
        frame4.columnconfigure(0, weight=1)
        
        # 掩码保存
        ttk.Button(frame4, text="导出掩码为图像", command=self.export_masks_as_images).pack(fill=tk.X, padx=2, pady=2)
        
        # PKL保存
        ttk.Button(frame4, text="保存当前PKL", command=self.save_current_pkl).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(frame4, text="批量保存PKL", command=self.save_all_pkl).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(frame4, text="保存视频分割结果", command=self.save_video_segments).pack(fill=tk.X, padx=2, pady=2)
        
        # ========== 功能区5：点云/IMU ==========
        frame5 = ttk.LabelFrame(bottom_frame, text="点云/IMU", padding=4)
        frame5.grid(row=0, column=4, sticky="nsew", padx=2, pady=2)
        frame5.columnconfigure(0, weight=1)
        
        # 文件选择
        ttk.Button(frame5, text="选择点云文件 (.bin)", command=self.select_cloud_file).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(frame5, text="选择IMU文件 (.txt)", command=self.select_imu_file).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(frame5, text="选择校准文件 (.txt)", command=self.select_calib_file).pack(fill=tk.X, padx=2, pady=2)
        
        # 状态显示
        status_frame = ttk.LabelFrame(frame5, text="文件状态", padding=2)
        status_frame.pack(fill=tk.X, padx=2, pady=2)
        
        self.cloud_status_var = tk.StringVar(value="未选择点云文件")
        cloud_label = ttk.Label(status_frame, textvariable=self.cloud_status_var, background="#f0f0f0", padding=2)
        cloud_label.pack(fill=tk.X, padx=1, pady=1)
        
        self.imu_status_var = tk.StringVar(value="未选择IMU文件")
        imu_label = ttk.Label(status_frame, textvariable=self.imu_status_var, background="#f0f0f0", padding=2)
        imu_label.pack(fill=tk.X, padx=1, pady=1)
        
        self.calib_status_var = tk.StringVar(value="未选择校准文件")
        calib_label = ttk.Label(status_frame, textvariable=self.calib_status_var, background="#f0f0f0", padding=2)
        calib_label.pack(fill=tk.X, padx=1, pady=1)
        
        # -------------------- 状态栏 --------------------
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请加载媒体文件")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 新增：底部右侧模型和配置文件状态区
        bottom_status_frame = ttk.Frame(self.root)
        bottom_status_frame.place(relx=1.0, rely=1.0, anchor="se")
        # 配置文件状态
        self.model_config_var = tk.StringVar(value="未选择配置文件")
        self.model_weights_var = tk.StringVar(value="未选择模型文件")
        self.model_config_label = ttk.Label(bottom_status_frame, textvariable=self.model_config_var, width=28, anchor="e", background="#f0f0f0", padding=2)
        self.model_config_label.pack(side=tk.LEFT, padx=(0, 16), pady=2)
        self.model_weights_label = ttk.Label(bottom_status_frame, textvariable=self.model_weights_var, width=28, anchor="e", background="#f0f0f0", padding=2)
        self.model_weights_label.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_canvas_resize(self, event):
        """当画布大小改变时重新显示图片"""
        if self.current_image is not None:
            self.display_current_image()
    
    # 加载点云数据
    def select_cloud_file(self):
        """选择点云文件"""
        file_path = filedialog.askopenfilename(
            title="选择点云文件",
            filetypes=[("点云文件", "*.bin")]
        )
        if file_path:
            self.point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            self.cloud_file = file_path
            self.cloud_status_var.set(f"已选择: {os.path.basename(file_path)}")
            self.status_var.set(f"点云文件已设置: {file_path}")
    
    # 加载IMU数据
    def select_imu_file(self):
        """选择IMU文件"""
        file_path = filedialog.askopenfilename(
            title="选择IMU文件",
            filetypes=[("文本文件", "*.txt")]
        )
        if file_path:
            self.imu_file = file_path
            with open(file_path, 'r') as file:
                data = list(map(float, file.read().strip().split()))
            if len(data) != 30:
                raise ValueError("数据长度不符合要求，应为30个参数")

            self.imu_data = {
                "position": [data[0], data[1], data[2]],   # 经度, 纬度, 高度
                "imu_angle": [data[3], data[4], data[5]],  # 滚转角, 俯仰角, 偏航角
                "imu_acc": [data[11], data[12], data[13]]  # X加速度, Y加速度, Z加速度
            }
            self.imu_status_var.set(f"已选择: {os.path.basename(file_path)}")
            self.status_var.set(f"IMU文件已设置: {file_path}")
    
    # 加载校准文件
    def select_calib_file(self):
        """选择校准文件"""
        file_path = filedialog.askopenfilename(
            title="选择校准文件",
            filetypes=[("文本文件", "*.txt")]
        )
        if file_path:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 先尝试按冒号分割
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                    else:
                        # 否则按空格分割
                        parts = line.split()
                        key = parts[0]
                        value = ' '.join(parts[1:])
                    data = np.array(list(map(float, value.split())))

                    if key.startswith('P'):
                        self.calib_params[key] = data.reshape(3, 4)
                    elif key == 'R_rect':
                        self.calib_params[key] = data.reshape(3, 3)
                    elif key == 'Tr_velo_to_cam':
                        self.calib_params[key] = np.vstack((data.reshape(3, 4), [0, 0, 0, 1]))
            self.calib_file = file_path
            self.calib_status_var.set(f"已选择: {os.path.basename(file_path)}")
            self.status_var.set(f"校准文件已设置: {file_path}")
    
    def get_config_name_for_hydra(self, abs_path):
        """
        如果abs_path在sam2包的configs/目录下，返回相对路径；否则直接用原文件名复制到sam2/configs/下（如有重名则覆盖），返回相对路径。
        """
        sam2_configs_dir = os.path.join(sam2.__path__[0], 'configs')
        abs_path = os.path.abspath(abs_path)
        if abs_path.startswith(sam2_configs_dir):
            # 在包内 configs 目录下
            return os.path.relpath(abs_path, sam2.__path__[0])
        else:
            os.makedirs(sam2_configs_dir, exist_ok=True)
            filename = os.path.basename(abs_path)
            dst_path = os.path.join(sam2_configs_dir, filename)
            shutil.copy(abs_path, dst_path)  # 直接覆盖
            return os.path.relpath(dst_path, sam2.__path__[0])
    
    # TODO 待修改模块，选择自定义位置的sam2配置文件会报错
    def select_config_file(self):
        """选择配置文件(yaml)"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件(yaml)",
            filetypes=[("配置文件", "*.yaml")]
        )
        if file_path:
            self.model_config_path = file_path
            self.model_config_name = self.get_config_name_for_hydra(file_path)
            self.model_config_var.set(f"已选择: {os.path.basename(file_path)}")
            self.status_var.set(f"配置文件已设置: {file_path}")
    
    # TODO 待修改模块，选择自定义位置的sam2模型文件会报错
    def select_model_file(self):
        """选择模型文件(pt)"""
        file_path = filedialog.askopenfilename(
            title="选择SAM2模型文件",
            filetypes=[("模型文件", "*.pt")]
        )
        if file_path:
            self.model_weights_path = file_path
            self.model_weights_var.set(f"已选择: {os.path.basename(file_path)}")
            self.status_var.set(f"模型文件已设置: {file_path}")
    
    # TODO 加载会调用build_sam2函数，这个函数只允许使用相对路径
    def load_sam2_model(self):
        """加载SAM2模型（同时加载图片和视频模型）"""
        try:
            self.model_status_var.set("正在加载模型...")
            self.root.update()
            # 自动加载默认模型和配置文件
            if not self.model_config_path or not self.model_weights_path or not self.model_config_name:
                # 默认配置文件（包内）
                default_config = os.path.join(sam2.__path__[0], "configs", "sam2.1_hiera_l.yaml")
                self.model_config_path = default_config
                self.model_config_name = self.get_config_name_for_hydra(default_config)
                self.model_config_var.set(f"默认: sam2.1_hiera_l.yaml")
                # 默认权重文件（项目根目录下）
                project_root = os.path.abspath(os.path.dirname(__file__)).split("notebooks")[0]
                default_weights = os.path.join(project_root, "checkpoints", "sam2.1_hiera_large.pt")
                self.model_weights_path = default_weights
                self.model_weights_var.set(f"默认: sam2.1_hiera_large.pt")
                self.status_var.set("使用默认模型和配置文件")

            config_name = self.model_config_name
            weights_path = self.model_weights_path
            # 加载图片模型
            self.sam2_model = build_sam2(config_name, weights_path, device=self.device)
            self.image_predictor = SAM2ImagePredictor(self.sam2_model)
            # 加载视频模型
            self.video_predictor = build_sam2_video_predictor(
                config_name,
                weights_path,
                device=self.device
            )
            # 新增：自动掩码生成器
            self.automatic_mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            self.model_status_var.set("已加载")
            self.model_status_label.config(foreground="green")
            self.status_var.set("SAM2模型加载成功")
        except Exception as e:
            log_file = "sam2_model_error.txt"
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{datetime.datetime.now()}] 错误: {str(e)}\n")
                error_msg = f"模型加载失败，错误详情已保存至: {os.path.abspath(log_file)}"
            except:
                error_msg = f"模型加载失败: {str(e)}"
            self.sam2_model = None
            self.image_predictor = None
            self.video_predictor = None
            self.automatic_mask_generator = None
            self.model_status_var.set("加载失败")
            self.model_status_label.config(foreground="red")
            messagebox.showerror("错误", error_msg)
            self.status_var.set(error_msg)
    
    # 保存当前图片的pkl文件
    def save_current_pkl(self):
        """保存当前图片的PKL文件"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载媒体文件")
            return
            
        if self.current_image_path not in self.image_masks or not self.image_masks[self.current_image_path]:
            messagebox.showwarning("警告", "当前媒体没有掩码，请先进行分割")
            return
            
        if not all([self.cloud_file, self.imu_file, self.calib_file]):
            messagebox.showwarning("警告", "请先选择点云、IMU和校准文件")
            return
            
        try:
            # 获取基本文件名
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            
            # 生成保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存PKL文件",
                initialfile=f"{base_name}_result.pkl",
                defaultextension=".pkl",
                filetypes=[("PKL文件", "*.pkl"), ("所有文件", "*.*")]
            )
            
            if not save_path:
                return
                
            # 提取IMU信息
            imu_data = self.imu_data
            if imu_data is None:
                messagebox.showerror("错误", "无法提取IMU信息")
                return
                
            # 读取点云数据
            point_cloud = self.point_cloud
            if point_cloud is None:
                messagebox.showerror("错误", "无法读取点云文件")
                return
                
            # 加载校准参数
            calib_params = self.calib_params
            if calib_params is None:
                messagebox.showerror("错误", "无法解析校准文件")
                return
                
            # 准备数据 - 使用新的数据结构
            pkl_data = {
                "IMU": imu_data  # 包含position, imu_angle, imu_acc
            }
            
            # 添加对象数据
            masks_info = self.image_masks[self.current_image_path]
            image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
            
            for i, mask_info in enumerate(masks_info):
                segmentation = mask_info['segmentation']
                
                # 获取ROI点云并计算3D尺寸
                roi_point_cloud = get_roi_point_cloud(
                    point_cloud, 
                    calib_params, 
                    segmentation, 
                    image
                )
                
                dim_3d = calculate_roi_dimensions(roi_point_cloud)
                
                # 确保掩码数据是可序列化的
                if isinstance(segmentation, np.ndarray):
                    segmentation = segmentation.tolist()
                
                # 获取边界框（如果存在）
                bbox = mask_info.get('bbox', None)
                
                # 获取CID
                cid = mask_info.get('CID', '')
                
                object_key = f"object_{i+1}"
                pkl_data[object_key] = {
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "3d_dimensions": dim_3d,
                    "CID": cid
                }
            
            # 保存为PKL文件
            with open(save_path, 'wb') as f:
                pickle.dump(pkl_data, f)
                
            messagebox.showinfo("成功", f"PKL文件已保存到: {save_path}")
            self.status_var.set(f"PKL文件保存成功: {os.path.basename(save_path)}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"保存PKL文件失败: {str(e)}")
            self.status_var.set(f"PKL保存失败: {str(e)}")
    
    # TODO 待测试功能 批量保存所有图片的PKL文件  
    def save_all_pkl(self):
        """批量保存所有图片的PKL文件"""
        if not self.image_masks:
            messagebox.showwarning("警告", "没有推理结果，请先运行推理")
            return
            
        # 统计有掩码的图片数量
        images_with_masks = [path for path in self.image_list if path in self.image_masks and self.image_masks[path]]
        if not images_with_masks:
            messagebox.showwarning("警告", "没有包含掩码的图片")
            return
            
        # 检查必要的文件
        if not all([self.cloud_file, self.imu_file, self.calib_file]):
            messagebox.showwarning("警告", "请先选择点云、IMU和校准文件")
            return
            
        # 读取点云数据
        point_cloud = self.point_cloud
        if point_cloud is None:
            messagebox.showerror("错误", "无法读取点云文件")
            return
            
        # 加载校准参数
        calib_params = self.calib_params
        if calib_params is None:
            messagebox.showerror("错误", "无法解析校准文件")
            return
            
        # 提取IMU信息（只提取一次）
        imu_data = self.imu_data
        if imu_data is None:
            messagebox.showerror("错误", "无法提取IMU信息")
            return
            
        # 让用户选择保存位置
        save_dir = filedialog.askdirectory(title="选择批量保存PKL的文件夹")
        if not save_dir:
            return
            
        try:
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量保存进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="正在批量保存PKL文件...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(images_with_masks)
            
            success_count = 0
            fail_count = 0
            
            for i, image_path in enumerate(images_with_masks):
                try:
                    # 更新进度
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    progress_label.config(text=f"正在处理: {base_name}")
                    progress_bar['value'] = i
                    progress_window.update()
                    
                    # 生成保存路径
                    save_path = os.path.join(save_dir, f"{base_name}_result.pkl")
                    
                    # 准备数据 - 使用新的数据结构
                    pkl_data = {
                        "IMU": imu_data  # 包含position, imu_angle, imu_acc
                    }
                    
                    # 添加对象数据
                    masks_info = self.image_masks[image_path]
                    
                    # 加载图片
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"无法读取图片: {image_path}")
                    
                    for j, mask_info in enumerate(masks_info):
                        segmentation = mask_info['segmentation']
                        
                        # 获取ROI点云并计算3D尺寸
                        roi_point_cloud = get_roi_point_cloud(
                            point_cloud, 
                            calib_params, 
                            segmentation, 
                            image
                        )
                        
                        dim_3d = calculate_roi_dimensions(roi_point_cloud)
                        
                        # 确保掩码数据是可序列化的
                        if isinstance(segmentation, np.ndarray):
                            segmentation = segmentation.tolist()
                        
                        # 获取边界框（如果存在）
                        bbox = mask_info.get('bbox', None)
                        
                        # 获取CID
                        cid = mask_info.get('CID', '')
                        
                        object_key = f"object_{j+1}"
                        pkl_data[object_key] = {
                            "segmentation": segmentation,
                            "bbox": bbox,
                            "3d_dimensions": dim_3d,
                            "CID": cid
                        }
                    
                    # 保存为PKL文件
                    with open(save_path, 'wb') as f:
                        pickle.dump(pkl_data, f)
                    
                    success_count += 1
                except Exception as e:
                    print(f"保存 {base_name} 失败: {str(e)}")
                    fail_count += 1
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示结果
            messagebox.showinfo("完成", 
                               f"批量保存完成！\n成功: {success_count} 个文件\n失败: {fail_count} 个文件")
            self.status_var.set(f"批量保存完成: 成功 {success_count}, 失败 {fail_count}")
            
        except Exception as e:
            messagebox.showerror("错误", f"批量保存过程中出错: {str(e)}")
            self.status_var.set(f"批量保存失败: {str(e)}")
    
    # 添加单张图片
    def add_single_image(self):
        """添加单张图片到列表"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            # 检查是否已存在
            if file_path not in self.image_list:
                self.image_list.append(file_path)
                # 为新图片初始化标注状态
                self.image_annotations[file_path] = {
                    'positive_points': [],
                    'negative_points': [],
                    'bboxes': [],
                    'operation_history': []
                }
                self.is_video = False
                
            # 跳转到该图片
            self.current_image_index = self.image_list.index(file_path)
            self.load_current_image()
            messagebox.showinfo("成功", f"图片已添加，当前列表共有 {len(self.image_list)} 个媒体文件")
    
    # 添加视频文件
    def add_video_file(self):
        """添加视频文件到列表"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            # 检查是否已存在
            if file_path not in self.image_list:
                self.image_list.append(file_path)
                # 为新视频初始化标注状态
                self.image_annotations[file_path] = {
                    'positive_points': [],
                    'negative_points': [],
                    'bboxes': [],
                    'operation_history': []
                }
                self.is_video = True
                
            # 跳转到该视频
            self.current_image_index = self.image_list.index(file_path)
            self.load_current_image()
            messagebox.showinfo("成功", f"视频已添加，当前列表共有 {len(self.image_list)} 个媒体文件")

    # 清空媒体列表
    def clear_image_list(self):
        """清空媒体列表"""
        if messagebox.askyesno("确认", "确定要清空所有媒体列表吗？这将清除所有标注数据。"):
            self.image_list = []
            self.image_annotations = {}
            self.image_masks = {}
            self.video_segments = {}
            self.current_image_index = 0
            self.current_image = None
            self.current_image_path = None
            self.is_video = False
            self.clear_current_annotations()
            self.canvas.delete("all")
            self.update_image_info()
            self.update_mask_list()
            self.status_var.set("媒体列表已清空")
            
    def save_current_image_state(self):
        """保存当前媒体的交互状态"""
        if self.current_image_path:
            self.image_annotations[self.current_image_path] = {
                'positive_points': self.positive_points.copy(),
                'negative_points': self.negative_points.copy(),
                'bboxes': self.bboxes.copy(),
                'operation_history': self.operation_history.copy()
            }
            
    def load_image_state(self, image_path):
        """加载指定媒体的交互状态"""
        if image_path in self.image_annotations:
            state = self.image_annotations[image_path]
            self.positive_points = state['positive_points'].copy()
            self.negative_points = state['negative_points'].copy()
            self.bboxes = state['bboxes'].copy()
            self.operation_history = state['operation_history'].copy()
            
            # 确保掩码信息中有CID字段
            if image_path in self.image_masks:
                for mask_info in self.image_masks[image_path]:
                    if 'CID' not in mask_info:
                        mask_info['CID'] = ''
        else:
            # 为新媒体初始化状态
            self.positive_points = []
            self.negative_points = []
            self.bboxes = []
            self.operation_history = []
            self.image_annotations[image_path] = {
                'positive_points': [],
                'negative_points': [],
                'bboxes': [],
                'operation_history': []
            }

    def load_single_image(self):
        """加载单张图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.image_list = [file_path]
            self.current_image_index = 0
            self.is_video = False
            # 初始化标注状态
            self.image_annotations = {file_path: {
                'positive_points': [],
                'negative_points': [],
                'bboxes': [],
                'operation_history': []
            }}
            self.load_current_image()
            
    def load_image_folder(self):
        """加载图片文件夹"""
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        if folder_path:
            # 获取文件夹中的所有图片
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            new_image_list = []
            for ext in image_extensions:
                new_image_list.extend(glob.glob(os.path.join(folder_path, ext)))
                new_image_list.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if new_image_list:
                new_image_list.sort()
                # 添加到现有列表，避免重复
                for img_path in new_image_list:
                    if img_path not in self.image_list:
                        self.image_list.append(img_path)
                        # 为新图片初始化标注状态
                        self.image_annotations[img_path] = {
                            'positive_points': [],
                            'negative_points': [],
                            'bboxes': [],
                            'operation_history': []
                        }
                        self.is_video = False
                
                self.current_image_index = 0
                self.load_current_image()
                messagebox.showinfo("成功", f"文件夹已加载，当前列表共有 {len(self.image_list)} 个媒体文件")
            else:
                messagebox.showwarning("警告", "文件夹中未找到图片文件")
                
    def load_current_image(self):
        """加载当前媒体（图片或视频第一帧）"""
        if not self.image_list:
            return
            
        # 保存之前媒体的状态
        if self.current_image_path:
            self.save_current_image_state()
            
        media_path = self.image_list[self.current_image_index]
        self.current_image_path = media_path
        
        # 重置选中的掩码
        self.selected_mask_index = None
        
        try:
            # 如果是视频文件
            if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.is_video = True
                self.status_var.set(f"加载视频: {os.path.basename(media_path)}")
                
                # 提取视频第一帧
                cap = cv2.VideoCapture(media_path)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {media_path}")
                
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"无法读取视频第一帧: {media_path}")
                
                self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
                
                # 加载该视频的交互状态
                self.load_image_state(media_path)
                
                # 显示第一帧
                self.display_current_image()
                
                # 更新信息
                self.update_image_info()
                self.update_points_info()
                self.update_mask_list()
                
            # 如果是图片文件
            else:
                self.is_video = False
                # 读取图片
                self.current_image = cv2.imread(media_path)
                if self.current_image is None:
                    raise ValueError(f"无法读取图片: {media_path}")
                    
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # 加载该图片的交互状态
                self.load_image_state(media_path)
                
                # 显示图片
                self.display_current_image()
                
                # 更新信息
                self.update_image_info()
                self.update_points_info()
                self.update_mask_list()
                self.status_var.set(f"已加载: {os.path.basename(media_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"媒体加载失败: {str(e)}")
            
    def create_display_image(self):
        """创建用于显示的图像 - 确保每次都从原始图像和掩码重新开始"""
        if self.current_image is None:
            return None
            
        # 从原始图像开始
        display_img = self.current_image.copy()
        
        # 如果有推理结果，应用掩码
        if self.current_image_path in self.image_masks:
            masks_info = self.image_masks[self.current_image_path]
            # 使用原始图像应用掩码
            display_img = self.apply_masks(display_img, masks_info)
        
        # 绘制标注
        self.draw_annotations(display_img)
        
        # 如果有选中的掩码，在中心绘制五角星
        if self.selected_mask_index is not None:
            masks_info = self.image_masks.get(self.current_image_path, [])
            if self.selected_mask_index < len(masks_info):
                mask_info = masks_info[self.selected_mask_index]
                mask = mask_info['segmentation']
                center_x, center_y = self.calculate_mask_center(mask)
                
                # 绘制黄色五角星
                cv2.drawMarker(
                    display_img, 
                    (center_x, center_y), 
                    (255, 255, 0),  # 黄色 (BGR格式)
                    markerType=cv2.MARKER_STAR,
                    markerSize=30,
                    thickness=2
                )
        
        return display_img
        
    def calculate_mask_center(self, mask):
        """计算掩码的中心坐标"""  
        # 找到掩码中所有非零像素的坐标
        points = np.argwhere(mask > 0.5)
        if points.size == 0:
            # 如果没有有效点，返回图像中心
            height, width = mask.shape[:2]
            return width // 2, height // 2
            
        # 计算所有非零像素的平均坐标
        center_y = np.mean(points[:, 0])
        center_x = np.mean(points[:, 1])
        return int(center_x), int(center_y)
        
    def display_current_image(self):
        """在画布上显示当前媒体"""
        display_img = self.create_display_image()
        if display_img is None:
            return
            
        # 转换为PIL图片
        pil_image = Image.fromarray(display_img)
        
        # 计算缩放比例以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            self.scale_factor = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
            
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter格式
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图片
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        
        # 更新画布滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def apply_masks(self, image, masks_info):
        """应用掩码到图像，每个掩码使用不同颜色"""
        # 确保我们有原始图像的副本
        result = image.copy().astype(np.float32)  # 使用float32避免精度损失
        
        # 应用掩码（半透明覆盖）
        for mask_info in masks_info:
            mask = mask_info['segmentation']
            color = mask_info['color']
            
            # 将掩码调整到图像大小
            if mask.shape != result.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (result.shape[1], result.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # 确保掩码是二值格式
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # 创建彩色掩码
            colored_mask = np.zeros_like(result, dtype=np.float32)
            # 将掩码区域设置为指定颜色
            for c in range(3):  # 对于RGB三个通道
                colored_mask[:, :, c] = np.where(mask_binary, color[c], 0)
            
            # 半透明叠加
            alpha = 0.5  # 增加透明度，使掩码更明显
            for c in range(3):
                result[:, :, c] = np.where(
                    mask_binary, 
                    (1 - alpha) * result[:, :, c] + alpha * colored_mask[:, :, c],
                    result[:, :, c]
                )
        
        # 转回uint8类型
        return np.clip(result, 0, 255).astype(np.uint8)

    def draw_annotations(self, image):
        """在图片上绘制标注"""
        # 绘制正向点（绿色）
        for point in self.positive_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.circle(image, point, 7, (0, 0, 0), 2)
            
        # 绘制负向点（红色）
        for point in self.negative_points:
            cv2.circle(image, point, 5, (255, 0, 0), -1)
            cv2.circle(image, point, 7, (0, 0, 0), 2)
            
        # 绘制边界框（蓝色）
        for bbox in self.bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
        # 绘制临时边界框
        if self.temp_bbox:
            cv2.rectangle(image, (self.temp_bbox[0], self.temp_bbox[1]), 
                         (self.temp_bbox[2], self.temp_bbox[3]), (255, 255, 0), 2)
            
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为图片坐标"""
        if self.scale_factor <= 0:
            return int(canvas_x), int(canvas_y)
        return int(canvas_x / self.scale_factor), int(canvas_y / self.scale_factor)
        
    def on_canvas_click(self, event):
        """画布点击事件"""
        if self.current_image is None:
            return
            
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # 确保坐标在图片范围内
        img_height, img_width = self.current_image.shape[:2]
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        
        if self.mode_var.get() == "positive":
            self.add_positive_point(img_x, img_y)
        elif self.mode_var.get() == "negative":
            self.add_negative_point(img_x, img_y)
        elif self.mode_var.get() == "bbox":
            self.bbox_start = (img_x, img_y)
            
    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        if self.mode_var.get() == "bbox" and self.bbox_start and self.current_image is not None:
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            img_height, img_width = self.current_image.shape[:2]
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))
            
            self.temp_bbox = (min(self.bbox_start[0], img_x), min(self.bbox_start[1], img_y),
                             max(self.bbox_start[0], img_x), max(self.bbox_start[1], img_y))
            self.display_current_image()
            
    def on_canvas_release(self, event):
        """画布释放事件"""
        if self.mode_var.get() == "bbox" and self.bbox_start and self.temp_bbox:
            self.add_bbox(self.temp_bbox)
            self.temp_bbox = None
            self.bbox_start = None
            
    def add_positive_point(self, x, y):
        """添加正向点"""
        self.positive_points.append((x, y))
        self.operation_history.append(("add_positive", (x, y)))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def add_negative_point(self, x, y):
        """添加负向点"""
        self.negative_points.append((x, y))
        self.operation_history.append(("add_negative", (x, y)))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def add_bbox(self, bbox):
        """添加边界框"""
        self.bboxes.append(bbox)
        self.operation_history.append(("add_bbox", bbox))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def change_mode(self):
        """改变交互模式"""
        self.current_mode = self.mode_var.get()
        
    def undo_operation(self):
        """撤销上一个操作"""
        if not self.operation_history:
            messagebox.showinfo("信息", "没有可撤销的操作")
            return
            
        last_op = self.operation_history.pop()
        op_type, data = last_op
        
        if op_type == "add_positive":
            self.positive_points.remove(data)
        elif op_type == "add_negative":
            self.negative_points.remove(data)
        elif op_type == "add_bbox":
            self.bboxes.remove(data)
            
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def clear_current_annotations(self):
        """清除当前媒体的所有标注"""
        self.positive_points = []
        self.negative_points = []
        self.bboxes = []
        self.operation_history = []
        self.temp_bbox = None
        self.bbox_start = None
        self.selected_mask_index = None
        if self.current_image is not None:
            self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
            
    def clear_all_annotations(self):
        """清除所有标注"""
        self.clear_current_annotations()
        
    def update_points_info(self):
        """更新标注信息显示"""
        # 显示当前媒体的标注信息
        annotated_count = sum(1 for path in self.image_annotations 
                            if (self.image_annotations[path]['positive_points'] or 
                                self.image_annotations[path]['negative_points'] or 
                                self.image_annotations[path]['bboxes']))
        
        self.points_info_label.config(
            text=f"正向点: {len(self.positive_points)} | "
                 f"负向点: {len(self.negative_points)} | "
                 f"边界框: {len(self.bboxes)} | "
                 f"已标注: {annotated_count}/{len(self.image_list)}"
        )
        
    def update_image_info(self):
        """更新媒体信息显示"""
        if self.image_list and self.current_image_path:
            # 检查当前媒体是否有推理结果
            has_result = self.current_image_path in self.image_masks and self.image_masks[self.current_image_path]
            result_text = " (已推理)" if has_result else ""
            
            media_type = "[视频]" if self.is_video else "[图片]"
            
            self.image_info_label.config(
                text=f"{media_type} {os.path.basename(self.current_image_path)} - {self.current_image_index + 1}/{len(self.image_list)}{result_text}"
            )
        else:
            self.image_info_label.config(text="未加载媒体")
            
    def prev_image(self):
        """上一张媒体"""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def next_image(self):
        """下一张媒体"""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def run_current_inference(self):
        """运行当前媒体的SAM2推理"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载媒体")
            return
            
        if self.image_predictor is None:
            messagebox.showerror("错误", "SAM2模型未加载")
            return
            
        # 检查是否有标注
        if not self.positive_points and not self.negative_points and not self.bboxes:
            messagebox.showwarning("警告", "请先添加一些标注（点或边界框）")
            return
            
        try:
            self.status_var.set("正在运行SAM2推理...")
            self.root.update()
            
            # 保存当前批次的点和框的信息
            current_points = self.positive_points.copy()
            current_neg_points = self.negative_points.copy()
            current_bboxes = self.bboxes.copy()
            
            # 设置当前图像
            self.image_predictor.set_image(self.current_image)
            
            # 准备输入点坐标和标签
            point_coords = np.array(current_points + current_neg_points) if current_points or current_neg_points else None
            point_labels = np.array([1] * len(current_points) + [0] * len(current_neg_points)) if current_points or current_neg_points else None
            
            # 准备边界框输入
            box_input = np.array(current_bboxes) if current_bboxes else None
            
            # 根据PDF中的"Batched prompt inputs"部分处理多个框的情况
            if current_bboxes and len(current_bboxes) > 1:
                # 批量框推理
                masks, scores, _ = self.image_predictor.predict(
                    point_coords=None,  # 多个框时不使用点提示
                    point_labels=None,
                    box=box_input,
                    multimask_output=False
                )
                
                # 确保返回的结果是列表形式
                masks = masks if masks.ndim == 4 else [masks]
                
            # 情况1: 只有一个正向点 (单点分割)
            elif len(current_points) == 1 and not current_neg_points and not current_bboxes:
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True  # 返回3个候选掩码
                )
                # 选择分数最高的掩码
                best_idx = np.argmax(scores)
                masks = [masks[best_idx]]
                scores = [scores[best_idx]]
                
            # 其他情况（点+框、多个点、点+负向点等）
            else:
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_input[0] if box_input is not None and len(box_input) == 1 else None,
                    multimask_output=False
                )
                # 确保返回的结果是列表形式
                masks = [masks] if masks.ndim == 2 else masks
                
            # 处理推理结果
            if len(masks) > 0:
                # 初始化当前媒体的掩码列表（如果不存在）
                if self.current_image_path not in self.image_masks:
                    self.image_masks[self.current_image_path] = []
                
                # 获取新掩码的起始索引
                start_index = len(self.image_masks[self.current_image_path])
                
                # 处理每个生成的掩码
                for i, mask in enumerate(masks):
                    # 对于批量框推理，每个框对应一个掩码
                    if current_bboxes and len(current_bboxes) > 1:
                        mask = mask.squeeze(0)  # 移除批次维度
                    
                    # 获取分数（如果有）
                    score = scores[i] if i < len(scores) else 0.0
                    
                    # 计算边界框
                    if np.any(mask > 0.5):
                        coords = np.argwhere(mask > 0.5)
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # XYWH格式
                    else:
                        bbox = None
                    
                    # 构建掩码信息字典
                    mask_info = {
                        'segmentation': mask,
                        'area': int(np.sum(mask > 0.5)),
                        'bbox': bbox,
                        'predicted_iou': float(score),
                        'color': self.get_next_color(),
                        'user_points': {
                            'positive': current_points.copy(),
                            'negative': current_neg_points.copy()
                        },
                        'user_bboxes': current_bboxes.copy(),
                        'CID': ''  # 新增CID字段，初始为空
                    }
                    
                    # 添加新生成的掩码信息
                    self.image_masks[self.current_image_path].append(mask_info)
                
                # 清除当前标注
                self.positive_points = []
                self.negative_points = []
                self.bboxes = []
                self.operation_history = []
                
                self.display_current_image()
                self.update_image_info()
                self.update_mask_list()
                
                # 显示生成的mask信息
                mask_count = len(self.image_masks[self.current_image_path])
                last_mask = self.image_masks[self.current_image_path][-1]
                self.status_var.set(f"SAM2推理完成 - 生成 {len(masks)} 个掩码")
                messagebox.showinfo("推理结果", 
                                f"生成 {len(masks)} 个新掩码\n最后掩码面积: {last_mask['area']} 像素, 分数: {last_mask['predicted_iou']:.3f}")
            else:
                messagebox.showerror("错误", "推理失败，未生成掩码")
                self.status_var.set("推理失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"推理过程中出错: {str(e)}")
            self.status_var.set(f"推理失败: {str(e)}")

    def segment_entire_video(self):
        """分割整个视频，支持多对象"""
        if not self.is_video:
            messagebox.showwarning("警告", "当前媒体不是视频文件")
            return
        if not self.image_predictor or not self.video_predictor:
            messagebox.showerror("错误", "SAM2模型未加载")
            return
        if self.current_image_path not in self.image_masks or not self.image_masks[self.current_image_path]:
            messagebox.showwarning("警告", "请先对视频第一帧进行分割")
            return
        if not messagebox.askyesno("确认", "确定要分割整个视频吗？这可能需要较长时间。"):
            return
        try:
            video_path = self.current_image_path
            output_dir = self.extract_video_frames(video_path)
            frame_names = sorted([p for p in os.listdir(output_dir)
                                  if p.lower().endswith(('.jpg', '.jpeg'))],
                                 key=lambda p: int(os.path.splitext(p)[0]))
            self.video_frames_dir = output_dir
            inference_state = self.video_predictor.init_state(video_path=output_dir)
            # 多对象支持：遍历所有掩码，分别add_new_points_or_box
            obj_id_base = 0
            self.video_cid_mapping = {}  # 存储obj_id到CID的映射
            for idx, mask_info in enumerate(self.image_masks[video_path]):
                points = np.array(mask_info['user_points']['positive'] + mask_info['user_points']['negative'], dtype=np.float32)
                labels = np.array([1]*len(mask_info['user_points']['positive']) + [0]*len(mask_info['user_points']['negative']), dtype=np.int32)
                box = None
                if mask_info['user_bboxes']:
                    box = np.array(mask_info['user_bboxes'][0], dtype=np.float32)
                obj_id = obj_id_base + idx
                # 记录CID映射
                self.video_cid_mapping[obj_id] = mask_info.get('CID', '')
                self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points if len(points) else None,
                    labels=labels if len(labels) else None,
                    box=box
                )
            # 传播分割
            self.video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                self.video_segments[out_frame_idx] = {
                    obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, obj_id in enumerate(out_obj_ids)
                }
            self.status_var.set("视频分割完成")
            messagebox.showinfo("成功", f"视频分割完成，共处理 {len(self.video_segments)} 帧")
        except Exception as e:
            messagebox.showerror("错误", f"视频分割失败: {str(e)}")
            self.status_var.set(f"视频分割失败: {str(e)}")
            traceback.print_exc()
    
    def extract_video_frames(self, input_video: str, quality: int = 2, start_number: int = 0, 
                            ffmpeg_path: Optional[str] = None, fps: Optional[int] = None) -> str:
        """使用ffmpeg提取视频帧"""
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"视频文件不存在: {input_video}")
        
        # 创建输出目录
        video_dir = os.path.dirname(input_video)
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        output_dir = os.path.join(video_dir, video_name + "_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_pattern = os.path.join(output_dir, "%05d.jpg")
        ffmpeg_cmd = ffmpeg_path or "ffmpeg"
        
        # 构建ffmpeg命令
        cmd = [
            ffmpeg_cmd,
            "-i", input_video,
            "-q:v", str(quality),
            "-start_number", str(start_number),
        ]
        
        # 添加帧率参数
        if fps is not None:
            cmd.extend(["-r", str(fps)])
        
        cmd.extend([
            output_pattern,
            "-y"
        ])
        
        # 运行命令
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_dir

    def save_video_segments(self):
        """保存视频分割结果"""
        # 优先保存自动分割结果
        video_segments = getattr(self, 'video_segments_auto', None)
        if video_segments and len(video_segments) > 0:
            pass  # 已赋值
        elif self.video_segments and self.is_video:
            video_segments = self.video_segments
        else:
            messagebox.showwarning("警告", "没有视频分割结果可保存")
            return
        try:
            save_dir = filedialog.askdirectory(title="选择保存视频分割结果的文件夹")
            if not save_dir:
                return
            video_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            video_save_dir = os.path.join(save_dir, video_name + "_seg_results")
            os.makedirs(video_save_dir, exist_ok=True)
            progress_window = tk.Toplevel(self.root)
            progress_window.title("保存进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_label = ttk.Label(progress_window, text="正在保存视频分割结果...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(video_segments)
            for i, (frame_idx, masks) in enumerate(video_segments.items()):
                frame_dir = os.path.join(video_save_dir, f"frame_{frame_idx:05d}")
                os.makedirs(frame_dir, exist_ok=True)
                frame_path = os.path.join(self.video_frames_dir, f"{frame_idx:05d}.jpg")
                if not os.path.exists(frame_path):
                    continue
                frame = cv2.imread(frame_path)
                if frame is None or frame.size == 0:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 保存原始帧
                orig_frame_path = os.path.join(frame_dir, "original.jpg")
                cv2.imwrite(orig_frame_path, frame)
                # 保存掩码和可视化
                mask_list = []
                color_list = []
                # 兼容两种掩码结构
                if isinstance(masks, dict):
                    # 旧版：{obj_id: mask}
                    for obj_id, mask in masks.items():
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        # 不再保存mask图像
                        mask_list.append(mask_bin)
                        color_list.append(self.colors[obj_id % len(self.colors)])
                elif isinstance(masks, list):
                    # 自动分割：list of mask dict
                    for idx, mask_info in enumerate(masks):
                        mask = mask_info['segmentation']
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        # 不再保存mask图像
                        mask_list.append(mask_bin)
                        color_list.append(self.colors[idx % len(self.colors)])
                # 保存notebook风格可视化
                vis_path = os.path.join(frame_dir, f"visualization_overlay.png")
                self.save_mask_overlay_png(frame_rgb, mask_list, color_list, vis_path)
                # === 新增：保存PKL文件 ===
                pkl_path = os.path.join(frame_dir, "segmentation.pkl")
                self.save_video_frame_pkl(masks, pkl_path)
                progress_label.config(text=f"正在保存帧 {frame_idx}")
                progress_bar['value'] = i + 1
                progress_window.update()
            progress_window.destroy()
            messagebox.showinfo("成功", f"视频分割结果已保存到: {video_save_dir}")
            self.status_var.set(f"视频分割结果保存完成")
        except Exception as e:
            messagebox.showerror("错误", f"保存视频分割结果失败: {str(e)}")
            self.status_var.set(f"保存失败: {str(e)}")
            traceback.print_exc()

    # 保存视频帧的PKL文件
    def save_video_frame_pkl(self, masks, save_path):
        """保存视频帧的分割结果到PKL文件"""
        # IMU字段结构与图像分割一致，但内容为空
        pkl_data = {
            "IMU": {
                "position": [],
                "imu_angle": [],
                "imu_acc": []
            }
        }
        obj_count = 1
        if isinstance(masks, dict):
            # 旧版：{obj_id: mask}
            for obj_id, mask in masks.items():
                mask = np.squeeze(mask)
                if np.any(mask > 0.5):
                    coords = np.argwhere(mask > 0.5)
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    bbox = None
                dim_3d = [0.0] * 10  # 强制为list[float]
                cid = str(mask_info.get('CID', ''))  # 强制为字符串
                if isinstance(mask, np.ndarray):
                    mask = mask.tolist()
                object_key = f"object_{obj_count}"
                pkl_data[object_key] = {
                    "segmentation": mask,
                    "bbox": bbox,
                    "3d_dimensions": dim_3d,
                    "CID": cid
                }
                obj_count += 1
        elif isinstance(masks, list):
            # 自动分割：list of mask dict
            for mask_info in masks:
                mask = mask_info.get('segmentation')
                bbox = mask_info.get('bbox')
                dim_3d = [0.0] * 10  # 强制为list[float]
                cid = str(mask_info.get('CID', ''))  # 强制为字符串
                if isinstance(mask, np.ndarray):
                    mask = mask.tolist()
                object_key = f"object_{obj_count}"
                pkl_data[object_key] = {
                    "segmentation": mask,
                    "bbox": bbox,
                    "3d_dimensions": dim_3d,
                    "CID": cid
                }
                obj_count += 1
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_data, f)
    
    def run_batch_inference(self):
        """批量推理所有图片"""
        if not self.image_list:
            messagebox.showwarning("警告", "请先加载图片")
            return
            
        if self.image_predictor is None:
            messagebox.showerror("错误", "SAM2模型未加载")
            return
            
        # 检查有标注的图片
        annotated_images = []
        for img_path in self.image_list:
            if img_path in self.image_annotations:
                annotations = self.image_annotations[img_path]
                if (annotations['positive_points'] or 
                    annotations['negative_points'] or 
                    annotations['bboxes']):
                    annotated_images.append(img_path)
        
        if not annotated_images:
            messagebox.showwarning("警告", "没有已标注的图片，请先对图片进行标注")
            return
            
        # 确认批量推理
        if not messagebox.askyesno("确认", f"将对 {len(annotated_images)} 张已标注的图片进行批量推理，继续吗？"):
            return
            
        try:
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量推理进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="正在进行批量推理...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(annotated_images)
            
            successful_count = 0
            failed_count = 0
            
            for i, img_path in enumerate(annotated_images):
                progress_label.config(text=f"正在处理: {os.path.basename(img_path)}")
                progress_bar['value'] = i
                progress_window.update()
                
                # 临时设置状态以运行推理
                original_state = self._save_temp_state()
                self._load_temp_state(img_path)
                
                try:
                    # 保存该图片的现有掩码CID信息
                    existing_cids = {}
                    if img_path in self.image_masks:
                        for j, mask_info in enumerate(self.image_masks[img_path]):
                            if 'CID' in mask_info:
                                existing_cids[j] = mask_info['CID']
                    
                    # 加载图片
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError(f"无法读取图片: {img_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 设置当前图像
                    self.image_predictor.set_image(image)
                    
                    # 准备输入点坐标和标签
                    current_points = self.positive_points.copy()
                    current_neg_points = self.negative_points.copy()
                    current_bboxes = self.bboxes.copy()
                    
                    point_coords = np.array(current_points + current_neg_points)
                    point_labels = np.array([1] * len(current_points) + [0] * len(current_neg_points))
                    
                    # 准备边界框输入
                    box_input = None
                    if current_bboxes:
                        box_input = np.array([current_bboxes[0]])  # 只取第一个框
                    
                    # 运行推理
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=point_coords if point_coords.size > 0 else None,
                        point_labels=point_labels if point_labels.size > 0 else None,
                        box=box_input,
                        multimask_output=False  # 只返回一个掩码
                    )
                    
                    if len(masks) > 0:
                        # 取分数最高的掩码
                        best_idx = np.argmax(scores)
                        best_mask = masks[best_idx]
                        best_score = scores[best_idx]
                        
                        # 计算边界框
                        if np.any(best_mask > 0.5):
                            coords = np.argwhere(best_mask > 0.5)
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # XYWH格式
                        else:
                            bbox = None
                        
                        # 构建掩码信息字典
                        mask_info = {
                            'segmentation': best_mask,
                            'area': int(np.sum(best_mask > 0.5)),
                            'bbox': bbox,
                            'predicted_iou': float(best_score),
                            'color': self.get_next_color(),
                            'user_points': {
                                'positive': current_points.copy(),
                                'negative': current_neg_points.copy()
                            },
                            'user_bboxes': current_bboxes.copy(),
                            'CID': ''  # 初始为空
                        }
                        
                        # 初始化当前图片的掩码列表（如果不存在）
                        if img_path not in self.image_masks:
                            self.image_masks[img_path] = []
                        
                        # 获取新掩码的起始索引
                        start_index = len(self.image_masks[img_path])
                        
                        # 添加新生成的掩码
                        self.image_masks[img_path].append(mask_info)
                        
                        # 恢复之前存在的CID值
                        for idx, cid_value in existing_cids.items():
                            if idx < start_index:  # 只恢复之前的掩码，不覆盖新生成的
                                self.image_masks[img_path][idx]['CID'] = cid_value
                        
                        successful_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"图片 {img_path} 推理失败: {e}")
                    failed_count += 1
                
                # 恢复原始状态
                self._restore_temp_state(original_state)
            
            progress_bar['value'] = len(annotated_images)
            progress_window.update()
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示结果
            messagebox.showinfo("批量推理完成", 
                              f"批量推理完成！\n成功: {successful_count} 张\n失败: {failed_count} 张")
            
            # 更新当前显示
            if self.current_image_path in self.image_masks and self.image_masks[self.current_image_path]:
                self.display_current_image()
            self.update_image_info()
            self.update_mask_list()
            self.status_var.set(f"批量推理完成 - 成功 {successful_count} 张")
            
        except Exception as e:
            messagebox.showerror("错误", f"批量推理过程中出错: {str(e)}")
            self.status_var.set("批量推理失败")
    
    def reset_video_state(self):
        """重置视频分割状态"""
        if hasattr(self, 'inference_state') and self.inference_state:
            self.video_predictor.reset_state(self.inference_state)
            self.video_segments = {}
            self.status_var.set("视频分割状态已重置")

    def _save_temp_state(self):
        """保存临时状态"""
        return {
            'positive_points': self.positive_points.copy(),
            'negative_points': self.negative_points.copy(),
            'bboxes': self.bboxes.copy(),
            'operation_history': self.operation_history.copy()
        }
    
    def _load_temp_state(self, img_path):
        """加载临时状态"""
        if img_path in self.image_annotations:
            state = self.image_annotations[img_path]
            self.positive_points = state['positive_points'].copy()
            self.negative_points = state['negative_points'].copy()
            self.bboxes = state['bboxes'].copy()
            self.operation_history = state['operation_history'].copy()
    
    def _restore_temp_state(self, state):
        """恢复临时状态"""
        self.positive_points = state['positive_points']
        self.negative_points = state['negative_points']
        self.bboxes = state['bboxes']
        self.operation_history = state['operation_history']
    
    def update_mask_list(self):
        """更新掩码列表显示"""
        # 清除现有数据
        for item in self.mask_tree.get_children():
            self.mask_tree.delete(item)
        
        if self.current_image_path and self.current_image_path in self.image_masks:
            masks_info = self.image_masks[self.current_image_path]
            for i, mask_info in enumerate(masks_info):
                mask = mask_info['segmentation']
                center_x, center_y = self.calculate_mask_center(mask)
                
                # 确保CID字段存在
                if 'CID' not in mask_info:
                    mask_info['CID'] = ''
                    
                cid = mask_info['CID']
                
                # 添加数据到Treeview
                self.mask_tree.insert("", "end", values=(f"mask_{i}", f"({center_x}, {center_y})", cid))
    
    def on_mask_selected(self, event):
        """当掩码列表中的项被选中时调用"""
        if not self.current_image_path or self.current_image_path not in self.image_masks:
            return
            
        selection = self.mask_tree.selection()
        if selection:
            # 获取选中项的索引
            item = self.mask_tree.item(selection[0])
            item_id = item['values'][0]  # 格式为'mask_0'
            self.selected_mask_index = int(item_id.split('_')[1])
            self.display_current_image()  # 重绘图片以显示五角星
    
    def on_cid_double_click(self, event):
        """编辑并更新CID值，并立即更新到数据结构"""
        # 确定点击的列
        region = self.mask_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        # 获取点击的行和列
        column = self.mask_tree.identify_column(event.x)
        item = self.mask_tree.identify_row(event.y)

        if not item or column != "#3":  # 只处理CID列（第三列）
            return

        # 获取当前CID值
        current_values = self.mask_tree.item(item, "values")
        current_cid = current_values[2] if len(current_values) > 2 else ""

        # 创建编辑对话框
        new_cid = simpledialog.askstring("编辑CID", "请输入CID:",
                                        initialvalue=current_cid,
                                        parent=self.root)
        if new_cid is None:  # 用户取消
            return

        # 更新Treeview中的显示
        new_values = (current_values[0], current_values[1], new_cid)
        self.mask_tree.item(item, values=new_values)

        # 同时更新数据结构
        # 解析掩码索引
        mask_id = current_values[0]  # 格式为'mask_0'
        try:
            mask_index = int(mask_id.split('_')[1])
            if self.current_image_path and self.current_image_path in self.image_masks:
                if 0 <= mask_index < len(self.image_masks[self.current_image_path]):
                    # 更新掩码信息中的CID
                    self.image_masks[self.current_image_path][mask_index]['CID'] = new_cid
                    self.status_var.set(f"掩码 {mask_index} 的CID已更新为: {new_cid}")
        except (IndexError, ValueError) as e:
            print(f"更新CID出错: {str(e)}")
    
    def delete_selected_mask(self):
        """删除选中的掩码"""
        if not self.current_image_path or self.current_image_path not in self.image_masks:
            return
            
        selection = self.mask_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个掩码")
            return
            
        # 获取选中项的索引
        item = self.mask_tree.item(selection[0])
        item_id = item['values'][0]  # 格式为'mask_0'
        index = int(item_id.split('_')[1])
        masks_info = self.image_masks[self.current_image_path]
        
        if index < len(masks_info):
            # 删除选中的掩码
            del masks_info[index]
            
            # 如果没有掩码了，从字典中删除该条目
            if not masks_info:
                del self.image_masks[self.current_image_path]
            
            # 重置选中状态
            self.selected_mask_index = None
            
            # 更新显示
            self.display_current_image()
            self.update_mask_list()
            self.status_var.set("掩码已删除")
    
    def clear_all_masks(self):
        """清除当前媒体的所有掩码"""
        if self.current_image_path and self.current_image_path in self.image_masks:
            if messagebox.askyesno("确认", "确定要清除当前媒体的所有掩码吗？"):
                del self.image_masks[self.current_image_path]
                self.selected_mask_index = None
                self.display_current_image()
                self.update_mask_list()
                self.status_var.set("已清除所有掩码")
    
    
    def export_masks_as_images(self):
        """导出掩码为图像文件"""
        if not self.current_image_path or self.current_image_path not in self.image_masks:
            messagebox.showwarning("警告", "当前媒体没有掩码，请先进行分割")
            return
        try:
            save_dir = filedialog.askdirectory(title="选择保存掩码图像的文件夹")
            if not save_dir:
                return
            masks_info = self.image_masks[self.current_image_path]
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            mask_list = []
            color_list = []
            for i, mask_info in enumerate(masks_info):
                mask = mask_info['segmentation']
                mask_img = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(save_dir, f"{base_name}_mask_{i}.png")
                cv2.imwrite(mask_path, mask_img)
                mask_list.append(mask > 0.5)
                color_list.append(mask_info['color'])
            # 保存notebook风格可视化
            if self.current_image is not None:
                vis_path = os.path.join(save_dir, f"{base_name}_visualization_overlay.png")
                self.save_mask_overlay_png(self.current_image, mask_list, color_list, vis_path)
            messagebox.showinfo("成功", f"{len(masks_info)} 个掩码图像已保存到: {save_dir}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
    
    def save_mask_overlay_png(self, frame, masks, colors, save_path):
        """
        将多个掩码半透明叠加到原图，保存为无坐标轴的PNG。
        frame: HWC, uint8, RGB or BGR
        masks: list of 2D np.ndarray, bool或0/1
        colors: list of (R,G,B) tuple
        save_path: 输出PNG路径
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import numpy as np
        from PIL import Image
        fig = Figure(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(frame)
        h, w = frame.shape[:2]
        for mask, color in zip(masks, colors):
            # 修正：确保掩码为2D且与frame一致
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = np.squeeze(mask)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(float)
            color_arr = np.array(color) / 255.0
            overlay = np.zeros((h, w, 4), dtype=float)
            overlay[..., :3] = color_arr
            overlay[..., 3] = mask * 0.5  # alpha
            ax.imshow(overlay)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[..., :3]
        Image.fromarray(img).save(save_path)

    def clear_model_selection(self):
        """清除已选择的配置文件和模型文件，并清空已加载的模型对象"""
        self.model_config_path = None
        self.model_weights_path = None
        self.model_config_name = None
        self.model_config_var.set("未选择配置文件")
        self.model_weights_var.set("未选择模型文件")
        self.status_var.set("已清除模型和配置文件选择，请重新选择")
        # 清空模型对象
        self.sam2_model = None
        self.image_predictor = None
        self.video_predictor = None

    def auto_segment_video(self):
        """自动分割视频每一帧的所有对象（带进度条）"""
        if not self.is_video:
            messagebox.showwarning("警告", "当前媒体不是视频文件")
            return
        if not hasattr(self, "automatic_mask_generator") or self.automatic_mask_generator is None:
            messagebox.showerror("错误", "SAM2自动掩码生成器未加载")
            return
        try:
            video_path = self.current_image_path
            output_dir = self.extract_video_frames(video_path)
            frame_names = sorted([p for p in os.listdir(output_dir)
                                   if p.lower().endswith(('.jpg', '.jpeg'))],
                                  key=lambda p: int(os.path.splitext(p)[0]))
            self.video_frames_dir = output_dir
            self.video_segments_auto = {}  # {frame_idx: [mask_dict, ...]}
            total_obj = 0

            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("自动视频分割进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_label = ttk.Label(progress_window, text="正在分割...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(frame_names)

            for i, fname in enumerate(frame_names):
                frame_idx = int(os.path.splitext(fname)[0])
                frame_path = os.path.join(output_dir, fname)
                image = cv2.imread(frame_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                masks = self.automatic_mask_generator.generate(image_rgb)
                self.video_segments_auto[frame_idx] = masks
                total_obj += len(masks)
                progress_label.config(text=f"正在处理帧 {i+1}/{len(frame_names)}: {fname}")
                progress_bar['value'] = i+1
                progress_window.update()

            progress_window.destroy()
            self.status_var.set(f"自动视频分割完成，共{len(frame_names)}帧，总对象数{total_obj}")
            messagebox.showinfo("成功", f"自动视频分割完成！\n共{len(frame_names)}帧，总对象数{total_obj}")
        except Exception as e:
            try:
                progress_window.destroy()
            except:
                pass
            messagebox.showerror("错误", f"自动视频分割失败: {str(e)}")
            self.status_var.set(f"自动视频分割失败: {str(e)}")

def main():
    root = tk.Tk()
    app = SAM2GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# 2025.07.08 17:05
# 模型和代码文件现在放在了notebooks文件夹下，可以正确运行了
# 解决了 同时圈出多个边界框时会报错的问题
# 解决了 手动添加类型ID
# 2025-07-08 19:37:39 TODO 1. CID修改后，需要点击'确认CID'按钮后，进行下一次图像分割才不会导致CID消失
# 2025-07-08 20:07:57 解决了 CID 的问题，可以直接修改了
# 2. TODO 先尝试保存为pkl文件，再看保存的内容对不对
# 2025-07-09 14:06:32 可以正确保存为pkl文件了，pkl文件中的数据结构为：
# {
#     "IMU": {
#         "position": [49.014323937889, 8.3527311915474, 110.23770141602],
#         "imu_angle": [0.030998, 0.019566, -1.5778929803847],
#         "imu_acc": [-1.2681048447099, 0.26155146853317, 9.7581527084322]
#         },
#     "object_1": {
#         "segmentation":[...]
#         "bbox": [...],
#         "3d_dimensions": [...],
#         "CID": "12"
#     "object_2": {
#         "segmentation":[...]
#         "bbox": [...],
#         "3d_dimensions": [...],
#         "CID": "12"
# }
# 修复一些其他bug 修复一些其他bug 给清除标注功能新增一个功能，也就是点击这个按钮时，可以把图片上因为选中掩码而产生的中心也给删除
# 2025-07-09 14:23:15 解决了上边 编号3 问题
# 2025-07-09 14:25:25 新增视频分割功能
# 2025-07-09 17:12:20 暂时没有解决视频分割的报错问题
# 2025-07-09 17:51:05 同上
# 2025-07-09 19:58:46 可以分割一点了，但是还是报错：
# 2025-07-09 20:28:22 没有解决视频分割的问题
# 2025-07-09 21:18:10 第2171 仍然报错 
# 2025-07-10 10:51:19 可以分割视频了，但是有两个bug，1. 先打点，运行图片推理后，打的点会消失，没有传给sam2的视频分割函数，需要自己重新打点，2. 保存视频分割结果时报错
# 2025-07-10 11:54:46 可以正确保存视频帧分割后的图片了，但是没有保存掩码信息
# 2025-07-10 11:55:11 修改视频帧的保存，使之能够保存pkl文件
# 2025-07-10 14:45:16 优化了代码结构，但是未解决上述问题
# 2025-07-10 15:20:51 可以对视频进行分割，包括多目标追踪，保存也没有问题 只是PKl里IMU字段为空，仍然需要解决
# 2025-07-10 15:31:03 解决了pkl文件的里IMU字段为空的问题  为视频文件添加点云等3D信息
# 2025-07-10 16:05:22 修改了保存的分割文件名，暂为解决上个 
# 2025-07-10 16:22:34 修改了extract_video_frames函数，可以按照视频长度提取不同的帧数
# 2025-07-10 18:23:42 修改模型加载的问题，可以手动指定模型的位置了，TODO 添加默认模型位置的功能 (DONE: 2025-07-10 19:08:29)
# 2025-07-10 21:07:36 TODO  1. 为视频文件添加点云等3D信息  2. 不能解决视频里对象的出现和消失的问题，只能针对一直出现在视频里的对象进行追踪和分割