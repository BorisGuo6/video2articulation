# iTACO 数据采集指南

本文档说明如何采集 iTACO 系统所需的输入数据。

## 硬件要求

- **iPhone 12 Pro** 或更新机型（必须带 LiDAR 传感器）
- 支持的设备：iPhone 12 Pro/Pro Max, iPhone 13 Pro/Pro Max, iPhone 14 Pro/Pro Max, iPhone 15 Pro/Pro Max, iPad Pro (2020+)

## 软件要求

| App | 用途 | 价格 |
|-----|------|------|
| [Record3D](https://record3d.app/) | 录制交互视频 | 免费 |
| [Polycam](https://poly.cam/) | 静态 3D 扫描 | 免费（基础功能） |

---

## 数据结构

最终需要准备以下目录结构：

```
your_object/
├── video_rgb/           # 交互视频 RGB 帧
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video_depth/         # 交互视频深度图
│   ├── 000000.depth
│   ├── 000001.depth
│   └── ...
├── surface_rgb/         # 静态扫描 RGB 关键帧
│   ├── 7324108294.jpg
│   └── ...
├── surface_depth/       # 静态扫描深度图
│   ├── 7324108294.png
│   └── ...
└── metadata/
    ├── metadata.json    # 相机内参
    └── surface/
        ├── surface.ply  # 3D 点云
        ├── mesh_info.json
        └── keyframes/
            └── corrected_cameras/
                ├── 7324108294.json
                └── ...
```

---

## Step 1: 录制交互视频 (Record3D)

### 1.1 安装和设置

1. 从 App Store 下载 [Record3D](https://apps.apple.com/app/record3d/id1493712442)
2. 打开 App，授予相机权限
3. 点击右上角设置，确保：
   - **Depth Sensor**: LiDAR
   - **Export Format**: RGBD Video 或 Shareable

### 1.2 录制步骤

1. 将物体放置在稳定的平面上
2. 打开 Record3D，对准物体
3. 点击红色录制按钮开始录制
4. **用手操作物体**（如打开书页、开关抽屉等）
5. 录制 2-5 秒的操作过程
6. 点击停止按钮

### 1.3 导出数据

1. 在 App 中点击录制好的视频
2. 点击 "Share" → "Export as RGBD Frames"
3. 选择导出到电脑（通过 AirDrop/文件 App/iTunes）
4. 得到的文件：
   - `rgb/` 目录 → 重命名为 `video_rgb/`
   - `depth/` 目录 → 重命名为 `video_depth/`
   - `metadata.json` → 复制到 `metadata/metadata.json`

### 1.4 文件格式

| 文件类型 | 格式 | 分辨率 |
|---------|------|--------|
| RGB | `.jpg` | 960×720 或 1920×1440 |
| Depth | `.depth` (二进制) | 192×256 (LiDAR 原始分辨率) |

---

## Step 2: 静态表面扫描 (Polycam)

### 2.1 安装和设置

1. 从 App Store 下载 [Polycam](https://apps.apple.com/app/polycam/id1501811181)
2. 打开 App，授予相机权限
3. 选择 "LiDAR" 扫描模式

### 2.2 扫描步骤

1. **物体保持静止**（与录制视频时的初始状态相同）
2. 打开 Polycam，选择 "LiDAR" 模式
3. 对准物体开始扫描
4. **缓慢围绕物体移动**，从多个角度拍摄
5. 确保覆盖物体的所有可见表面
6. 拍摄约 20-40 张关键帧
7. 等待 App 完成 3D 重建

### 2.3 导出数据

1. 扫描完成后，点击项目
2. 点击 "Export" → 选择 "Raw Data" 格式
3. 导出到电脑
4. 解压得到的文件包含：

| 原始路径 | 目标路径 |
|---------|---------|
| `keyframes/corrected_images/*.jpg` | `surface_rgb/` |
| `keyframes/depth/*.png` | `surface_depth/` |
| `keyframes/corrected_cameras/*.json` | `metadata/surface/keyframes/corrected_cameras/` |
| `mesh_info.json` | `metadata/surface/mesh_info.json` |
| `surface.ply` 或 `texturedMesh.ply` | `metadata/surface/surface.ply` |

---

## 参数说明

### metadata.json (来自 Record3D)

```json
{
  "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],  // 3x3 相机内参矩阵（按行展开）
  "poses": [...]  // 每帧的相机位姿
}
```

### corrected_cameras/*.json (来自 Polycam)

```json
{
  "fx": 772.48,           // 焦距 X
  "fy": 772.48,           // 焦距 Y
  "cx": 499.29,           // 主点 X
  "cy": 388.28,           // 主点 Y
  "width": 1024,          // 图像宽度
  "height": 768,          // 图像高度
  "t_00": ..., "t_01": ..., "t_02": ..., "t_03": ...,  // 外参矩阵第1行
  "t_10": ..., "t_11": ..., "t_12": ..., "t_13": ...,  // 外参矩阵第2行
  "t_20": ..., "t_21": ..., "t_22": ..., "t_23": ...,  // 外参矩阵第3行
  "timestamp": 7324108294,
  "blur_score": 91.96,    // 清晰度评分
  "neighbors": [...]      // 相邻帧列表
}
```

### mesh_info.json

```json
{
  "alignmentTransform": [...]  // 4x4 对齐变换矩阵（按行展开）
}
```

---

## 注意事项

### 拍摄建议

1. **光照**：确保均匀照明，避免强烈阴影
2. **背景**：使用简单、非反光的背景
3. **稳定性**：拍摄时尽量保持手稳
4. **对齐**：交互视频的第一帧应与静态扫描的物体状态一致

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| 深度图分辨率太低 | 正常现象，pipeline 会用 PromptDA 放大 |
| Polycam 导出没有 depth | 确保使用 "Raw Data" 格式导出 |
| 文件名不匹配 | Polycam 使用时间戳命名，不影响处理 |
| 手部遮挡物体 | 脚本会自动用 Grounded-SAM-2 去除手部 |

---

## 示例数据

参考 `example/book/` 目录的结构：

```
example/book/
├── video_rgb/        # 50 帧交互视频
├── video_depth/      # 50 张深度图
├── surface_rgb/      # 31 张静态扫描
├── surface_depth/    # 31 张深度图
└── metadata/         # 元数据文件
```

---

## 运行 Pipeline

数据准备好后，将文件夹放入 `example/` 目录，修改 `run_example.sh` 中的路径：

```bash
DATA_NAME="your_object"
VIDEO_RGB_DIR="$SCRIPT_DIR/example/your_object/video_rgb"
VIDEO_DEPTH_DIR="$SCRIPT_DIR/example/your_object/video_depth"
SURFACE_RGB_DIR="$SCRIPT_DIR/example/your_object/surface_rgb"
SURFACE_DEPTH_DIR="$SCRIPT_DIR/example/your_object/surface_depth"
```

然后运行：

```bash
./run_example.sh
```

