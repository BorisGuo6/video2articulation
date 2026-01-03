#!/bin/bash
# =============================================================================
# iTACO Example Pipeline - 使用 book 示例数据
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# 配置参数
# =============================================================================

# 数据类型
DATA_TYPE="real"

# 数据名称
DATA_NAME="book_example"

# 输入数据路径 (使用绝对路径)
VIDEO_RGB_DIR="$SCRIPT_DIR/example/book/video_rgb"
VIDEO_DEPTH_DIR="$SCRIPT_DIR/example/book/video_depth"
SURFACE_RGB_DIR="$SCRIPT_DIR/example/book/surface_rgb"
SURFACE_DEPTH_DIR="$SCRIPT_DIR/example/book/surface_depth"

# MonST3R 运动阈值
MOTION_MASK_THRESH=0.35

# =============================================================================
# 自动设置路径
# =============================================================================

DATA_ROOT="real_data"
VIEW_DIR="${DATA_ROOT}/raw_data/${DATA_NAME}"
PREPROCESS_DIR="${DATA_ROOT}/exp_results/preprocessing/${DATA_NAME}"
PREDICTION_DIR="${DATA_ROOT}/exp_results/prediction/${DATA_NAME}"

# =============================================================================
# 辅助函数
# =============================================================================

print_step() {
    echo ""
    echo "=============================================="
    echo "Step: $1"
    echo "=============================================="
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: Directory not found: $1"
        exit 1
    fi
}

# =============================================================================
# 参数检查
# =============================================================================

check_dir "$VIDEO_RGB_DIR"
check_dir "$VIDEO_DEPTH_DIR"
check_dir "$SURFACE_RGB_DIR"
check_dir "$SURFACE_DEPTH_DIR"

echo "Input data:"
echo "  Video RGB:    $VIDEO_RGB_DIR ($(ls "$VIDEO_RGB_DIR" | wc -l) frames)"
echo "  Video Depth:  $VIDEO_DEPTH_DIR ($(ls "$VIDEO_DEPTH_DIR" | wc -l) maps)"
echo "  Surface RGB:  $SURFACE_RGB_DIR ($(ls "$SURFACE_RGB_DIR" | wc -l) images)"
echo "  Surface Depth:$SURFACE_DEPTH_DIR ($(ls "$SURFACE_DEPTH_DIR" | wc -l) maps)"

# =============================================================================
# Step 0: 准备数据目录结构
# =============================================================================

print_step "0. Preparing data directory structure"

mkdir -p "$VIEW_DIR/rgb"
mkdir -p "$VIEW_DIR/depth"
mkdir -p "$VIEW_DIR/rgb_reverse"
mkdir -p "$VIEW_DIR/surface/keyframes/corrected_images"
mkdir -p "$VIEW_DIR/surface/keyframes/depth"
mkdir -p "$PREPROCESS_DIR"
mkdir -p "$PREDICTION_DIR"

# 复制视频数据
echo "Copying video RGB frames..."
cp -r "$VIDEO_RGB_DIR"/* "$VIEW_DIR/rgb/"

echo "Copying video depth maps..."
cp -r "$VIDEO_DEPTH_DIR"/* "$VIEW_DIR/depth/"

# 创建反向视频帧 (用于SAM2, 需要.jpg格式)
echo "Creating reversed video frames..."
cd "$VIEW_DIR/rgb"
files=($(ls -1 | sort))
count=${#files[@]}
for i in "${!files[@]}"; do
    new_idx=$((count - 1 - i))
    new_name=$(printf "%05d.jpg" $new_idx)
    cp "${files[$i]}" "../rgb_reverse/$new_name"
done
cd "$SCRIPT_DIR"

# 复制表面数据
echo "Copying surface RGB images..."
cp -r "$SURFACE_RGB_DIR"/* "$VIEW_DIR/surface/keyframes/corrected_images/"

echo "Copying surface depth maps..."
cp -r "$SURFACE_DEPTH_DIR"/* "$VIEW_DIR/surface/keyframes/depth/"

# 复制元数据文件 (从 example/book/metadata 获取)
echo "Copying metadata files..."
METADATA_DIR="$SCRIPT_DIR/example/book/metadata"
if [ -f "$METADATA_DIR/metadata.json" ]; then
    cp "$METADATA_DIR/metadata.json" "$VIEW_DIR/"
fi
if [ -f "$METADATA_DIR/surface/mesh_info.json" ]; then
    cp "$METADATA_DIR/surface/mesh_info.json" "$VIEW_DIR/surface/"
fi
if [ -d "$METADATA_DIR/surface/keyframes/corrected_cameras" ]; then
    cp -r "$METADATA_DIR/surface/keyframes/corrected_cameras" "$VIEW_DIR/surface/keyframes/"
fi
if [ -f "$METADATA_DIR/surface/surface.ply" ]; then
    cp "$METADATA_DIR/surface/surface.ply" "$VIEW_DIR/surface/"
fi

# 采样视频帧 (MonST3R需要)
echo "Sampling video frames for MonST3R..."
mkdir -p "$VIEW_DIR/sample_rgb"
cd "$VIEW_DIR/rgb"
files=($(ls -1 | sort))
count=${#files[@]}
# 均匀采样最多18帧
if [ $count -gt 18 ]; then
    step=$((count / 18))
    for i in $(seq 0 17); do
        idx=$((i * step))
        if [ $idx -lt $count ]; then
            new_name=$(printf "%05d.png" $i)
            cp "${files[$idx]}" "../sample_rgb/$new_name"
        fi
    done
else
    for i in "${!files[@]}"; do
        new_name=$(printf "%05d.png" $i)
        cp "${files[$i]}" "../sample_rgb/$new_name"
    done
fi
cd "$SCRIPT_DIR"

echo "Data directory structure prepared at: $VIEW_DIR"
echo "  - rgb/:          $(ls "$VIEW_DIR/rgb" | wc -l) frames"
echo "  - depth/:        $(ls "$VIEW_DIR/depth" | wc -l) maps"
echo "  - rgb_reverse/:  $(ls "$VIEW_DIR/rgb_reverse" | wc -l) frames"
echo "  - sample_rgb/:   $(ls "$VIEW_DIR/sample_rgb" | wc -l) sampled frames"
echo "  - surface/:      $(ls "$VIEW_DIR/surface/keyframes/corrected_images" | wc -l) images"

# =============================================================================
# Step 1: MonST3R - 计算视频运动图
# =============================================================================

print_step "1. Running MonST3R for motion segmentation"

cd monst3r
conda run -n monst3r python demo.py \
    --input "../$VIEW_DIR/sample_rgb/" \
    --output_dir "../$PREPROCESS_DIR/" \
    --seq_name monst3r \
    --motion_mask_thresh $MOTION_MASK_THRESH
cd "$SCRIPT_DIR"

echo "MonST3R completed. Results in: $PREPROCESS_DIR/monst3r/"

# =============================================================================
# Step 2: AutoSeg-SAM2 - 视频部件分割
# =============================================================================

print_step "2. Running AutoSeg-SAM2 for part segmentation"

cd AutoSeg-SAM2
conda run -n video_articulation python auto-mask-batch.py \
    --video_path "../$VIEW_DIR/rgb_reverse" \
    --output_dir "../$PREPROCESS_DIR/video_segment_reverse" \
    --batch_size 10 \
    --detect_stride 5 \
    --level small \
    --pred_iou_thresh 0.9 \
    --stability_score_thresh 0.95
cd "$SCRIPT_DIR"

echo "SAM2 segmentation completed."

# =============================================================================
# Step 3: PromptDA - 深度图放大
# =============================================================================

print_step "3. Running PromptDA for depth upscaling"

cd PromptDA

# 表面深度放大
conda run -n video_articulation python scale_depth.py \
    --image_dir "../$VIEW_DIR/surface/keyframes/corrected_images/" \
    --depth_dir "../$VIEW_DIR/surface/keyframes/depth/" \
    --save_dir "../$PREPROCESS_DIR/prompt_depth_surface"

# 视频深度放大
conda run -n video_articulation python scale_depth.py \
    --image_dir "../$VIEW_DIR/rgb/" \
    --depth_dir "../$VIEW_DIR/depth/" \
    --save_dir "../$PREPROCESS_DIR/prompt_depth_video"

cd "$SCRIPT_DIR"
echo "Depth upscaling completed."

# =============================================================================
# Step 4: Grounded-SAM-2 - 手部遮罩
# =============================================================================

print_step "4. Running Grounded-SAM-2 for hand masking"

cd Grounded-SAM-2
conda run -n video_articulation python mask_hand.py \
    --video_frame_dir "../$VIEW_DIR/rgb/" \
    --save_dir "../$PREPROCESS_DIR/hand_mask/"
cd "$SCRIPT_DIR"

echo "Hand masking completed."

# =============================================================================
# Step 5: 对齐表面和视频坐标
# =============================================================================

print_step "5. Aligning surface and video coordinates"

conda run -n video_articulation python align_surface_video.py \
    --view_dir "$VIEW_DIR" \
    --preprocess_dir "$PREPROCESS_DIR"

echo "Coordinate alignment completed."

# =============================================================================
# Step 6: 粗预测
# =============================================================================

print_step "6. Running coarse prediction"

conda run -n video_articulation python joint_coarse_prediction.py \
    --data_type $DATA_TYPE \
    --view_dir "$VIEW_DIR/" \
    --preprocess_dir "$PREPROCESS_DIR/" \
    --prediction_dir "$PREDICTION_DIR/" \
    --mask_type monst3r

echo "Coarse prediction completed."

# =============================================================================
# Step 7: 精细优化
# =============================================================================

print_step "7. Running refinement optimization"

conda run -n video_articulation python launch_joint_refinement.py \
    --data_type $DATA_TYPE \
    --exp_name refinement \
    --view_dir "$VIEW_DIR/" \
    --preprocess_dir "$PREPROCESS_DIR/" \
    --prediction_dir "$PREDICTION_DIR/" \
    --mask_type monst3r \
    --loss chamfer

echo "Refinement completed."

# =============================================================================
# Step 8: 网格重建 (NKSR)
# =============================================================================

print_step "8. Running mesh reconstruction with NKSR"

REFINEMENT_DIR="$PREDICTION_DIR/refinement/monst3r/chamfer/0"

conda run -n nksr python extract_mesh.py \
    --data_type $DATA_TYPE \
    --view_dir "$VIEW_DIR/" \
    --preprocess_dir "$PREPROCESS_DIR/" \
    --refinement_results_dir "$REFINEMENT_DIR/"

echo "Mesh reconstruction completed."

# =============================================================================
# 完成
# =============================================================================

print_step "Pipeline completed!"

echo ""
echo "Results saved to:"
echo "  - View data:     $VIEW_DIR/"
echo "  - Preprocessing: $PREPROCESS_DIR/"
echo "  - Predictions:   $PREDICTION_DIR/"
echo "  - Final Mesh:    $REFINEMENT_DIR/"

# =============================================================================
# Step 9: 保存中间成果到 examples 文件夹
# =============================================================================

print_step "9. Saving results to examples folder"

EXAMPLE_RESULTS="$SCRIPT_DIR/example/book/results"
mkdir -p "$EXAMPLE_RESULTS"

# 保存预处理结果
echo "Saving preprocessing results..."
mkdir -p "$EXAMPLE_RESULTS/preprocessing"
cp -r "$PREPROCESS_DIR/monst3r" "$EXAMPLE_RESULTS/preprocessing/" 2>/dev/null || true
cp -r "$PREPROCESS_DIR/video_segment_reverse" "$EXAMPLE_RESULTS/preprocessing/" 2>/dev/null || true
cp "$PREPROCESS_DIR/cam2world.npy" "$EXAMPLE_RESULTS/preprocessing/" 2>/dev/null || true

# 保存预测结果
echo "Saving prediction results..."
mkdir -p "$EXAMPLE_RESULTS/prediction"
cp -r "$PREDICTION_DIR"/* "$EXAMPLE_RESULTS/prediction/" 2>/dev/null || true

# 保存最终 mesh
echo "Saving final mesh..."
mkdir -p "$EXAMPLE_RESULTS/mesh"
cp "$REFINEMENT_DIR"/*.ply "$EXAMPLE_RESULTS/mesh/" 2>/dev/null || true
cp "$REFINEMENT_DIR"/*.npy "$EXAMPLE_RESULTS/mesh/" 2>/dev/null || true

echo ""
echo "Results saved to: $EXAMPLE_RESULTS/"
echo ""
echo "Output files:"
ls -lh "$EXAMPLE_RESULTS/mesh/"*.ply 2>/dev/null || echo "  (no .ply files found)"

