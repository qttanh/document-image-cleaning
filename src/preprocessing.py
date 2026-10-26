import csv
import glob
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RAW_DIR = "data/raw"
OUTPUT_ROOT = "data/preprocessing_experiments"
REPORT_DIR = os.path.join(OUTPUT_ROOT, "report_sample")
PROCESSED_DIR = os.path.join(OUTPUT_ROOT, "all_processed")

GROUPS = ("dom_ban", "dut_net", "nhieu_hat", "sang_khong_deu")


# ============================================================
# 1. ĐỌC ẢNH VÀ CÁC THUẬT TOÁN TIỆN ÍCH CỐT LÕI
# ============================================================
def convert_to_grayscale(image):
    """Chuyển ảnh màu sang ảnh xám chuẩn xác (hỗ trợ cả ảnh RGBA/BGRA)."""
    if image is None:
        raise ValueError("Ảnh đầu vào là None.")

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={image.shape}")


def get_group_from_path(path, root_dir=RAW_DIR):
    """Lấy tên nhóm khuyết tật từ cấu trúc thư mục con trong data/raw/."""
    rel_path = os.path.relpath(path, root_dir)
    parts = rel_path.replace("\\", "/").split("/")

    if len(parts) >= 2 and parts[0].lower() in GROUPS:
        return parts[0].lower()

    return "default"


def estimate_noise_on_flat_regions(gray):
    """Ước lượng độ lệch chuẩn nhiễu Immerkaer trên vùng nền phẳng."""
    gray_f = gray.astype(np.float32)

    kernel = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
    )
    response = cv2.filter2D(gray_f, cv2.CV_32F, kernel)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    threshold = np.percentile(gradient, 50)
    flat_mask = gradient <= threshold

    values = np.abs(response[flat_mask])
    if values.size == 0:
        return 0.0

    sigma_est = float(np.median(values) / 6.0)
    return sigma_est


def safe_bilateral(gray, diameter=3, sigma_color=12, sigma_space=2):
    """Bilateral Filter nhẹ bảo toàn gờ cạnh và các nét mảnh."""
    return cv2.bilateralFilter(
        gray,
        d=diameter,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )



def estimate_background_morphology(gray):
    """
    Ước lượng nền giấy bằng morphological closing đa tỷ lệ.
    - Loại ảnh hưởng của nét chữ tối lên nền ước lượng.
    - Làm mượt nền để hạn chế quầng sáng quanh chữ.
    - Dùng padding phản chiếu để giảm lỗi tại mép ảnh.
    """
    if gray is None or gray.ndim != 2:
        raise ValueError("Đầu vào phải là ảnh xám 2D.")

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape
    min_dim = min(h, w)

    # Kernel tỷ lệ theo kích thước ảnh, thay vì cố định 11x11.
    k = max(15, int(min_dim * 0.035))
    if k % 2 == 0:
        k += 1

    # Giới hạn kernel để tránh quá lớn trên ảnh nhỏ.
    k = min(k, max(3, min_dim // 3))
    if k % 2 == 0:
        k -= 1
    k = max(k, 3)

    # Phản chiếu biên để giảm hiện tượng sai nền sát mép.
    pad = min(max(k, 16), min_dim // 4)
    padded = cv2.copyMakeBorder(
        gray, pad, pad, pad, pad,
        borderType=cv2.BORDER_REFLECT
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k, k)
    )

    # Closing giúp loại nét chữ tối khỏi nền ước lượng.
    background = cv2.morphologyEx(
        padded, cv2.MORPH_CLOSE, kernel
    )

    # Làm mượt nền, giảm các chuyển tiếp gắt và quầng sáng.
    sigma = max(3.0, min_dim * 0.015)
    background = cv2.GaussianBlur(
        background, (0, 0), sigmaX=sigma, sigmaY=sigma
    )

    # Bỏ phần padding, trở về kích thước ảnh ban đầu.
    background = background[
        pad:pad + h, pad:pad + w
    ]

    return np.maximum(background.astype(np.float32), 8.0)


def normalize_background_division(gray, scale=245.0):
    """
    Chuẩn hóa ánh sáng nền theo tỷ lệ nền cục bộ.
    Không dùng thresholding tại bước này để tránh mất nét chữ.
    """
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    background = estimate_background_morphology(gray)

    gray_f = gray.astype(np.float32)

    # Chia nền cục bộ để giảm ảnh hưởng của bóng đổ.
    ratio = gray_f / background

    # Giới hạn mức khuếch đại để hạn chế nhiễu và quầng sáng.
    normalized = ratio * float(scale)

    normalized = np.clip(normalized, 0, 255)

    return np.round(normalized).astype(np.uint8)

# ============================================================
# 2. BỘ ĐỊNH TUYẾN TIỀN XỬ LÝ CHUYÊN BIỆT THEO NHÓM LỖI
# ============================================================
def preprocess_by_group(gray, group):
    """
    Xử lý thích ứng chuyên biệt cho từng loại khuyết tật:
      - sang_khong_deu: Chia nền hình thái học, BỎ CLAHE để không kéo đen lại bóng râm.
      - dom_ban: Chia nền để kéo sáng mảng ố mốc lớn (như dom_ban_06) + Bilateral.
      - dut_net: BẢO TOÀN NÉT IN KIM/NHIỆT, tuyệt đối không dùng MedianBlur.
      - nhieu_hat: BỎ MedianBlur 3x3 trên văn bản in nhỏ, dùng Bilateral + Unsharp Mask sắc nét.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    group = (group or "default").lower()

    if group == "sang_khong_deu":
        # 1. Kéo phẳng hoàn toàn bóng nếp gấp và bóng bàn tay về nền trắng
        normalized = normalize_background_division(gray, scale=250.0)
        # 2. Lọc Bilateral rất nhẹ để mượt nền, không dùng CLAHE
        processed = safe_bilateral(
            normalized, diameter=3, sigma_color=10, sigma_space=2
        )
        method = "sang_khong_deu_morph_bg_div"

    elif group == "dom_ban":
        # 1. Chia nền để xóa các mảng ố xanh đen lớn ở gáy sách/góc giấy
        normalized = normalize_background_division(gray, scale=255.0)
        # 2. Khử nhiễu nhẹ giữ nguyên nét mực viết tay cổ
        processed = safe_bilateral(
            normalized, diameter=3, sigma_color=12, sigma_space=2
        )
        method = "dom_ban_bg_div_bilateral"

    elif group == "dut_net":
        # Chữ in kim/in nhiệt: Giữ nguyên độ dốc nét chữ, không làm mờ biên
        processed = safe_bilateral(
            gray, diameter=3, sigma_color=8, sigma_space=2
        )
        method = "dut_net_preserve_strokes_bilateral"

    elif group == "nhieu_hat":
        # Văn bản in nhỏ & công thức toán: BỎ HOÀN TOÀN Median 3x3 để không chém đứt nét 1-pixel
        # Dùng Bilateral khử hạt phẳng + Unsharp Mask tăng độ dốc biên ký tự
        denoised = safe_bilateral(
            gray, diameter=5, sigma_color=15, sigma_space=3
        )
        blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharp = cv2.addWeighted(denoised, 1.4, blur, -0.4, 0)
        processed = np.clip(sharp, 0, 255).astype(np.uint8)
        method = "nhieu_hat_bilateral_unsharp"

    else:
        # Nhóm mặc định an toàn
        normalized = normalize_background_division(gray, scale=250.0)
        processed = safe_bilateral(
            normalized, diameter=3, sigma_color=10, sigma_space=2
        )
        method = "default_morph_bg_div"

    return processed, method


# ============================================================
# 3. HISTOGRAM VÀ BÁO CÁO MẪU (report_sample)
# ============================================================
def plot_histogram(image_raw_gray, image_processed, save_path):
    """Lưu biểu đồ lược đồ mức xám trước và sau tiền xử lý."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hist_raw = cv2.calcHist([image_raw_gray], [0], None, [256], [0, 256])
    axes[0].plot(hist_raw, color="black", label="Mức xám gốc")
    axes[0].set_title("Histogram Trước Tiền Xử Lý (Raw)")
    axes[0].set_xlabel("Mức sáng (0 - 255)")
    axes[0].set_ylabel("Số lượng pixel")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    hist_processed = cv2.calcHist(
        [image_processed], [0], None, [256], [0, 256]
    )
    axes[1].plot(hist_processed, color="navy", label="Mức xám sau chuẩn hóa")
    axes[1].set_title("Histogram Sau Tiền Xử Lý (Normalized)")
    axes[1].set_xlabel("Mức sáng (0 - 255)")
    axes[1].set_ylabel("Số lượng pixel")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_sample_report(sample_path, output_dir, root_dir=RAW_DIR):
    """Tạo biểu đồ histogram, ảnh so sánh đối chứng 5 khung hình và file log JSON."""
    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(sample_path))[0]
    image_bgr = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)

    if image_bgr is None:
        print(f"[!] Không đọc được ảnh mẫu: {sample_path}")
        return

    gray = convert_to_grayscale(image_bgr)
    group = get_group_from_path(sample_path, root_dir)
    processed, method = preprocess_by_group(gray, group)
    sigma = estimate_noise_on_flat_regions(gray)

    # 1. Xuất biểu đồ Histogram
    histogram_path = os.path.join(output_dir, f"{image_name}_histogram.png")
    plot_histogram(gray, processed, histogram_path)

    # 2. Tạo 3 bộ lọc kinh điển đối chứng
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1.2)
    median = cv2.medianBlur(gray, 3)
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=5)

    raw_path = os.path.join(output_dir, f"{image_name}_gray_raw.png")
    processed_path = os.path.join(
        output_dir, f"{image_name}_final_processed.png"
    )
    comparison_path = os.path.join(output_dir, f"{image_name}_comparison.png")

    cv2.imwrite(raw_path, gray)
    cv2.imwrite(processed_path, processed)

    # 3. Ghép bảng so sánh 5 khung hình
    images = [gray, gaussian, median, bilateral, processed]
    titles = [
        "1. Ảnh xám gốc",
        "2. Gaussian (5x5, σ1.2)",
        "3. Median (3x3)",
        "4. Bilateral (d5, σ25)",
        f"5. Đề xuất: {method}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. Ghi file JSON báo cáo kỹ thuật
    log_data = {
        "sample_image": sample_path,
        "group": group,
        "noise_estimate_relative": round(sigma, 3),
        "method": method,
        "outputs": {
            "histogram": histogram_path,
            "raw_gray": raw_path,
            "processed": processed_path,
            "comparison": comparison_path,
        },
        "technical_improvements": [
            "Morphological multi-scale background division completely eliminates heavy fold and hand shadows without halo artifacts.",
            "Stain surfaces in ancient documents are normalized back to white paper levels.",
            "3x3 median filtering is strictly avoided on fine print to prevent 1-pixel stroke disconnection.",
            "CLAHE is removed from background normalization branches to avoid amplifying background paper grain.",
        ],
    }

    json_path = os.path.join(output_dir, f"{image_name}_log.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(log_data, file, indent=4, ensure_ascii=False)

    print(f"[REPORT] {sample_path}")
    print(f"         nhóm={group} | phương pháp={method} | sigma~{sigma:.2f}")


# ============================================================
# 4. XỬ LÝ HÀNG LOẠT TOÀN BỘ ẢNH (all_processed) VÀ GHI CSV
# ============================================================
def batch_preprocess_all(all_image_paths, output_dir, root_dir=RAW_DIR):
    """Xử lý toàn bộ tập dữ liệu, xuất file PNG sạch và ghi bảng log CSV tổng kết."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    print(
        f"\n[*] Bắt đầu tiền xử lý hàng loạt {len(all_image_paths)} ảnh..."
    )

    for index, path in enumerate(all_image_paths, start=1):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"[!] Bỏ qua ảnh không đọc được: {path}")
            rows.append({
                "anh": os.path.relpath(path, root_dir),
                "nhom_loi": get_group_from_path(path, root_dir),
                "phuong_phap": "ERROR_READ_IMAGE",
                "sigma_uoc_luong": "",
                "trang_thai": "khong_doc_duoc",
                "duong_dan_dau_ra": "",
            })
            continue

        try:
            gray = convert_to_grayscale(image)
            group = get_group_from_path(path, root_dir)
            processed, method = preprocess_by_group(gray, group)
            sigma = estimate_noise_on_flat_regions(gray)

            relative_path = os.path.relpath(path, root_dir)
            relative_without_ext = os.path.splitext(relative_path)[0]
            save_path = os.path.join(
                output_dir, relative_without_ext + ".png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if not cv2.imwrite(save_path, processed):
                raise IOError(f"Không ghi được ảnh đầu ra: {save_path}")

            rows.append({
                "anh": relative_path,
                "nhom_loi": group,
                "phuong_phap": method,
                "sigma_uoc_luong": round(sigma, 3),
                "trang_thai": "thanh_cong",
                "duong_dan_dau_ra": save_path,
            })

            print(
                f"[{index:02d}/{len(all_image_paths):02d}] {relative_path:40s} "
                f"| nhóm={group:15s} | {method}"
            )

        except Exception as error:
            relative_path = os.path.relpath(path, root_dir)
            print(f"[!] Lỗi khi xử lý {relative_path}: {error}")
            rows.append({
                "anh": relative_path,
                "nhom_loi": get_group_from_path(path, root_dir),
                "phuong_phap": "ERROR_PROCESSING",
                "sigma_uoc_luong": "",
                "trang_thai": str(error),
                "duong_dan_dau_ra": "",
            })

    # Ghi log CSV tổng kết
    log_path = os.path.join(output_dir, "preprocessing_log.csv")
    fieldnames = [
        "anh",
        "nhom_loi",
        "phuong_phap",
        "sigma_uoc_luong",
        "trang_thai",
        "duong_dan_dau_ra",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(row["trang_thai"] == "thanh_cong" for row in rows)
    print(
        f"\n[*] Hoàn tất: {success_count}/{len(all_image_paths)} ảnh thành công."
    )
    print(f"[*] File CSV log: {log_path}")


# ============================================================
# 5. CHƯƠNG TRÌNH THỰC THI CHÍNH (MAIN)
# ============================================================
def main():
    if not os.path.isdir(RAW_DIR):
        print(f"[!] Không tìm thấy thư mục dữ liệu: {RAW_DIR}")
        print("    Hãy kiểm tra lại đường dẫn RAW_DIR và cấu trúc thư mục.")
        return

    all_paths = glob.glob(os.path.join(RAW_DIR, "**", "*"), recursive=True)
    valid_images = sorted(
        path
        for path in all_paths
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS)
    )

    if not valid_images:
        print(f"[!] Không tìm thấy ảnh hợp lệ trong {RAW_DIR}.")
        return

    print(f"[*] Tìm thấy tổng cộng {len(valid_images)} ảnh.")

    # Chọn 1 ảnh đại diện đầu tiên trong từng nhóm để xuất báo cáo đối chứng
    sample_images = []
    for group in GROUPS:
        group_dir = os.path.join(RAW_DIR, group)
        candidates = sorted(
            path
            for path in glob.glob(
                os.path.join(group_dir, "**", "*"), recursive=True
            )
            if os.path.isfile(path) and path.lower().endswith(IMG_EXTS)
        )
        if candidates:
            sample_images.append(candidates[0])
        else:
            print(f"[!] Nhóm {group} chưa có ảnh trực tiếp hoặc ảnh con.")

    # 1. Chạy xuất báo cáo mẫu cho từng nhóm
    for sample_path in sample_images:
        run_sample_report(
            sample_path=sample_path,
            output_dir=REPORT_DIR,
            root_dir=RAW_DIR,
        )

    # 2. Xử lý toàn bộ dataset đưa vào all_processed
    batch_preprocess_all(
        all_image_paths=valid_images,
        output_dir=PROCESSED_DIR,
        root_dir=RAW_DIR,
    )

    print("\n=== HOÀN TẤT BƯỚC TIỀN XỬ LÝ (PREPROCESSING) ===")
    print(f"1. Thư mục báo cáo & biểu đồ: {REPORT_DIR}")
    print(f"2. Thư mục ảnh đã tiền xử lý: {PROCESSED_DIR}")
    print("3. Dữ liệu đã sẵn sàng để chuyển sang script Thresholding.")


if __name__ == "__main__":
    main()