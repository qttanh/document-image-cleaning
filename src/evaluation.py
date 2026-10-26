from pathlib import Path
import csv
import cv2
import numpy as np

# Thư mục gốc chứa 4 folder con và thư mục kết quả sau Morphology
DIR_ORIGINAL = Path("data/preprocessing_experiments/all_processed")
DIR_PROCESSED = Path("data/processed")
OUTPUT_REPORT = Path("evaluation_no_reference_report.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image_utf8(path: Path) -> np.ndarray:
    """Đọc ảnh an toàn hỗ trợ đường dẫn tiếng Việt và dấu tiếng Việt."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return img


def build_original_index(orig_dir: Path) -> dict:
    """
    Quét đệ quy toàn bộ các thư mục con (dom_ban, dut_net, nhieu_hat, sang_khong_deu)
    và lập danh bạ tìm kiếm ảnh gốc.
    """
    index = {}
    for p in orig_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            # Lưu theo cả tên đầy đủ và stem (tên không kèm đuôi file)
            index[p.name.lower()] = p
            index[p.stem.lower()] = p
    return index


def find_matched_original(proc_path: Path, orig_index: dict) -> Path | None:
    """Ghép cặp thông minh giữa ảnh đã xử lý và ảnh gốc trong các folder con."""
    stem_clean = proc_path.stem.lower()

    # 1. Khớp trực tiếp tên file hoặc stem
    if stem_clean in orig_index:
        return orig_index[stem_clean]

    # 2. Xử lý trường hợp tên bị ghép folder (ví dụ dom_ban_dom_ban_01 -> dom_ban_01)
    for orig_stem, orig_path in orig_index.items():
        if stem_clean.endswith(orig_stem) or orig_stem.endswith(stem_clean):
            return orig_path

    return None


def evaluate_no_reference(original_gray: np.ndarray, binary_img: np.ndarray) -> dict:
    """Đo đạc chất lượng ảnh nhị phân dựa trên thông tin cạnh và nền thực tế."""
    h, w = binary_img.shape
    total_pixels = h * w

    _, bin_clean = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    ink_mask = (bin_clean == 0)
    dark_ratio = float(np.mean(ink_mask))

    # 1. Đo độ bám biên chữ thực tế (Edge Recall so với Gradient ảnh gốc)
    smooth = cv2.GaussianBlur(original_gray, (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    strong_edges = grad > max(float(np.percentile(grad, 92)), 2.5 * float(grad.mean()))

    bin_edge = cv2.morphologyEx(bin_clean, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if np.any(bin_edge) and np.any(strong_edges):
        dist = cv2.distanceTransform((~bin_edge).astype(np.uint8), cv2.DIST_L2, 3)
        edge_recall = float(np.mean(dist[strong_edges] <= 2.5)) * 100.0
    else:
        edge_recall = 0.0

    # 2. Phân tích thành phần liên thông (Connected Components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask.astype(np.uint8), connectivity=8
    )

    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    tiny_speckles = int(np.sum(areas <= 3)) if areas.size else 0
    tiny_speckle_ratio = (tiny_speckles / len(areas) * 100.0) if areas.size else 0.0
    large_blocks = int(np.sum(areas > 0.05 * total_pixels)) if areas.size else 0

    bg_purity = max(0.0, 100.0 - tiny_speckle_ratio)

    # 3. Điểm đánh giá chất lượng (Quality Score - Thang 100)
    score = 100.0

    if dark_ratio < 0.005:
        score -= 70.0
    elif dark_ratio < 0.015:
        score -= 20.0

    if dark_ratio > 0.35:
        score -= 50.0
    if large_blocks > 0:
        score -= 40.0 * large_blocks

    if tiny_speckles > 100:
        score -= min(30.0, (tiny_speckles - 100) * 0.1)

    score -= 35.0 * (1.0 - min(edge_recall / 75.0, 1.0))
    score = max(0.0, min(100.0, score))

    return {
        "Edge_Recall_%": round(edge_recall, 2),
        "Background_Purity_%": round(bg_purity, 2),
        "Dark_Ratio_%": round(dark_ratio * 100.0, 2),
        "Tiny_Speckles": tiny_speckles,
        "Quality_Score": round(score, 2),
    }


def main():
    if not DIR_ORIGINAL.exists():
        print(f"LỖI: Không tìm thấy thư mục gốc [{DIR_ORIGINAL}]")
        return
    if not DIR_PROCESSED.exists():
        print(f"LỖI: Không tìm thấy thư mục kết quả [{DIR_PROCESSED}]")
        return

    print("Đang quét ảnh gốc từ tất cả các thư mục con...")
    orig_index = build_original_index(DIR_ORIGINAL)
    print(f"-> Đã lập danh bạ {len(orig_index) // 2} ảnh gốc từ 4 nhóm.")

    processed_files = sorted([
        p for p in DIR_PROCESSED.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not processed_files:
        print(f"Không tìm thấy ảnh kết quả nào trong [{DIR_PROCESSED}].")
        return

    rows = []
    print(f"\nBắt đầu đánh giá {len(processed_files)} ảnh...")
    print("-" * 80)
    print(f"{'Tên File':<32} | {'Edge Recall':<12} | {'Sạch Nền':<10} | {'Điểm (100)':<10}")
    print("-" * 80)

    for proc_path in processed_files:
        orig_path = find_matched_original(proc_path, orig_index)

        if orig_path is None:
            print(f"Bỏ qua [{proc_path.name}]: Không tìm thấy ảnh gốc trong 4 folder con.")
            continue

        orig_img = read_image_utf8(orig_path)
        proc_img = read_image_utf8(proc_path)

        if orig_img.shape != proc_img.shape:
            proc_img = cv2.resize(
                proc_img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        metrics = evaluate_no_reference(orig_img, proc_img)
        metrics["File"] = proc_path.name
        metrics["Original_Folder"] = orig_path.parent.name
        rows.append(metrics)

        print(f"{proc_path.name:<32} | {metrics['Edge_Recall_%']:>10}% | {metrics['Background_Purity_%']:>8}% | {metrics['Quality_Score']:>9}")

    if not rows:
        print("Không có ảnh nào khớp cặp để đánh giá.")
        return

    avg_edge = np.mean([r["Edge_Recall_%"] for r in rows])
    avg_purity = np.mean([r["Background_Purity_%"] for r in rows])
    avg_score = np.mean([r["Quality_Score"] for r in rows])

    print("=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH TOÀN BỘ 4 NHÓM ({len(rows)} ảnh):")
    print(f"- Độ bám biên chữ trung bình (Edge Recall) : {avg_edge:.2f} %")
    print(f"- Độ tinh khiết nền trung bình (Purity)    : {avg_purity:.2f} %")
    print(f"- Điểm chất lượng trung bình (Quality)     : {avg_score:.2f} / 100")
    print("=" * 80)

    fieldnames = [
        "File", "Original_Folder", "Quality_Score",
        "Edge_Recall_%", "Background_Purity_%", "Dark_Ratio_%", "Tiny_Speckles"
    ]
    with open(OUTPUT_REPORT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "File": "TRUNG_BINH",
            "Original_Folder": "ALL",
            "Quality_Score": round(avg_score, 2),
            "Edge_Recall_%": round(avg_edge, 2),
            "Background_Purity_%": round(avg_purity, 2),
            "Dark_Ratio_%": "-",
            "Tiny_Speckles": "-",
        })

    print(f"\nĐã xuất báo cáo chi tiết theo từng nhóm vào: {OUTPUT_REPORT.resolve()}")


if __name__ == "__main__":
    main()