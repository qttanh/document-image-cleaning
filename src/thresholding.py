import os
import csv
import cv2
import numpy as np

INPUT_DIR = "data/preprocessing_experiments/all_processed"
OUTPUT_DIR = "data/thresholding_experiments/all_processed"
REPORT_DIR = "data/thresholding_experiments/comparison"
ALTERNATIVE_DIR = os.path.join(REPORT_DIR, "alternatives")

GROUPS = ("dom_ban", "dut_net", "nhieu_hat", "sang_khong_deu")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SAVE_TOP = 3
SAVE_COMPARISONS = True


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(ALTERNATIVE_DIR, exist_ok=True)


def read_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        color = img[:, :, :3].astype(np.float32)
        img = (color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img = remove_scanner_border(img)
    return img


def remove_scanner_border(gray):
    """Phát hiện và xóa vùng nền scanner màu đen bao quanh mép giấy."""
    h, w = gray.shape
    corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    if np.median(corners) < 50:
        dark_mask = (gray < 45).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        cleaned = gray.copy()
        for i in range(1, n):
            x, y, gw, gh, area = stats[i]
            if (x == 0 or y == 0 or (x + gw) >= w or (y + gh) >= h) and area > 0.005 * (h * w):
                cleaned[labels == i] = 255
        return cleaned
    return gray


def save_image(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        path += ".png"
        ext = ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Khong luu duoc anh: {path}")
    encoded.tofile(path)


def get_group(path):
    normalized = os.path.normpath(path).lower()
    parts = normalized.split(os.sep)
    for group in GROUPS:
        if group in parts or group in normalized:
            return group
    return "default"


def odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def estimate_stroke_width(gray):
    sm = cv2.GaussianBlur(gray, (0, 0), 1.0)
    _, bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) == 0:
        return 2.0
    dt = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    peaks = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1)
    vals = dt[peaks]
    if vals.size == 0:
        return 2.0
    return float(np.clip(2.0 * np.median(vals), 1.0, 15.0))


def flatten_background(gray, strength=0.80):
    h, w = gray.shape
    min_dim = min(h, w)
    sigma = max(10.0, min_dim * 0.04)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = np.maximum(bg.astype(np.float32), 15.0)
    img = gray.astype(np.float32)
    target = float(np.median(bg))
    normalized = np.clip(img / bg * target, 0, 255)
    normalized = np.clip((1.0 - strength) * img + strength * normalized, 0, 255)
    return normalized.astype(np.uint8)


def unsharp_mask(gray, sigma=1.5, strength=1.6):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def make_preprocessed(gray, group):
    candidates = [("original", gray.copy())]

    flat = flatten_background(gray, 0.75)
    candidates.append(("background_flatten", flat))

    sharp_strength = 2.2 if group == "nhieu_hat" else 1.5
    sharp = unsharp_mask(flat, sigma=1.2, strength=sharp_strength)
    candidates.append(("sharp_flat", sharp))

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(flat)
    candidates.append(("flat_clahe", cv2.addWeighted(flat, 0.75, clahe, 0.25, 0)))

    denoise_h = 3 if group in ("dut_net", "default") else 4
    den = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    candidates.append(("denoise_flat", flatten_background(den, 0.70)))

    if group == "dom_ban":
        median = cv2.medianBlur(gray, 3)
        candidates.append(("median3_flat", flatten_background(median, 0.75)))

    return candidates


# --- CÁC HÀM PHÂN NGƯỠNG ---

def otsu(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_gaussian(gray, window, c, min_std=7.0):
    window = min(odd(window), min(gray.shape) if min(gray.shape) % 2 else min(gray.shape) - 1)
    window = max(3, window)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, window, c)

    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    bw[std < min_std] = 255
    return bw


def sauvola(gray, window, k, R=128.0, min_std=7.0):
    window = min(odd(window), min(gray.shape) if min(gray.shape) % 2 else min(gray.shape) - 1)
    window = max(3, window)
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary = np.where((std >= min_std) & (f <= threshold), 0, 255).astype(np.uint8)
    return binary


def dual_threshold_binarize(gray, window=25, k_strict=0.35, k_loose=0.15, min_std=6.0):
    """
    Phân ngưỡng kép: dùng hạt giống strict để lọc sạch hạt tiêu,
    dùng loose để cứu nét đứt/mờ, dùng gradient để chống phình chữ.
    """
    h, w = gray.shape
    win = min(odd(window), min(h, w) if min(h, w) % 2 else min(h, w) - 1)
    win = max(3, win)

    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    contrast_gate = (std >= min_std)

    t_strict = mean * (1.0 + k_strict * (std / 128.0 - 1.0))
    t_loose = mean * (1.0 + k_loose * (std / 128.0 - 1.0))

    strict_bw = contrast_gate & (f <= t_strict)
    loose_bw = contrast_gate & (f <= t_loose)

    # Gradient mask: chống phình viền cho chữ mờ
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    if grad.size > 0:
        p70 = float(np.percentile(grad, 70))
        grad_mask = grad > p70
    else:
        grad_mask = np.ones_like(strict_bw)

    loose_bw = loose_bw & (strict_bw | grad_mask)

    # Ghép nối hạt giống bằng Connected Components (vectorized O(1))
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        loose_bw.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return np.full_like(gray, 255)

    strict_labels = np.unique(labels[strict_bw])
    strict_labels = strict_labels[strict_labels > 0]

    valid_mask = np.zeros(num_labels, dtype=bool)
    valid_mask[strict_labels] = True

    final_ink = valid_mask[labels]
    return np.where(final_ink, 0, 255).astype(np.uint8)


def make_threshold_candidates(gray, group):
    h, w = gray.shape
    sw = estimate_stroke_width(gray)
    min_dim = min(h, w)
    windows = sorted(set(
        min(odd(v), min_dim if min_dim % 2 else max(3, min_dim - 1))
        for v in (9, 13, max(17, 6 * sw), max(25, 12 * sw), min_dim * 0.07)
    ))
    windows = [max(3, v) for v in windows]

    if group == "dut_net":
        adaptive_cs = (3, 6, 9)
        sauvola_ks = (0.10, 0.18, 0.26)
        min_std = 3.5
        dt_params = [(0.25, 0.08)]
    elif group == "dom_ban":
        adaptive_cs = (6, 10, 14)
        sauvola_ks = (0.20, 0.28, 0.35)
        min_std = 5.5
        dt_params = [(0.35, 0.18)]
    elif group == "nhieu_hat":
        adaptive_cs = (8, 14, 20, 26)
        sauvola_ks = (0.20, 0.30, 0.40)
        min_std = 3.5 
        dt_params = [(0.28, 0.10), (0.38, 0.18)]
    elif group == "sang_khong_deu":
        adaptive_cs = (5, 9, 14)
        sauvola_ks = (0.15, 0.25, 0.35)
        min_std = 6.5
        dt_params = [(0.32, 0.15)]
    else:
        adaptive_cs = (6, 10, 14)
        sauvola_ks = (0.18, 0.28, 0.38)
        min_std = 7.0
        dt_params = [(0.35, 0.15)]

    results = []
    for prep_name, work in make_preprocessed(gray, group):
        results.append((f"{prep_name}_otsu", otsu(work)))
        for win in windows:
            # 1. Thêm ứng viên Phân ngưỡng kép (Dual Threshold)
            for ks, kl in dt_params:
                results.append((
                    f"{prep_name}_dualthresh_w{win}_ks{ks:.2f}_kl{kl:.2f}",
                    dual_threshold_binarize(work, window=win, k_strict=ks, k_loose=kl, min_std=min_std)
                ))
            # 2. Ứng viên Adaptive Gaussian
            for c in adaptive_cs:
                results.append((f"{prep_name}_adaptive_w{win}_c{c}",
                                adaptive_gaussian(work, win, c, min_std=min_std)))
            # 3. Ứng viên Sauvola
            for k in sauvola_ks:
                results.append((f"{prep_name}_sauvola_w{win}_k{k:.2f}",
                                sauvola(work, win, k, min_std=min_std)))
    return results, sw


def binary_metrics(binary, gray, sw):
    black = (binary == 0)
    dark_ratio = float(black.mean())

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        black.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=np.int32)
    total = binary.size
    tiny_area = max(2, int(0.35 * sw * sw))
    tiny_count = int(np.sum(areas <= tiny_area)) if areas.size else 0
    tiny_ratio = tiny_count / max(1, len(areas))
    largest_ratio = float(areas.max() / total) if areas.size else 0.0

    h, w = binary.shape
    bw = max(1, min(h, w) // 100)
    border = np.zeros_like(black)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    border_ratio = float(black[border].mean())

    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    strong = grad > max(float(np.percentile(grad, 94)), 2.5 * float(grad.mean()))

    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if np.any(edge) and np.any(strong):
        dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        edge_recall = float(np.mean(dist[strong] <= max(2.0, 0.6 * sw)))
    else:
        edge_recall = 0.0

    if np.any(black):
        dt = cv2.distanceTransform(black.astype(np.uint8), cv2.DIST_L2, 3)
        vals = dt[dt > 0]
        median_half_width = float(np.median(vals)) if vals.size else 0.0
        thickness_ratio = median_half_width / max(sw / 2.0, 0.5)
    else:
        thickness_ratio = 0.0

    return {
        "dark_ratio": dark_ratio,
        "components": int(len(areas)),
        "tiny_count": tiny_count,
        "tiny_ratio": float(tiny_ratio),
        "largest_ratio": largest_ratio,
        "border_black_ratio": border_ratio,
        "edge_recall": edge_recall,
        "thickness_ratio": thickness_ratio,
    }


def score_candidate(m, group):
    r = m["dark_ratio"]
    score = 100.0

    if r < 0.005:
        score -= 95
    elif r < 0.012:
        score -= 40
    elif r < 0.02:
        score -= 15

    max_dark_limit = 0.16 if group == "nhieu_hat" else 0.22
    if r > 0.40:
        score -= 85
    elif r > 0.28:
        score -= 60
    elif r > max_dark_limit:
        score -= 40

    if m["largest_ratio"] > 0.08:
        score -= 70
    elif m["largest_ratio"] > 0.04:
        score -= 25

    if m["border_black_ratio"] > 0.50:
        score -= 35
    elif m["border_black_ratio"] > 0.25:
        score -= 15

    if m["tiny_count"] > 180:
        score -= min(50.0, 15.0 + (m["tiny_count"] - 180) * 0.08)
    elif m["tiny_count"] > 80 and m["tiny_ratio"] > 0.85:
        score -= 15

    score -= 40.0 * (1.0 - min(m["edge_recall"] / 0.65, 1.0))

    tr = m["thickness_ratio"]
    if tr > 2.2:
        score -= 60
    elif tr > 1.7:
        score -= 40
    elif tr > 1.3:
        score -= 20

    return round(score, 2)


def candidate_is_duplicate(a, b):
    if a.shape != b.shape:
        return False
    small_a = cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, (128, 128), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(small_a.astype(np.float32) - small_b.astype(np.float32)))) < 0.003


def rank_candidates(gray, group):
    generated, sw = make_threshold_candidates(gray, group)
    ranked = []
    seen = []

    for name, binary in generated:
        metrics = binary_metrics(binary, gray, sw)
        score = score_candidate(metrics, group)
        if any(candidate_is_duplicate(binary, old) for old in seen):
            continue
        seen.append(binary)
        ranked.append({
            "name": name,
            "binary": binary,
            "metrics": metrics,
            "score": score,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked, sw, len(generated)


def make_status(best, gray):
    m = best["metrics"]
    warnings = []
    if m["dark_ratio"] < 0.005:
        warnings.append("CO_NGUY_CO_MAT_NET")
    if m["dark_ratio"] > 0.35:
        warnings.append("ANH_QUA_NHIEU_VUNG_DEN")
    if m["tiny_count"] > 200:
        warnings.append("NHIEU_HAT_NEN")
    if m["edge_recall"] < 0.25:
        warnings.append("CO_THE_MAT_NET_CHU")
    if m["border_black_ratio"] > 0.35:
        warnings.append("VIEN_DEN_CAN_KIEM_TRA")
    if best["score"] < 45:
        warnings.append("KHONG_CO_UNG_VIEN_TU_DONG_DANG_TIN")
    return ("REVIEW" if warnings else "OK"), warnings


def save_comparison(original, binary, path):
    h, w = original.shape
    title_h = max(30, int(h * 0.035))
    left = np.full((h + title_h, w), 255, np.uint8)
    right = np.full((h + title_h, w), 255, np.uint8)
    left[title_h:] = original
    right[title_h:] = binary
    scale = max(0.55, title_h / 55.0)
    cv2.putText(left, "BEFORE", (10, int(title_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1, cv2.LINE_AA)
    cv2.putText(right, "THRESHOLDING", (10, int(title_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1, cv2.LINE_AA)
    save_image(path, np.hstack([left, right]))


def safe_stem(relative_path):
    stem = os.path.splitext(relative_path)[0]
    return stem.replace("\\", "_").replace("/", "_")


def collect_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def process_all():
    if not os.path.isdir(INPUT_DIR):
        print(f"Khong tim thay thu muc dau vao: {INPUT_DIR}")
        return

    ensure_dirs()
    paths = collect_images(INPUT_DIR)
    if not paths:
        print("Khong tim thay anh dau vao.")
        return

    report_path = os.path.join(REPORT_DIR, "thresholding_report.csv")
    fields = [
        "file", "group", "rank", "selected", "status", "warnings",
        "candidate", "score", "dark_ratio", "components", "tiny_ratio",
        "largest_ratio", "border_black_ratio", "edge_recall",
        "thickness_ratio", "output"
    ]
    rows = []

    for idx, path in enumerate(paths, 1):
        rel = os.path.relpath(path, INPUT_DIR)
        group = get_group(rel)
        stem = safe_stem(rel)
        print(f"[{idx}/{len(paths)}] Dang xu ly: {rel}")

        try:
            gray = read_gray(path)
            ranked, sw, total = rank_candidates(gray, group)
            if not ranked:
                raise RuntimeError("Khong tao duoc ung vien thresholding.")

            best = ranked[0]
            status, warnings = make_status(best, gray)
            out_path = os.path.join(OUTPUT_DIR, stem + ".png")
            save_image(out_path, best["binary"])

            if SAVE_COMPARISONS:
                save_comparison(
                    gray, best["binary"],
                    os.path.join(REPORT_DIR, stem + "_comparison.png")
                )

            for rank, cand in enumerate(ranked[:SAVE_TOP], 1):
                alt_path = os.path.join(ALTERNATIVE_DIR, stem)
                save_image(f"{alt_path}_rank{rank}_binary.png", cand["binary"])
                if SAVE_COMPARISONS:
                    save_comparison(
                        gray, cand["binary"],
                        f"{alt_path}_rank{rank}_comparison.png"
                    )

                m = cand["metrics"]
                rows.append({
                    "file": rel,
                    "group": group,
                    "rank": rank,
                    "selected": int(rank == 1),
                    "status": status if rank == 1 else "ALTERNATIVE",
                    "warnings": ";".join(warnings) if rank == 1 else "",
                    "candidate": cand["name"],
                    "score": cand["score"],
                    "dark_ratio": round(m["dark_ratio"], 5),
                    "components": m["components"],
                    "tiny_ratio": round(m["tiny_ratio"], 4),
                    "largest_ratio": round(m["largest_ratio"], 5),
                    "border_black_ratio": round(m["border_black_ratio"], 4),
                    "edge_recall": round(m["edge_recall"], 4),
                    "thickness_ratio": round(m["thickness_ratio"], 3),
                    "output": out_path if rank == 1 else f"{alt_path}_rank{rank}_binary.png",
                })

            print(f"  -> {status}; score={best['score']}; candidate={best['name']}")
            if warnings:
                print("  -> Kiem tra: " + ", ".join(warnings))

        except Exception as exc:
            print(f"  LOI: {exc}")
            rows.append({
                "file": rel, "group": group, "rank": "", "selected": "",
                "status": "ERROR", "warnings": str(exc), "candidate": "",
                "score": "", "dark_ratio": "", "components": "", "tiny_ratio": "",
                "largest_ratio": "", "border_black_ratio": "", "edge_recall": "",
                "thickness_ratio": "", "output": ""
            })

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nHOAN TAT.")


if __name__ == "__main__":
    process_all()