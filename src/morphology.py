from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Đường dẫn cố định
DEFAULT_INPUT_DIR = Path("data/thresholding_experiments/all_processed")
DEFAULT_OUTPUT_DIR = Path("data/processed")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".tif", ".tiff", ".webp"}
NOISE_TYPES = ("dom_ban", "dut_net", "nhieu_hat", "sang_khong_deu")


def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {input_dir}")
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong: {input_dir}")
    return paths


def read_binary(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Lỗi ghi ảnh: {path}")
    with open(path, "wb") as f:
        f.write(encoded.tobytes())


def get_noise_type(filename: str) -> str:
    filename_lower = filename.lower()
    for n in NOISE_TYPES:
        if n in filename_lower:
            return n
    return "default"


# --- CÁC HÀM MORPHOLOGY ĐÃ TINH CHỈNH TỐI ƯU ---

def remove_small_speckles(binary: np.ndarray, min_area: int = 4) -> np.ndarray:
    """Xóa các đốm nhiễu li ti, bảo vệ dấu câu và dấu thanh tiếng Việt."""
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    
    clean_inv = np.zeros_like(inv)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_inv[labels == i] = 255
            
    return cv2.bitwise_not(clean_inv)


def remove_border_artifacts(binary: np.ndarray, border_dist: int = 3, min_area_ratio: float = 0.002) -> np.ndarray:
    """Dọn sạch dải đen ngón tay, mép bàn quét bám sát 4 cạnh ảnh."""
    inv = cv2.bitwise_not(binary)
    h, w = inv.shape
    min_area = int(h * w * min_area_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    
    clean_inv = inv.copy()
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        touches_edge = (x <= border_dist or y <= border_dist or 
                        (x + bw) >= w - border_dist or (y + bh) >= h - border_dist)
        
        # Nếu mảng bám sát mép và có diện tích lớn hoặc dài bất thường -> xóa sạch
        if touches_edge and (area >= min_area or bw > w * 0.25 or bh > h * 0.25):
            clean_inv[labels == i] = 0
            
    return cv2.bitwise_not(clean_inv)

def clear_outer_frame(binary: np.ndarray, margin: int = 4) -> np.ndarray:
    """Xóa trắng an toàn dải viền 4px sát mép ngoài để triệt tiêu vệt sợi bám biên."""
    cleaned = binary.copy()
    cleaned[:margin, :] = 255
    cleaned[-margin:, :] = 255
    cleaned[:, :margin] = 255
    cleaned[:, -margin:] = 255
    return cleaned

def connect_broken_strokes(binary: np.ndarray) -> np.ndarray:
    """Nối nét đứt theo chiều ngang tự nhiên của dòng chữ, không dính dòng."""
    inv = cv2.bitwise_not(binary)
    
    # 1. Nối nét ngang (3x1): bắc cầu các khoảng đứt giữa các nét bút cùng dòng
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, h_kernel)
    
    # 2. Nối nhẹ theo phương chéo/dọc bằng kernel chữ thập 3x3
    cross_kernel = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cross_kernel)
    return cv2.bitwise_not(inv)


def slim_bloated_strokes(binary: np.ndarray) -> np.ndarray:
    """Gọt mỏng viền đối xứng bằng kernel chữ thập để mở rỗng ruột chữ (e, o, a)."""
    inv = cv2.bitwise_not(binary)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    eroded = cv2.erode(inv, kernel, iterations=1)
    return cv2.bitwise_not(eroded)


def process_morphology(binary: np.ndarray, noise_type: str) -> np.ndarray:
    # 1. Dọn sạch dị vật mép ngoài cùng
    img = remove_border_artifacts(binary)
    img = clear_outer_frame(img, margin=4)

    # 2. Điều phối thông minh theo từng loại nhiễu
    if noise_type == "dut_net":
        # Nối nét trước bằng kernel ngang 3x1
        img = connect_broken_strokes(img)
        # Giữ min_area=3 để bảo toàn dấu chấm số tiền và dấu tiếng Việt
        img = remove_small_speckles(img, min_area=3)

    elif noise_type == "dom_ban":
        # Nền tài liệu cổ: lọc đốm vừa phải để không làm mất nét chữ mực phai
        img = remove_small_speckles(img, min_area=5)

    elif noise_type == "nhieu_hat":
        # Tự động thích ứng: chỉ gọt mỏng khi chữ bị phình quá đậm
        dark_ratio = float(np.mean(img == 0))
        if dark_ratio > 0.08:
            img = slim_bloated_strokes(img)
        else:
            # Nếu chữ đã mỏng/vỡ do phân ngưỡng kép, nối nhẹ lại nét
            img = connect_broken_strokes(img)
        img = remove_small_speckles(img, min_area=4)

    else:
        img = remove_small_speckles(img, min_area=4)

    return img


def run(input_dir: Path, output_dir: Path) -> int:
    input_p = Path(input_dir)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    images = list_images(input_p)
    total = len(images)
    print(f"Đang xử lý Morphology: {total} ảnh từ [{input_p}] -> [{output_p}]")

    for idx, path in enumerate(images, 1):
        noise = get_noise_type(path.name)
        binary = read_binary(path)
        result = process_morphology(binary, noise)
        
        out_file = output_p / f"{path.stem}.png"
        write_image(out_file, result)
        print(f"[{idx}/{total}] Đã xử lý: {path.name} ({noise})")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Chương trình Morphology xử lý kết quả cuối cùng.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Thư mục ảnh đầu vào")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục lưu ảnh kết quả")
    args = parser.parse_args()

    run(args.input, args.output)
    print(f"\nHoàn tất. Toàn bộ ảnh đã được lưu vào thư mục: {args.output}")


if __name__ == "__main__":
    main()