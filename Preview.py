"""
Trích xuất 1 bảng lớn (MTRL INDIC...SURF TRT) + 2 bảng nhỏ (DATE/REV/CLS,
PARTS NO/PARTS NAME) — CHỊU ĐƯỢC xê dịch vị trí giữa các file.
========================================================================

Ý tưởng "neo động" (anchor):
  Toạ độ tuyệt đối cố định sẽ sai nếu file khác bị lệch vị trí (do scan/
  margin khác nhau). Nhưng NHÃN TIÊU ĐỀ (chữ "MTRL INDIC.") luôn giống hệt
  nhau ở mọi file cùng khuôn mẫu. Nên:
    1. OCR tìm vị trí text "MTRL INDIC" trên trang -> đây là điểm neo.
    2. Tính độ lệch (dx, dy) so với vị trí neo trong file mẫu đã đo sẵn.
    3. Dịch cả 3 vùng cắt (đã đo sẵn dạng offset so với điểm neo) theo đúng
       độ lệch đó -> luôn đúng vị trí dù file bị xê dịch bao nhiêu.

Cài đặt:
    pip install pymupdf easyocr opencv-python-headless numpy

Cách dùng:
    python ocr_anchor_regions.py input.pdf
    python ocr_anchor_regions.py input_folder/
"""

import sys
import os
import csv
import re
import fitz
import cv2
import numpy as np
import easyocr

DPI = 300
LANGUAGES = ['en']
OUTPUT_DIR = "output"
UPSCALE_FACTOR = 3

# Text neo (anchor) — dùng để định vị. Có thể sai khác nhẹ do OCR (vd "MTRLINDIC"),
# nên so khớp bằng cách loại khoảng trắng/dấu câu rồi so "chứa chuỗi con".
ANCHOR_TEXT = "MTRLINDIC"

# Vùng tìm neo: OCR toàn bộ nửa dưới trang (neo luôn nằm ở title-block, không
# cần OCR cả trang -> nhanh hơn). Chỉnh nếu title-block nằm ở vị trí khác.
ANCHOR_SEARCH_REGION = (0.0, 0.55, 1.0, 0.45)  # (x, y, w, h) tỉ lệ trang

# ------------------------------------------------------------------
# HIỆU CHỈNH TỪ FILE MẪU (QC1-6046-000D201001.pdf, DPI=300, trang 2481x3508)
# ------------------------------------------------------------------
# Vị trí neo THẬT trong file mẫu (góc trên-trái của text "MTRL INDIC.")
REF_ANCHOR_XY = (678, 2592)

# 3 vùng cần cắt, lưu dưới dạng OFFSET (px, tại DPI=300) SO VỚI ĐIỂM NEO,
# không phải toạ độ tuyệt đối -> khi neo dịch chuyển, vùng cắt dịch theo.
REF_REGIONS = {
    "BIG_TABLE":         {"dx": 0,   "dy": 0,   "w": 839,  "h": 284},
    "DATE_ROW":          {"dx": 390, "dy": 461, "w": 1026, "h": 116},
    "PARTSNO_NAME_ROW":  {"dx": 390, "dy": 577, "w": 1026, "h": 236},
}
# (dx,dy,w,h tính từ: BIG_TABLE=(678,2592,1517,2876); DATE_ROW=(1068,3053,2094,3169);
#  PARTSNO_NAME_ROW=(1068,3169,2094,3405); trừ đi REF_ANCHOR_XY=(678,2592))


def render_first_page(pdf_path, dpi=DPI):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = doc[0].get_pixmap(matrix=matrix)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    return img


def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def find_anchor(reader, img):
    """
    OCR vùng ANCHOR_SEARCH_REGION để tìm text neo "MTRL INDIC".
    Trả về (x, y) góc trên-trái của text neo trong ảnh GỐC (đã cộng lại offset
    vùng tìm kiếm), hoặc None nếu không tìm thấy.
    """
    h, w = img.shape[:2]
    rx, ry, rw, rh = ANCHOR_SEARCH_REGION
    x0, y0 = int(rx * w), int(ry * h)
    x1, y1 = int((rx + rw) * w), int((ry + rh) * h)
    search_crop = img[y0:y1, x0:x1]

    results = reader.readtext(search_crop, detail=1)
    best = None
    for (bbox, text, conf) in results:
        norm = normalize(text)
        if ANCHOR_TEXT in norm or norm in ANCHOR_TEXT:
            # bbox = 4 điểm góc; lấy điểm trên-trái
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            candidate = (x0 + min(xs), y0 + min(ys))
            if best is None or conf > best[1]:
                best = (candidate, conf)
    return best[0] if best else None


def crop_region(img, x, y, w, h, upscale=UPSCALE_FACTOR):
    ih, iw = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(iw, int(x + w)), min(ih, int(y + h))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if upscale != 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return crop


def ocr_crop(reader, crop_img):
    if crop_img is None:
        return ""
    results = reader.readtext(crop_img, detail=0, paragraph=True)
    return "\n".join(results).strip()


def process_pdf(pdf_path, reader, summary_rows):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(page_dir, exist_ok=True)

    print(f"Xử lý: {pdf_path}")
    img = render_first_page(pdf_path)

    anchor = find_anchor(reader, img)
    row = {"file": base_name}

    if anchor is None:
        print("  !! KHÔNG tìm thấy điểm neo 'MTRL INDIC' -> bỏ qua file này "
              "(kiểm tra ANCHOR_SEARCH_REGION hoặc chất lượng scan).")
        row["anchor_found"] = "NO"
        for name in REF_REGIONS:
            row[name] = ""
        summary_rows.append(row)
        return

    ax, ay = anchor
    dx_offset, dy_offset = ax - REF_ANCHOR_XY[0], ay - REF_ANCHOR_XY[1]
    print(f"  Neo tìm thấy tại ({ax},{ay}) -> lệch so với mẫu: ({dx_offset:+d},{dy_offset:+d})")
    row["anchor_found"] = "YES"
    row["offset_x"] = dx_offset
    row["offset_y"] = dy_offset

    for name, r in REF_REGIONS.items():
        x = ax + r["dx"]
        y = ay + r["dy"]
        crop = crop_region(img, x, y, r["w"], r["h"])
        if crop is not None:
            cv2.imwrite(os.path.join(page_dir, f"{name}.png"), crop)
        text = ocr_crop(reader, crop)
        row[name] = text
        print(f"  {name}: {text[:80]!r}{'...' if len(text) > 80 else ''}")

    summary_rows.append(row)


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python ocr_anchor_regions.py <file.pdf hoặc thư_mục>")
        sys.exit(1)

    input_path = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isdir(input_path):
        pdf_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                     if f.lower().endswith(".pdf")]
    else:
        pdf_files = [input_path]

    if not pdf_files:
        print("Không tìm thấy file PDF nào.")
        sys.exit(1)

    print(f"Khởi tạo EasyOCR reader (ngôn ngữ: {LANGUAGES})...")
    reader = easyocr.Reader(LANGUAGES)

    summary_rows = []
    for pdf_path in pdf_files:
        process_pdf(pdf_path, reader, summary_rows)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    fieldnames = ["file", "anchor_found", "offset_x", "offset_y"] + list(REF_REGIONS.keys())
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nHoàn tất. Xem: {summary_path}")
    print("File nào có anchor_found=NO nghĩa là không định vị được -> kiểm tra tay riêng.")


if __name__ == "__main__":
    main()
