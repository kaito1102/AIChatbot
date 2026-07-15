"""
Trích xuất 1 bảng lớn (MTRL INDIC...SURF TRT) + 2 bảng nhỏ (DATE/REV/CLS,
PARTS NO/PARTS NAME) — dùng PaddleOCR thay EasyOCR (đọc chữ kỹ thuật/lẫn
tiếng Nhật-Anh chính xác hơn).
========================================================================

Cùng cơ chế "neo động" (anchor) như bản EasyOCR:
    1. OCR tìm vị trí text "MTRL INDIC" trên trang -> điểm neo.
    2. Tính độ lệch (dx, dy) so với vị trí neo trong file mẫu.
    3. Dịch cả 3 vùng cắt theo đúng độ lệch đó -> đúng vị trí dù file bị xê dịch.

Cài đặt:
    pip install paddlepaddle paddleocr pymupdf opencv-python-headless numpy

LƯU Ý VỀ PHIÊN BẢN: script này viết cho PaddleOCR >= 3.0 (API `.predict()`,
kết quả trả về dạng dict với khoá 'rec_texts'/'rec_boxes'/'rec_scores').
Nếu `pip show paddleocr` ra bản 2.x, API cũ dùng `.ocr(img, cls=True)` và
cấu trúc kết quả khác hẳn — báo lại để mình viết bản tương thích 2.x.

Cách dùng:
    python ocr_anchor_regions_paddle.py input.pdf
    python ocr_anchor_regions_paddle.py input_folder/
"""

import sys
import os
import csv
import re
import fitz
import cv2
import numpy as np
from paddleocr import PaddleOCR

DPI = 300
LANG = 'en'   # PaddleOCR: 'en'=Anh, 'ch'=Trung+Anh, 'japan'=Nhật, 'korean'=Hàn...
OUTPUT_DIR = "output"
UPSCALE_FACTOR = 3
MIN_CONFIDENCE = 0.5

ANCHOR_TEXT = "MTRLINDIC"
ANCHOR_SEARCH_REGION = (0.0, 0.55, 1.0, 0.45)  # (x, y, w, h) tỉ lệ trang

REF_ANCHOR_XY = (678, 2592)
REF_REGIONS = {
    "BIG_TABLE":         {"dx": 0,   "dy": 0,   "w": 839,  "h": 284},
    "DATE_ROW":          {"dx": 390, "dy": 461, "w": 1026, "h": 116},
    "PARTSNO_NAME_ROW":  {"dx": 390, "dy": 577, "w": 1026, "h": 236},
}


def get_ocr_engine():
    print(f"Khởi tạo PaddleOCR (ngôn ngữ: {LANG})... lần đầu sẽ tải model.")
    # enable_mkldnn=False: bắt buộc trên nhiều máy CPU Windows vì PaddlePaddle 3.x có
    # lỗi đã biết (NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support)
    # khi bật oneDNN với "new executor" PIR — xem PaddlePaddle/PaddleOCR issue #17955, #18162.
    try:
        return PaddleOCR(
            lang=LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )
    except TypeError:
        # bản PaddleOCR/paddlex cũ hơn có thể không nhận kwarg này qua path khởi tạo
        # -> thử lại không có enable_mkldnn, và set qua biến môi trường trước khi import paddle
        print("  (enable_mkldnn không được nhận trực tiếp, thử lại không có tham số này)")
        return PaddleOCR(
            lang=LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )


def run_ocr(engine, img_bgr):
    """
    Chạy PaddleOCR trên 1 ảnh (numpy BGR). Trả về list (text, score, box)
    với box = [x1, y1, x2, y2] (toạ độ TRONG ảnh truyền vào).
    """
    results = engine.predict(img_bgr)
    if not results:
        return []
    res = results[0]
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    boxes = res.get("rec_boxes", [])
    out = []
    for text, score, box in zip(texts, scores, boxes):
        if score >= MIN_CONFIDENCE:
            x1, y1, x2, y2 = [float(v) for v in box]
            out.append((text, float(score), (x1, y1, x2, y2)))
    return out


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


def find_anchor(engine, img):
    """Tìm text neo 'MTRL INDIC' trong vùng ANCHOR_SEARCH_REGION. Trả về (x,y) góc trên-trái, hoặc None."""
    h, w = img.shape[:2]
    rx, ry, rw, rh = ANCHOR_SEARCH_REGION
    x0, y0 = int(rx * w), int(ry * h)
    x1, y1 = int((rx + rw) * w), int((ry + rh) * h)
    search_crop = img[y0:y1, x0:x1]

    detections = run_ocr(engine, search_crop)
    best = None
    for text, score, (bx1, by1, bx2, by2) in detections:
        norm = normalize(text)
        if ANCHOR_TEXT in norm or norm in ANCHOR_TEXT:
            candidate = (x0 + bx1, y0 + by1)
            if best is None or score > best[1]:
                best = (candidate, score)
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


def ocr_crop_to_text(engine, crop_img):
    if crop_img is None:
        return ""
    detections = run_ocr(engine, crop_img)
    # sắp theo thứ tự đọc: trên->dưới, trái->phải (dùng tâm Y để gom theo hàng)
    detections.sort(key=lambda d: (round(d[2][1] / 20), d[2][0]))
    return "\n".join(text for text, score, box in detections).strip()


def process_pdf(pdf_path, engine, summary_rows):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(page_dir, exist_ok=True)

    print(f"Xử lý: {pdf_path}")
    img = render_first_page(pdf_path)

    anchor = find_anchor(engine, img)
    row = {"file": base_name}

    if anchor is None:
        print("  !! KHÔNG tìm thấy điểm neo 'MTRL INDIC' -> bỏ qua file này.")
        row["anchor_found"] = "NO"
        for name in REF_REGIONS:
            row[name] = ""
        summary_rows.append(row)
        return

    ax, ay = anchor
    dx_offset, dy_offset = ax - REF_ANCHOR_XY[0], ay - REF_ANCHOR_XY[1]
    print(f"  Neo tìm thấy tại ({ax:.0f},{ay:.0f}) -> lệch so với mẫu: ({dx_offset:+.0f},{dy_offset:+.0f})")
    row["anchor_found"] = "YES"
    row["offset_x"] = round(dx_offset)
    row["offset_y"] = round(dy_offset)

    for name, r in REF_REGIONS.items():
        x = ax + r["dx"]
        y = ay + r["dy"]
        crop = crop_region(img, x, y, r["w"], r["h"])
        if crop is not None:
            cv2.imwrite(os.path.join(page_dir, f"{name}.png"), crop)
        text = ocr_crop_to_text(engine, crop)
        row[name] = text
        print(f"  {name}: {text[:80]!r}{'...' if len(text) > 80 else ''}")

    summary_rows.append(row)


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python ocr_anchor_regions_paddle.py <file.pdf hoặc thư_mục>")
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

    engine = get_ocr_engine()

    summary_rows = []
    for pdf_path in pdf_files:
        process_pdf(pdf_path, engine, summary_rows)

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
