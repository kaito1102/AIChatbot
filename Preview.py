"""
Pipeline OCR cho PDF scan CÓ TÁCH BẢNG (table) trước khi OCR
=============================================================

Ý tưởng: bảng trong bản vẽ/tài liệu scan thường có đường kẻ ngang/dọc rõ.
Script này dò các đường kẻ đó bằng OpenCV để tìm vùng bảng, cắt riêng
từng vùng bảng ra thành ảnh nhỏ, rồi mới chạy OCR trên từng ảnh đã cắt.
OCR trên ảnh đã cắt + phóng to sẽ chính xác hơn nhiều so với OCR nguyên
cả trang (nhiễu nét vẽ kỹ thuật, chữ nhỏ, mật độ cao).

Cài đặt (chỉ cần pip):
    pip install pymupdf easyocr opencv-python-headless numpy

Cách dùng:
    python ocr_pdf_pipeline_v2.py input.pdf
    python ocr_pdf_pipeline_v2.py input_folder/

Kết quả trong output/<ten_file>/:
    page_001.png              - ảnh cả trang
    page_001_table_01.png     - từng vùng bảng đã cắt
    page_001_table_01.txt     - text OCR của riêng vùng bảng đó
    page_001_nontable.txt     - text OCR phần còn lại của trang (không phải bảng)
    debug_page_001.png        - ảnh debug: khung đỏ = vùng bảng đã phát hiện
output/summary.csv            - bảng tổng hợp toàn bộ, có cột "region" (table/nontable)
"""

import sys
import os
import csv
import cv2
import numpy as np
import fitz  # PyMuPDF
import easyocr

# ------------------------------------------------------------------
# CẤU HÌNH
# ------------------------------------------------------------------
DPI = 300
LANGUAGES = ['en']         # đổi thành ['vi','en'] hoặc ['ja'] tuỳ tài liệu
OUTPUT_DIR = "output"
MIN_CONFIDENCE = 0.3

# Tham số dò bảng — chỉnh nếu bảng bị bỏ sót hoặc phát hiện thừa
MIN_TABLE_AREA_RATIO = 0.01   # vùng bảng phải chiếm ít nhất 1% diện tích trang
MAX_TABLE_AREA_RATIO = 0.4    # loại bỏ vùng quá lớn (thường là khung viền ngoài của cả trang, không phải bảng)
LINE_KERNEL_SCALE = 30        # càng lớn -> chỉ bắt đường kẻ càng dài (giảm nhiễu)
PADDING = 6                   # padding quanh vùng bảng khi cắt (px)
UPSCALE_FACTOR = 2            # phóng to ảnh bảng trước khi OCR để tăng độ chính xác


def render_pdf_to_images(pdf_path, out_dir, dpi=DPI):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        img_path = os.path.join(out_dir, f"page_{i+1:03d}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def detect_table_regions(image_path):
    """
    Dò các vùng bảng trong ảnh dựa trên đường kẻ ngang/dọc.
    Trả về danh sách bounding box (x, y, w, h), đã lọc theo diện tích tối thiểu.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nhị phân hoá đảo màu: nét vẽ/chữ = trắng, nền = đen
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h, w = gray.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // LINE_KERNEL_SCALE, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // LINE_KERNEL_SCALE))

    horiz_lines = cv2.erode(binary, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=1)

    vert_lines = cv2.erode(binary, vert_kernel, iterations=1)
    vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=1)

    # Kết hợp đường ngang + dọc -> khung bảng
    table_mask = cv2.add(horiz_lines, vert_lines)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # RETR_TREE để lấy cả các khung lồng bên trong (bảng thật thường nằm trong khung viền
    # ngoài cùng của cả trang, nên phải nhìn cả contour con, không chỉ contour ngoài cùng)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_TABLE_AREA_RATIO * w * h
    max_area = MAX_TABLE_AREA_RATIO * w * h
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # loại vùng quá nhỏ (nhiễu) và quá lớn (khung viền ngoài cả trang, không phải bảng)
        if min_area <= area <= max_area:
            boxes.append((x, y, cw, ch))

    # Gộp các box chồng lấn/lồng nhau (tránh phát hiện trùng)
    boxes = merge_overlapping_boxes(boxes)
    # Sắp theo thứ tự đọc: trên xuống dưới, trái sang phải
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, img


def merge_overlapping_boxes(boxes):
    if not boxes:
        return []
    rects = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, x2, y2 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                a1, b1, a2, b2 = rects[j]
                # overlap check
                if x1 < a2 and a1 < x2 and y1 < b2 and b1 < y2:
                    x1, y1, x2, y2 = min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)
                    used[j] = True
                    merged = True
            result.append([x1, y1, x2, y2])
            used[i] = True
        rects = result
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in rects]


def crop_and_upscale(img, box, padding=PADDING, scale=UPSCALE_FACTOR):
    x, y, w, h = box
    ih, iw = img.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(iw, x + w + padding), min(ih, y + h + padding)
    crop = img[y0:y1, x0:x1]
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def mask_out_boxes(img, boxes):
    """Trả về bản sao ảnh với các vùng bảng đã bị che trắng (để OCR riêng phần còn lại)."""
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return out


def ocr_array(reader, image_array, min_confidence=MIN_CONFIDENCE):
    results = reader.readtext(image_array, detail=1)
    return [(text, conf) for (bbox, text, conf) in results if conf >= min_confidence]


def process_pdf(pdf_path, reader, summary_rows):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(page_dir, exist_ok=True)

    print(f"[1/4] Render PDF -> ảnh: {pdf_path}")
    image_paths = render_pdf_to_images(pdf_path, page_dir)

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[2/4] Dò vùng bảng trang {idx}: {img_path}")
        boxes, img = detect_table_regions(img_path)
        print(f"       -> tìm thấy {len(boxes)} vùng bảng")

        # Ảnh debug: vẽ khung đỏ quanh vùng bảng phát hiện được
        debug_img = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(page_dir, f"debug_page_{idx:03d}.png"), debug_img)

        # OCR riêng từng vùng bảng
        for t_idx, box in enumerate(boxes, start=1):
            crop = crop_and_upscale(img, box)
            crop_path = os.path.join(page_dir, f"page_{idx:03d}_table_{t_idx:02d}.png")
            cv2.imwrite(crop_path, crop)

            print(f"[3/4] OCR bảng {t_idx}/{len(boxes)} (trang {idx})")
            lines = ocr_array(reader, crop)
            table_text = "\n".join(text for text, conf in lines)
            with open(crop_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                f.write(table_text)

            for text, conf in lines:
                summary_rows.append({
                    "file": base_name, "page": idx, "region": f"table_{t_idx}",
                    "text": text, "confidence": round(conf, 3),
                })

        # OCR phần còn lại của trang (đã che các vùng bảng để tránh trùng lặp)
        remainder = mask_out_boxes(img, boxes)
        print(f"[4/4] OCR phần ngoài bảng (trang {idx})")
        lines = ocr_array(reader, remainder)
        nontable_text = "\n".join(text for text, conf in lines)
        with open(os.path.join(page_dir, f"page_{idx:03d}_nontable.txt"), "w", encoding="utf-8") as f:
            f.write(nontable_text)

        for text, conf in lines:
            summary_rows.append({
                "file": base_name, "page": idx, "region": "nontable",
                "text": text, "confidence": round(conf, 3),
            })


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python ocr_pdf_pipeline_v2.py <file.pdf hoặc thư_mục>")
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
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "page", "region", "text", "confidence"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nHoàn tất. Xem: {summary_path}")
    print("Kiểm tra file debug_page_XXX.png để xác nhận vùng bảng có được dò đúng không.")


if __name__ == "__main__":
    main()
