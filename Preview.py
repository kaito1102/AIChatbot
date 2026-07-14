"""
Dò bảng bằng Table Transformer (deep learning) + OCR bằng EasyOCR
===================================================================

KHÁC BIỆT so với bản dùng OpenCV (ocr_pdf_pipeline_v2.py):
  - OpenCV: dò bảng dựa vào đường kẻ ngang/dọc -> dễ nhầm với đường kích thước,
    và bỏ sót bảng nếu đường kẻ mờ/đứt quãng.
  - Table Transformer: model AI học đặc trưng THỊ GIÁC của bảng (bố cục, mật độ
    chữ, khoảng cách...) từ hàng triệu bảng thật -> tổng quát hoá tốt hơn nhiều
    cho tài liệu có cấu trúc KHÔNG cố định, kể cả bảng không có đường viền.

  Đánh đổi: cần tải model (~115MB, 1 lần duy nhất) từ huggingface.co, và chạy
  chậm hơn OpenCV (vài giây/trang thay vì mili-giây).

Cài đặt:
    pip install transformers torch pillow timm pymupdf easyocr

Lưu ý về mạng công ty:
    Model tải từ huggingface.co (khác domain với GitHub bạn từng bị chặn trước
    đây). Nếu IT chặn cả huggingface.co, cách này sẽ không chạy được — khi đó
    quay lại dùng ocr_pdf_pipeline_v2.py (chế độ "auto") và tự kiểm tra bằng
    debug_page_XXX.png, chỉnh tham số MIN_TABLE_AREA_RATIO/LINE_KERNEL_SCALE
    theo từng loại file.

Cách dùng:
    python ocr_pdf_pipeline_v3_tabletransformer.py input.pdf
    python ocr_pdf_pipeline_v3_tabletransformer.py input_folder/
"""

import sys
import os
import csv
import fitz  # PyMuPDF
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import easyocr

DPI = 300
LANGUAGES = ['en']          # đổi thành ['vi','en'] hoặc ['ja'] tuỳ tài liệu
OUTPUT_DIR = "output"
MIN_CONFIDENCE = 0.3        # ngưỡng tin cậy OCR (easyocr)
TABLE_DETECT_THRESHOLD = 0.7  # ngưỡng tin cậy phát hiện bảng (Table Transformer) — tăng nếu bị lẫn vùng không phải bảng
PADDING = 10
UPSCALE_FACTOR = 2


def load_table_model():
    print("Tải model Table Transformer (lần đầu sẽ mất vài phút để tải ~115MB)...")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    model.eval()
    return processor, model


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


def detect_tables_dl(image_path, processor, model, threshold=TABLE_DETECT_THRESHOLD):
    """Dò bảng bằng Table Transformer. Trả về danh sách box (x, y, w, h)."""
    image = PILImage.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes = []
    for score, box in zip(results["scores"], results["boxes"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        boxes.append((x1, y1, x2 - x1, y2 - y1, float(score)))
    return boxes


def crop_and_upscale(img_pil, box, padding=PADDING, scale=UPSCALE_FACTOR):
    x, y, w, h = box
    iw, ih = img_pil.size
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(iw, x + w + padding), min(ih, y + h + padding)
    crop = img_pil.crop((x0, y0, x1, y1))
    if scale != 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), PILImage.LANCZOS)
    return crop


def ocr_pil_image(reader, pil_img, min_confidence=MIN_CONFIDENCE):
    import numpy as np
    arr = np.array(pil_img)
    results = reader.readtext(arr, detail=1)
    return [(text, conf) for (bbox, text, conf) in results if conf >= min_confidence]


def process_pdf(pdf_path, processor, model, reader, summary_rows):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(page_dir, exist_ok=True)

    print(f"[1/3] Render PDF -> ảnh: {pdf_path}")
    image_paths = render_pdf_to_images(pdf_path, page_dir)

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[2/3] Dò bảng (Table Transformer) trang {idx}")
        boxes = detect_tables_dl(img_path, processor, model)
        print(f"       -> tìm thấy {len(boxes)} vùng bảng")

        img_pil = PILImage.open(img_path).convert("RGB")

        # ảnh debug
        from PIL import ImageDraw
        debug_img = img_pil.copy()
        draw = ImageDraw.Draw(debug_img)
        for (x, y, w, h, score) in boxes:
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)
            draw.text((x, max(0, y - 20)), f"{score:.2f}", fill=(255, 0, 0))
        debug_img.save(os.path.join(page_dir, f"debug_page_{idx:03d}.png"))

        for t_idx, (x, y, w, h, score) in enumerate(boxes, start=1):
            crop = crop_and_upscale(img_pil, (x, y, w, h))
            crop_path = os.path.join(page_dir, f"page_{idx:03d}_table_{t_idx:02d}.png")
            crop.save(crop_path)

            print(f"[3/3] OCR bảng {t_idx}/{len(boxes)} (trang {idx}, độ tin cậy dò bảng={score:.2f})")
            lines = ocr_pil_image(reader, crop)
            table_text = "\n".join(text for text, conf in lines)
            with open(crop_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                f.write(table_text)

            for text, conf in lines:
                summary_rows.append({
                    "file": base_name, "page": idx, "region": f"table_{t_idx}",
                    "detect_score": round(score, 3), "text": text, "confidence": round(conf, 3),
                })


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python ocr_pdf_pipeline_v3_tabletransformer.py <file.pdf hoặc thư_mục>")
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

    processor, model = load_table_model()

    print(f"Khởi tạo EasyOCR reader (ngôn ngữ: {LANGUAGES})...")
    reader = easyocr.Reader(LANGUAGES)

    summary_rows = []
    for pdf_path in pdf_files:
        process_pdf(pdf_path, processor, model, reader, summary_rows)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "page", "region", "detect_score", "text", "confidence"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nHoàn tất. Xem: {summary_path}")
    print("QUAN TRỌNG: kiểm tra debug_page_XXX.png trước — nếu TABLE_DETECT_THRESHOLD")
    print("quá thấp sẽ lẫn vùng không phải bảng; quá cao sẽ bỏ sót bảng thật.")


if __name__ == "__main__":
    main()
