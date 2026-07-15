"""
Pipeline OCR cho PDF scan - v3: chọn bảng theo NEO TỪ KHOÁ (anchor) thay vì
chỉ dựa đường kẻ hình học.
=============================================================================
TẠI SAO ĐỔI CÁCH TIẾP CẬN (đã kiểm chứng trên file mẫu QC1-6046-000D201001.pdf):

Trong mẫu bản vẽ Canon này, 2 vấn đề khiến "auto" hình học thuần tuý không
thể tách đúng 3/6 bảng mong muốn dù chỉnh tham số thế nào:

1. Bảng "COMPRESSION SPG" và "COIL END SHAPE" được vẽ chung 1 khung viền
   NGOÀI DUY NHẤT trong file gốc (không phải do merge code sai) -> OpenCV
   luôn thấy đây là 1 contour. Phải cắt bằng đường kẻ dọc nội bộ.
2. Thử rule "đường kẻ dọc dài gần 100% chiều cao box = ranh giới 2 bảng
   khác nhau": đúng cho case (1) (đường chia tại frac=1.0, cột nội bộ của
   bảng SPG chỉ đạt frac~0.77) NHƯNG SAI cho khối "MTRL INDIC" + "REVISION
   NOTE": cột nhãn/giá trị NỘI BỘ của MTRL INDIC cũng đạt frac=1.0 y hệt
   ranh giới thật -> không có ngưỡng hình học nào tách đúng cả 2 case.

=> Kết luận: đây là vấn đề NGỮ NGHĨA (nội dung chữ), không giải quyết được
   chỉ bằng hình học. Giải pháp: dùng OCR để neo (anchor) từng bảng theo
   từ khoá tiêu đề của nó, rồi mới dùng đường kẻ để xác định biên chính xác.
   Cách này cũng chính là thứ bạn cần khi vị trí bảng xê dịch giữa các file
   khác nhau (từ khoá không đổi vị trí bảng có thể đổi).

Cài đặt (giống bản cũ, chỉ cần pip):
    pip install pymupdf easyocr opencv-python-headless numpy

Cách dùng:
    python ocr_pdf_pipeline_v3.py input.pdf
    python ocr_pdf_pipeline_v3.py input_folder/

Kết quả trong output/<ten_file>/:
    page_001.png                  - ảnh cả trang
    page_001_<table_name>.png     - từng bảng đã cắt (đặt tên theo TABLE_KEYWORDS)
    page_001_<table_name>.txt     - text OCR của riêng bảng đó
    page_001_nontable.txt         - text OCR phần còn lại
    debug_page_001.png            - khung đỏ = bảng đã chọn, khung xanh = box ứng viên bị loại
output/summary.csv                - bảng tổng hợp toàn bộ
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
LANGUAGES = ['en']
OUTPUT_DIR = "output"
MIN_CONFIDENCE = 0.3

MIN_TABLE_AREA_RATIO = 0.005
MAX_TABLE_AREA_RATIO = 0.4
LINE_KERNEL_SCALE = 40
MIN_GRID_LINES = 2
PADDING = 6
UPSCALE_FACTOR = 2
MERGE_GAP_RATIO = 0.02

# ------------------------------------------------------------------
# ĐỊNH NGHĨA 3 BẢNG CẦN LẤY — chỉnh từ khoá nếu đổi mẫu bản vẽ.
# "include": box ứng viên PHẢI chứa ít nhất 1 từ khớp 1 trong các keyword này.
# "exclude": nếu box ứng viên khớp include NHƯNG cũng chứa điểm khớp exclude
#            (trường hợp bị gộp nhầm với bảng lân cận), sẽ cắt bớt theo cạnh
#            gần điểm exclude nhất thay vì loại bỏ cả box.
# Khớp theo kiểu "substring, không phân biệt hoa/thường" trên text OCR.
# ------------------------------------------------------------------
TABLE_KEYWORDS = {
    "spec_table": {
        "include": ["compression", "spg", "wire dia", "active coils"],
        "exclude": ["coil end", "end shape", "clsd end", "open end"],
    },
    "material_block": {
        "include": ["mtrl indic", "manufctrer", "thick", "heat trt"],
        "exclude": [],
    },
    "revision_block": {
        "include": ["revision note", "parts no", "parts", "designed by"],
        "exclude": [],
    },
}


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


def count_grid_lines(line_mask, box, axis, min_run_ratio=0.5):
    x, y, w, h = box
    crop = line_mask[y:y + h, x:x + w]
    if crop.size == 0:
        return 0
    if axis == 'h':
        line_present = (crop.sum(axis=1) > (w * min_run_ratio * 255))
    else:
        line_present = (crop.sum(axis=0) > (h * min_run_ratio * 255))
    changes = np.diff(line_present.astype(int))
    return int((changes == 1).sum()) + (1 if len(line_present) and line_present[0] else 0)


def _boxes_should_merge(r1, r2, leaf_area1, leaf_area2, gap, align_ratio=0.5, min_fill_ratio=0.6):
    x1, y1, x2, y2 = r1
    a1, b1, a2, b2 = r2
    if x1 < a2 and a1 < x2 and y1 < b2 and b1 < y2:
        return True
    width1, width2 = x2 - x1, a2 - a1
    height1, height2 = y2 - y1, b2 - b1
    x_overlap = min(x2, a2) - max(x1, a1)
    y_overlap = min(y2, b2) - max(y1, b1)
    aligned_vert = x_overlap > align_ratio * min(width1, width2)
    aligned_horiz = y_overlap > align_ratio * min(height1, height2)

    def fill_ratio_ok(mx1, my1, mx2, my2):
        merged_area = (mx2 - mx1) * (my2 - my1)
        if merged_area <= 0:
            return False
        return (leaf_area1 + leaf_area2) / merged_area >= min_fill_ratio

    if aligned_vert:
        vgap = max(b1 - y2, y1 - b2)
        if 0 <= vgap <= gap and fill_ratio_ok(min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)):
            return True
    if aligned_horiz:
        hgap = max(a1 - x2, x1 - a2)
        if 0 <= hgap <= gap and fill_ratio_ok(min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)):
            return True
    return False


def merge_overlapping_boxes(boxes, gap=0):
    if not boxes:
        return []
    rects = [[x, y, x + w, y + h, w * h] for (x, y, w, h) in boxes]
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, x2, y2, leaf1 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                a1, b1, a2, b2, leaf2 = rects[j]
                if _boxes_should_merge([x1, y1, x2, y2], [a1, b1, a2, b2], leaf1, leaf2, gap):
                    x1, y1, x2, y2 = min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)
                    leaf1 = leaf1 + leaf2
                    used[j] = True
                    merged = True
            result.append([x1, y1, x2, y2, leaf1])
            used[i] = True
        rects = result
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2, _ in rects]


def get_candidate_boxes(image_path):
    """
    Lấy TẤT CẢ box ứng viên có cấu trúc lưới (đường kẻ ngang+dọc) trong ảnh,
    KHÔNG cố lọc xem box nào là "bảng đúng" — việc chọn bảng nào sẽ do bước
    neo từ khoá (anchor) phía sau quyết định, không phải do heuristic hình học.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
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

    table_mask = cv2.add(horiz_lines, vert_lines)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_TABLE_AREA_RATIO * w * h
    max_area = MAX_TABLE_AREA_RATIO * w * h
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if min_area <= area <= max_area:
            candidates.append((x, y, cw, ch))

    gap_px = int(MERGE_GAP_RATIO * w)
    merged = merge_overlapping_boxes(candidates, gap=gap_px)

    # Giữ luôn CẢ box gốc (chưa merge) lẫn box đã merge làm ứng viên — vì đôi khi
    # box merge đúng, đôi khi box merge bị gộp nhầm (như case SPG+COIL END) và
    # box gốc/box con mới là cái ta cần. Bước neo từ khoá sẽ tự chọn cái đúng.
    all_candidates = candidates + merged
    boxes = []
    for (x, y, cw, ch) in all_candidates:
        n_rows = count_grid_lines(horiz_lines, (x, y, cw, ch), axis='h')
        n_cols = count_grid_lines(vert_lines, (x, y, cw, ch), axis='v')
        if n_rows >= MIN_GRID_LINES and n_cols >= MIN_GRID_LINES:
            boxes.append((x, y, cw, ch))
    # loại box trùng lặp gần như hoàn toàn
    boxes = list(set(boxes))
    return boxes, img


def ocr_page_words(reader, img):
    """Chạy OCR 1 lần trên toàn trang, trả về list (cx, cy, text_lower)."""
    results = reader.readtext(img, detail=1)
    words = []
    for bbox, text, conf in results:
        if conf < MIN_CONFIDENCE:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        words.append((cx, cy, text.lower()))
    return words


def _trim_box_away_from_points(box, own_pts, foreign_pts, margin=10):
    """
    Cat bot box theo canh gan nhat voi cac diem "foreign" (khong thuoc bang nay),
    chua lai phan chua cac diem "own". Xu ly ca truc ngang (trai/phai) lan doc
    (tren/duoi) - chon truc nao tach 2 nhom diem ro rang hon.
    """
    x, y, w, h = box
    own_cx = sum(p[0] for p in own_pts) / len(own_pts)
    own_cy = sum(p[1] for p in own_pts) / len(own_pts)
    for_cx = sum(p[0] for p in foreign_pts) / len(foreign_pts)
    for_cy = sum(p[1] for p in foreign_pts) / len(foreign_pts)

    x_sep = abs(own_cx - for_cx) / max(w, 1)
    y_sep = abs(own_cy - for_cy) / max(h, 1)

    if x_sep >= y_sep:
        foreign_xs = [p[0] for p in foreign_pts]
        if for_cx >= own_cx:
            new_right = int(min(foreign_xs) - margin)
            w = max(10, new_right - x)
        else:
            new_x = int(max(foreign_xs) + margin)
            w = w - (new_x - x)
            x = new_x
    else:
        foreign_ys = [p[1] for p in foreign_pts]
        if for_cy >= own_cy:
            new_bottom = int(min(foreign_ys) - margin)
            h = max(10, new_bottom - y)
        else:
            new_y = int(max(foreign_ys) + margin)
            h = h - (new_y - y)
            y = new_y
    return (x, y, w, h)


def select_tables_by_anchor(candidate_boxes, page_words, keywords=TABLE_KEYWORDS):
    """
    Với mỗi bảng cần trong TABLE_KEYWORDS: tìm box ứng viên NHỎ NHẤT có chứa
    >=1 điểm từ khoá "include". Nếu box đó cũng lẫn điểm "exclude" -> cắt bớt
    theo cạnh gần điểm exclude nhất (không loại bỏ cả box).
    Trả về dict {table_name: (x, y, w, h)}.
    """
    def points_in_box(points, box):
        x, y, w, h = box
        return [(px, py) for (px, py, _t) in points if x <= px <= x + w and y <= py <= y + h]

    include_pts_by_name, exclude_pts_by_name = {}, {}
    for name, spec in keywords.items():
        include_pts_by_name[name] = [
            (px, py, t) for (px, py, t) in page_words if any(k in t for k in spec.get("include", []))
        ]
        exclude_pts_by_name[name] = [
            (px, py, t) for (px, py, t) in page_words if any(k in t for k in spec.get("exclude", []))
        ]

    best_box_by_name = {}
    for name, include_pts in include_pts_by_name.items():
        if not include_pts:
            print(f"       [!] Không tìm thấy từ khoá cho bảng '{name}' — bỏ qua.")
            continue
        best_box, best_area = None, None
        for box in candidate_boxes:
            if len(points_in_box(include_pts, box)) == 0:
                continue
            area = box[2] * box[3]
            if best_area is None or area < best_area:
                best_area, best_box = area, box
        if best_box is None:
            print(f"       [!] Có từ khoá nhưng không khớp box ứng viên nào cho '{name}'.")
            continue
        best_box_by_name[name] = best_box

    results = {}
    for name, box in best_box_by_name.items():
        own_pts = points_in_box(include_pts_by_name[name], box)
        # (a) điểm exclude riêng của bảng này
        foreign_pts = points_in_box(exclude_pts_by_name[name], box)
        # (b) điểm include của bảng KHÁC đang dùng chung box này (2 bảng bị gộp
        #     vào 1 candidate duy nhất, không có box con nào tách sẵn)
        for other_name, other_box in best_box_by_name.items():
            if other_name != name and other_box == box:
                foreign_pts += points_in_box(include_pts_by_name[other_name], box)

        if foreign_pts:
            new_box = _trim_box_away_from_points(box, own_pts, foreign_pts)
            print(f"       [i] Bảng '{name}' bị lẫn vùng khác -> đã cắt bớt biên {box} -> {new_box}.")
            box = new_box

        results[name] = box
    return results


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

    print(f"[1/5] Render PDF -> ảnh: {pdf_path}")
    image_paths = render_pdf_to_images(pdf_path, page_dir)

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[2/5] Dò box ứng viên trang {idx}: {img_path}")
        candidate_boxes, img = get_candidate_boxes(img_path)
        print(f"       -> {len(candidate_boxes)} box ứng viên")

        print(f"[3/5] OCR toàn trang để lấy vị trí từ khoá (trang {idx})")
        page_words = ocr_page_words(reader, img)

        print(f"[4/5] Neo từ khoá -> chọn bảng đích (trang {idx})")
        tables = select_tables_by_anchor(candidate_boxes, page_words)
        print(f"       -> chọn được {len(tables)}/{len(TABLE_KEYWORDS)} bảng: {list(tables.keys())}")

        # Ảnh debug: đỏ = bảng được chọn, xanh mờ = box ứng viên khác (để soi khi cần chỉnh keyword)
        debug_img = img.copy()
        for (x, y, w, h) in candidate_boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 200, 0), 1)
        for name, (x, y, w, h) in tables.items():
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(debug_img, name, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(page_dir, f"debug_page_{idx:03d}.png"), debug_img)

        selected_boxes = list(tables.values())
        for name, box in tables.items():
            crop = crop_and_upscale(img, box)
            crop_path = os.path.join(page_dir, f"page_{idx:03d}_{name}.png")
            cv2.imwrite(crop_path, crop)

            print(f"[5/5] OCR bảng '{name}' (trang {idx})")
            lines = ocr_array(reader, crop)
            table_text = "\n".join(text for text, conf in lines)
            with open(crop_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                f.write(table_text)

            for text, conf in lines:
                summary_rows.append({
                    "file": base_name, "page": idx, "region": name,
                    "text": text, "confidence": round(conf, 3),
                })

        remainder = mask_out_boxes(img, selected_boxes)
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
        print("Cách dùng: python ocr_pdf_pipeline_v3.py <file.pdf hoặc thư_mục>")
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
    print("Kiểm tra debug_page_XXX.png: khung đỏ = bảng đã chọn, khung vàng mờ = box ứng viên còn lại.")


if __name__ == "__main__":
    main()
