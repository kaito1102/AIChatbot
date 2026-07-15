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
MIN_TABLE_AREA_RATIO = 0.005   # vùng bảng phải chiếm ít nhất 0.5% diện tích trang (giảm để bắt bảng nhỏ)
MAX_TABLE_AREA_RATIO = 0.4     # loại bỏ vùng quá lớn (thường là khung viền ngoài của cả trang, không phải bảng)
LINE_KERNEL_SCALE = 40         # càng lớn -> kernel càng nhỏ -> bắt được đường kẻ ngắn hơn (bảng nhỏ)
MIN_GRID_LINES = 2              # số đường kẻ ngang/dọc tối thiểu để coi là "bảng" (loại đường kích thước/mũi tên)
PADDING = 6                     # padding quanh vùng bảng khi cắt (px)
UPSCALE_FACTOR = 2              # phóng to ảnh bảng trước khi OCR để tăng độ chính xác
MERGE_GAP_RATIO = 0.003          # gộp các bảng cách nhau trong khoảng này (tỉ lệ % chiều rộng trang)
                                 # -> ví dụ nhiều hàng title-block đứng sát nhau sẽ gộp thành 1 vùng crop
                                 # tăng giá trị này nếu vẫn còn bảng liền kề bị cắt riêng lẻ
CROP_MODE = "fixed"

# Sửa đổi phần CROP_REGIONS trong file gốc

CROP_REGIONS = [
    # Các vùng bảng cố định (điều chỉnh theo layout của bạn)
    {"name": "spec_table", "x": 0.410, "y": 0.289, "w": 0.541, "h": 0.281},
    {"name": "title_block", "x": 0.432, "y": 0.820, "w": 0.412, "h": 0.150},
    
    # THÊM BẢNG MTRL INDIC. - Vị trí cần điều chỉnh dựa trên ảnh thực tế
    # Dựa vào kết quả OCR, bảng này nằm ở bên phải, gần giữa trang
    {"name": "mtrl_indic", "x": 0.27490, "y": 0.7380, "w": 0.33530, "h": 0.0830},
]


def get_fixed_table_regions(img):
    """Chuyển CROP_REGIONS (tỉ lệ) thành box pixel (x, y, w, h) theo kích thước ảnh thực tế."""
    h, w = img.shape[:2]
    boxes = []
    for region in CROP_REGIONS:
        x = int(region["x"] * w)
        y = int(region["y"] * h)
        rw = int(region["w"] * w)
        rh = int(region["h"] * h)
        boxes.append((x, y, rw, rh))
    return boxes


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
    """
    Đếm số đường kẻ riêng biệt (dòng/cột) bên trong 1 box, dựa trên line_mask
    (horiz_lines hoặc vert_lines). Dùng để phân biệt "bảng thật" (nhiều dòng kẻ
    tạo thành lưới) với "đường kích thước/mũi tên" (chỉ 1-2 nét đơn).

    axis='h': đếm số hàng kẻ ngang phân biệt theo trục Y
    axis='v': đếm số cột kẻ dọc phân biệt theo trục X
    min_run_ratio: 1 đường kẻ phải dài tối thiểu bằng tỉ lệ này so với box để được tính
    """
    x, y, w, h = box
    crop = line_mask[y:y + h, x:x + w]
    if crop.size == 0:
        return 0

    if axis == 'h':
        row_has_line = (crop.sum(axis=1) > (w * min_run_ratio * 255))
        # đếm số cụm liên tiếp True (mỗi cụm = 1 đường kẻ ngang)
        changes = np.diff(row_has_line.astype(int))
        count = int((changes == 1).sum()) + (1 if row_has_line[0] else 0)
    else:
        col_has_line = (crop.sum(axis=0) > (h * min_run_ratio * 255))
        changes = np.diff(col_has_line.astype(int))
        count = int((changes == 1).sum()) + (1 if col_has_line[0] else 0)
    return count

def count_intersections(horiz_lines, vert_lines, box):
    """
    Đếm số giao điểm giữa line ngang và line dọc.
    Bảng thật sẽ có rất nhiều giao điểm.
    """

    x, y, w, h = box

    h_crop = horiz_lines[y:y+h, x:x+w]
    v_crop = vert_lines[y:y+h, x:x+w]

    intersections = cv2.bitwise_and(h_crop, v_crop)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        intersections
    )

    count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 2:
            count += 1

    return count

def grid_density(horiz_lines, vert_lines, box):

    x, y, w, h = box

    mask = cv2.add(
        horiz_lines[y:y+h, x:x+w],
        vert_lines[y:y+h, x:x+w]
    )

    line_pixels = cv2.countNonZero(mask)

    return line_pixels / (w * h)
    
def detect_table_cells(table_mask):

    contours, _ = cv2.findContours(
        table_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cells = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        if area < 5000:
            continue

        if h < 30:
            continue

        if w < 40:
            continue

        cells.append((x, y, w, h))

    return cells

def detect_table_regions(image_path):
    """
    Dò các vùng bảng trong ảnh dựa trên đường kẻ ngang/dọc.
    Trả về danh sách bounding box (x, y, w, h).

    Thứ tự xử lý quan trọng:
    1. Lấy TẤT CẢ vùng ứng viên theo diện tích (chưa lọc cấu trúc lưới) —
       vì 1 hàng đơn lẻ của title-block có thể chỉ có 1 đường kẻ, chưa đủ
       "lưới" để giữ lại nếu lọc ngay từ đầu.
    2. GỘP các vùng liền kề/chồng nhau trước (nhiều hàng title-block đứng
       sát nhau sẽ gộp thành 1 khối).
    3. Lọc cấu trúc lưới TRÊN VÙNG ĐÃ GỘP — khối đã gộp (nhiều hàng+cột)
       chắc chắn có nhiều đường kẻ hơn, dễ phân biệt với 1 hình chữ nhật
       đơn từ đường kích thước/mũi tên (vốn không gộp được với gì khác).
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ nhẹ trước khi nhị phân hoá -> giảm nhiễu moiré khi ảnh là ảnh chụp màn hình/scan chất lượng thấp
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

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
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        print(
            f"CANDIDATE: x={x} y={y} w={cw} h={ch}"
        )

        area = cw * ch

        if min_area <= area <= max_area:
            candidates.append((x, y, cw, ch))

    # Bước 1: gộp các ứng viên liền kề/chồng nhau trước khi lọc cấu trúc
    gap_px = int(MERGE_GAP_RATIO * w)
    merged = merge_overlapping_boxes(candidates, gap=gap_px)
    print("\n=== MERGED ===")

    for b in merged:
        if 2400 < b[1] < 2900:
            print(b)

    print("================")

    # Bước 2: lọc cấu trúc lưới trên từng vùng ĐÃ GỘP
    boxes = []
    for (x, y, cw, ch) in merged:

        n_rows = count_grid_lines(
            horiz_lines,
            (x, y, cw, ch),
            axis='h'
        )

        n_cols = count_grid_lines(
            vert_lines,
            (x, y, cw, ch),
            axis='v'
        )

        n_intersections = count_intersections(
            horiz_lines,
            vert_lines,
            (x, y, cw, ch)
        )

        density = grid_density(
            horiz_lines,
            vert_lines,
            (x, y, cw, ch)
        )

        aspect_ratio = cw / ch

        # loại vùng quá dẹt
        if aspect_ratio > 12:
            continue

        # loại vùng quá nhỏ theo chiều cao
        if ch < 25:
            continue

        # bảng thật
        print(
            f"BOX ({x},{y},{cw},{ch}) "
            f"rows={n_rows} "
            f"cols={n_cols} "
            f"inter={n_intersections} "
            f"density={density:.4f}"
        )
        is_table = (
            density >= 0.02
            and n_intersections >= 5
        )

        if is_table:
            boxes.append((x, y, cw, ch))

    # Sắp theo thứ tự đọc: trên xuống dưới, trái sang phải
    print("\n===== FINAL BOXES =====")

    for box in boxes:
        print(box)

    print("=======================\n")
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, img


def _boxes_should_merge(r1, r2, leaf_area1, leaf_area2, gap, align_ratio=0.5, min_fill_ratio=0.6):
    """
    2 box được gộp nếu chồng nhau thực sự, HOẶC thẳng hàng + khoảng cách <= gap
    + hình chữ nhật kết quả không bị "rỗng" quá nhiều. leaf_area1/2 là tổng
    diện tích các box GỐC (trước khi gộp) hợp thành r1/r2 — dùng giá trị này
    (thay vì diện tích r1/r2 hiện tại) để fill_ratio chính xác qua nhiều bước gộp.
    """
    x1, y1, x2, y2 = r1
    a1, b1, a2, b2 = r2

    if x1 < a2 and a1 < x2 and y1 < b2 and b1 < y2:
        return True

    width1, width2 = x2 - x1, a2 - a1
    height1, height2 = y2 - y1, b2 - b1

    x_overlap = min(x2, a2) - max(x1, a1)
    y_overlap = min(y2, b2) - max(y1, b1)

    aligned_vert = x_overlap > 0.8 * min(width1, width2)
    aligned_horiz = y_overlap > 0.8 * min(height1, height2)

    def fill_ratio_ok(mx1, my1, mx2, my2):
        merged_area = (mx2 - mx1) * (my2 - my1)
        if merged_area <= 0:
            return False
        return (leaf_area1 + leaf_area2) / merged_area >= min_fill_ratio

    if aligned_vert:

        vgap = max(b1 - y2, y1 - b2)

        if (
            0 <= vgap <= max(gap, 15)
            and fill_ratio_ok(
                min(x1, a1),
                min(y1, b1),
                max(x2, a2),
                max(y2, b2)
            )
        ):
            return True
    if aligned_horiz:
        hgap = max(a1 - x2, x1 - a2)
        if 0 <= hgap <= gap and fill_ratio_ok(min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)):
            return True
    return False


def merge_overlapping_boxes(boxes, gap=0):
    """
    Gộp các box chồng lấn HOẶC thẳng hàng + cách nhau <= gap px + không tạo
    khung "rỗng" quá mức (xem _boxes_should_merge). gap=0: chỉ gộp box chồng nhau.
    """
    if not boxes:
        return []
    # mỗi phần tử: [x1, y1, x2, y2, tổng_diện_tích_các_box_gốc_đã_gộp_vào_đây]
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
                if _boxes_should_merge(
                    [x1, y1, x2, y2],
                    [a1, b1, a2, b2],
                    leaf1,
                    leaf2,
                    gap
                ):

                    # không gộp 2 bảng nằm trên dưới nhau nếu khoảng cách
                    # lớn hơn 15 pixel
                    vertical_gap = max(b1 - y2, y1 - b2)

                    x1, y1, x2, y2 = min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)
                    leaf1 = leaf1 + leaf2
                    used[j] = True
                    merged = True
            result.append([x1, y1, x2, y2, leaf1])
            used[i] = True
        rects = result
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2, _ in rects]


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
        img = cv2.imread(img_path)
        ph, pw = img.shape[:2]

        if CROP_MODE == "fixed":
            boxes = get_fixed_table_regions(img)
            print(f"[2/4] Dùng {len(boxes)} vùng cắt cố định (trang {idx}, kích thước ảnh {pw}x{ph})")
        else:
            print(f"[2/4] Dò vùng bảng trang {idx}: {img_path} (kích thước ảnh {pw}x{ph})")
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
