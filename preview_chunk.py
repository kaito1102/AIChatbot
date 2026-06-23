import argparse
import os
from pathlib import Path
from smart_pdf_loader import load_directory_smart

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = os.path.join(BASE_DIR, "data_sources")
IMAGE_DIR   = os.path.join(BASE_DIR, "extracted_images")
OUTPUT_FILE = os.path.join(BASE_DIR, "chunks_preview.txt")


def main(data_dir: str, output_file: str):
    print(f"📂 Loading documents from '{data_dir}'...")
    chunks = load_directory_smart(
        data_dir,
        pdf_kwargs={
            "chunk_by":             "page",
            "table_format":         "markdown",
            "image_dir":            IMAGE_DIR,
            "min_image_size":       100,
            "skip_repeated_header": True,
        },
    )

    if not chunks:
        print("❌ Không tìm thấy tài liệu nào.")
        return

    # Không split — mỗi page doc là 1 chunk
    table_chunks = sum(1 for c in chunks if c.metadata.get("has_table"))
    image_chunks = sum(1 for c in chunks if c.metadata.get("has_image"))
    print(f"✅ {len(chunks)} page-chunks ({table_chunks} có bảng, {image_chunks} có ảnh)")

    lines = []
    lines.append("=" * 70)
    lines.append(f"THƯ MỤC   : {data_dir}")
    lines.append(f"TỔNG CHUNK: {len(chunks)}  |  Có bảng: {table_chunks}  |  Có ảnh: {image_chunks}")
    lines.append("=" * 70)

    for i, chunk in enumerate(chunks, 1):
        m        = chunk.metadata
        filename = Path(m.get("source", "unknown")).name

        # ── Ảnh: đọc từ images_meta (có tọa độ từng ảnh) ────────────
        images_meta = m.get("images_meta", [])
        img_names   = [img["name"] for img in images_meta]

        lines.append(f"\n{'─' * 70}")
        lines.append(
            f"CHUNK #{i:>3}  |  {filename}  |  Trang {m.get('page')}/{m.get('total_pages')}"
            f"  |  Bảng: {'✓' if m.get('has_table') else '✗'}"
            f"  |  Ảnh: {'✓' if m.get('has_image') else '✗'}"
            f"  |  {len(chunk.page_content)} ký tự"
        )

        if img_names:
            lines.append(f"🖼  Ảnh ({len(img_names)} file): {', '.join(img_names)}")
            # In tọa độ từng ảnh riêng
            for img in images_meta:
                lines.append(
                    f"   📍 {img['name']}: "
                    f"x0={img['x0']:.1f}, x1={img['x1']:.1f}, "
                    f"top={img['top']:.1f}, bottom={img['bottom']:.1f}"
                )

        lines.append("─" * 70)
        lines.append(chunk.page_content)

    lines.append(f"\n{'=' * 70}")
    lines.append(f"END — {len(chunks)} chunks total")
    lines.append("=" * 70)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Đã xuất {len(chunks)} chunks → '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview chunks từ tài liệu (page-level)")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output",   default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.data_dir, args.output)