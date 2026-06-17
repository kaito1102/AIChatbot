import asyncio
import glob
import os
from functools import partial
from networkx import config
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
from lightrag.utils import EmbeddingFunc

# ========== CẤU HÌNH ==========
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "qwen3:1.7b"          # Model chat
OLLAMA_VLM_MODEL = "llama3.2-vision"     # Model xử lý ảnh (nếu có)
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Model embedding
EMBED_DIM = 768                           # Dimension của nomic-embed-text

INPUT_DIR = "./documents"   # Thư mục chứa file tài liệu của bạn
OUTPUT_DIR = "./output"
STORAGE_DIR = "./rag_storage"
# ==============================


def build_llm_func(model_name):
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("base_url", None)
        kwargs.pop("host", None)
        kwargs.pop("model", None)  # Xóa nếu lightrag tự truyền vào
        return await _ollama_model_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            host=OLLAMA_BASE_URL,
            **kwargs,
        )
    return llm_func


def build_vision_func(llm_func):
    """Vision func: dùng VLM khi có ảnh, fallback về LLM thường cho text."""
    async def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        if messages:
            # Parse messages to extract prompt, system_prompt, history, and image URLs
            parsed_prompt = ""
            parsed_system_prompt = None
            parsed_history_messages = []
            parsed_image_inputs = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    parsed_system_prompt = content
                elif role == "user":
                    if isinstance(content, str):
                        parsed_prompt = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    parsed_prompt = item.get("text", "")
                                elif item.get("type") == "image_url":
                                    url = item.get("image_url", {}).get("url", "")
                                    parsed_image_inputs.append(url)
                else:
                    parsed_history_messages.append(msg)
                    
            return await _ollama_model_if_cache(
                OLLAMA_VLM_MODEL,
                parsed_prompt,
                system_prompt=parsed_system_prompt,
                history_messages=parsed_history_messages,
                image_inputs=parsed_image_inputs,
                host=OLLAMA_BASE_URL,
                **kwargs,
            )
        elif image_data:
            image_url = f"data:image/jpeg;base64,{image_data}" if not image_data.startswith("data:") else image_data
            return await _ollama_model_if_cache(
                OLLAMA_VLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                image_inputs=[image_url],
                host=OLLAMA_BASE_URL,
                **kwargs,
            )
        else:
            return await llm_func(prompt, system_prompt, history_messages, **kwargs)
    return vision_model_func


async def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=STORAGE_DIR,
        parser="docling",        # Dùng Docling — nhẹ hơn MinerU
        parse_method="auto",
        # enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    llm_func = build_llm_func(OLLAMA_LLM_MODEL)
    vision_func = build_vision_func(llm_func)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=lambda texts: ollama_embed.func(
            texts,
            embed_model=OLLAMA_EMBED_MODEL,
            host=OLLAMA_BASE_URL,
        ),
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        # vision_model_func=vision_func,
        embedding_func=embedding_func,
    )

    # ---- XỬ LÝ TÀI LIỆU ----
    print(f"Đang xử lý tài liệu trong '{INPUT_DIR}'...")
    supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.pptx', '*.ppt', '*.xlsx', '*.xls', '*.png', '*.jpg', '*.jpeg']
    all_files = []
    for ext in supported_extensions:
        all_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not all_files:
        print("Không tìm thấy file nào trong thư mục documents/")
        return

    for file_path in all_files:
        print(f"  Đang xử lý: {file_path}")
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            parse_method="auto"
        )

    print("Xử lý xong!\n")

    # ---- QUERY ----
    query = "Tóm tắt nội dung chính của tài liệu"
    print(f"Query: {query}")
    result = await rag.aquery(query, mode="hybrid")
    print(f"\nKết quả:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())