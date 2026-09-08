#!/usr/bin/env python3
"""
RAG on Book — index.py：把 parsed.jsonl 的块 embedding 后写入向量库

技术栈与 service/index.py 完全一致（V1 已验证）:
  - embedding:  bge-small-zh-v1.5 本地 CPU（512 维, 中文友好）
  - 向量库:     milvus-lite 嵌入式（.vector/book_chunks.db, 无服务无依赖）
  - 相似度:     COSINE + FLAT 索引

用法:
  python3 index.py --full         # 全量重建（清空 collection 重灌）
  python3 index.py --book ddia    # 只索引指定书（默认 books/ 下全部）
  python3 index.py                # 增量（V1: 解析文件 mtime 判断, 见 Future.md）
"""

import argparse
import json
import os
import sys
import time

BOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books")
VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vector")
DB_PATH = os.path.join(VECTOR_DIR, "book_chunks.db")
COLLECTION = "book_chunks"

# ---------- 负载控制（4 核机器防卡死, 同 service/index.py） ----------
ENCODE_BATCH = 32
ENCODE_THREADS = 2
ENCODE_SLEEP = 0.3


# ---------- 向量库 ----------
def get_client():
    from pymilvus import MilvusClient
    os.makedirs(VECTOR_DIR, exist_ok=True)
    return MilvusClient(DB_PATH)


def build_schema():
    from pymilvus import CollectionSchema, DataType, FieldSchema
    fields = [
        FieldSchema("pk", DataType.INT64, is_primary=True),
        FieldSchema("book", DataType.VARCHAR, max_length=64),
        FieldSchema("part", DataType.VARCHAR, max_length=16),
        FieldSchema("chapter", DataType.VARCHAR, max_length=16),
        FieldSchema("chapter_title", DataType.VARCHAR, max_length=256),
        FieldSchema("section", DataType.VARCHAR, max_length=256),
        FieldSchema("subsection", DataType.VARCHAR, max_length=256),
        FieldSchema("heading_path", DataType.VARCHAR, max_length=1024),
        FieldSchema("source", DataType.VARCHAR, max_length=512),
        FieldSchema("content", DataType.VARCHAR, max_length=4096),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=512),
    ]
    return CollectionSchema(fields, description="RAG on Book — book chunk semantic index")


def ensure_collection(client):
    from pymilvus.milvus_client import IndexParams
    if not client.has_collection(COLLECTION):
        client.create_collection(COLLECTION, schema=build_schema(),
                                 description="RAG on Book — book chunk semantic index")
        idx = IndexParams()
        idx.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(COLLECTION, idx)
        print("  已创建 collection book_chunks + COSINE/FLAT 索引")


# ---------- 主流程 ----------
def load_model():
    import torch
    torch.set_num_threads(ENCODE_THREADS)
    from sentence_transformers import SentenceTransformer
    print("加载 bge-small-zh-v1.5 ...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print(f"  模型就绪 ({time.time()-t0:.1f}s), 线程={ENCODE_THREADS}")
    return model


def load_chunks(book):
    path = os.path.join(BOOKS_DIR, book, "parsed.jsonl")
    if not os.path.exists(path):
        print(f"  {book}: 无 parsed.jsonl（先跑 parse.py）")
        return []
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


SCHEMA_FIELDS = [
    "pk", "book", "part", "chapter", "chapter_title", "section",
    "subsection", "heading_path", "source", "content",
]


def index_chunks(client, model, chunks):
    """chunks → embedding → upsert。分批 + 休眠控制负载。返回块数。

    V2（Layer 2）：把 heading_path 拼到 content 头部再编码，让 bge 看到章节位置。
    这样"第N章"/"某节标题"这种元信息 token 在向量空间里也能参与相似度匹配。
    """
    if not chunks:
        return 0
    # 拼接 heading_path 作为编码前缀（不入库，只用于 embedding）
    texts = [f"【{c['heading_path']}】\n{c['content']}" for c in chunks]
    batch = []
    n_done = 0
    for i in range(0, len(texts), ENCODE_BATCH):
        sub = texts[i:i + ENCODE_BATCH]
        vecs = model.encode(sub, normalize_embeddings=True)
        for c, v in zip(chunks[i:i + ENCODE_BATCH], vecs):
            row = {k: c.get(k) for k in SCHEMA_FIELDS}
            row["vector"] = v.tolist()
            batch.append(row)
            if len(batch) >= 64:
                client.upsert(COLLECTION, batch)
                batch = []
        n_done += len(sub)
        if i + ENCODE_BATCH < len(texts):
            time.sleep(ENCODE_SLEEP)
    if batch:
        client.upsert(COLLECTION, batch)
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="书块 → embedding → 向量库")
    ap.add_argument("--full", action="store_true", help="全量重建（drop 后重灌）")
    ap.add_argument("--book", default=None, help="指定书目录名（默认全部）")
    args = ap.parse_args()

    books = [args.book] if args.book else [
        d for d in sorted(os.listdir(BOOKS_DIR))
        if os.path.isdir(os.path.join(BOOKS_DIR, d))
    ]
    client = get_client()
    ensure_collection(client)

    if args.full:
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
            print("  已清空 collection")
        ensure_collection(client)

    model = load_model()
    total = 0
    for book in books:
        chunks = load_chunks(book)
        if not chunks:
            continue
        # V1 无增量（全量重建幂等）: 先删该书旧块, 再灌新块
        try:
            client.delete(COLLECTION, filter=f"book == '{book}'")
        except Exception:
            pass
        n = index_chunks(client, model, chunks)
        total += n
        print(f"  [{book}] 入库 {n} 块")

    print("=" * 50)
    print(f"完成: {len(books)} 本书, {total} 块 → {DB_PATH}")


if __name__ == "__main__":
    main()
