#!/usr/bin/env python3
"""
RAG on Book — index.py：把 parsed.jsonl 的块 embedding 后写入向量库

技术栈与 service/index.py 完全一致（V1 已验证）:
  - embedding:  bge-small-zh-v1.5 本地 CPU（512 维, 中文友好）
  - 向量库:     milvus-lite 嵌入式（无服务无依赖）
  - 相似度:     COSINE + FLAT 索引

用法:
  python3 index.py --full         # 全量重建（每本书各自重建）
  python3 index.py --book ddia    # 只索引指定书（默认 books/ 下全部）
  python3 index.py                # 增量（V1: 解析文件 mtime 判断, 见 Future.md）

一书一库（2026-11-18）：每本书有自己的向量库 books/<id>/vector/book_chunks.db，
所以主键书内自增（不需要跨书分段），同书重跑先清空该书 collection 再灌。
"""

import argparse
import json
import os
import sys
import time

import books as books_mod

COLLECTION = "book_chunks"

# ---------- 负载控制（4 核机器防卡死, 同 service/index.py） ----------
ENCODE_BATCH = 32
ENCODE_THREADS = 2
ENCODE_SLEEP = 0.3


# ---------- 向量库 ----------
def get_client(book_id: str):
    """打开（或新建）某本书自己的向量库。

    每本书一个独立的 Milvus Lite 文件 —— 实测多实例可同时打开互不干扰。
    """
    from pymilvus import MilvusClient
    vdir = books_mod.vector_dir(book_id)
    os.makedirs(vdir, exist_ok=True)
    try:
        return MilvusClient(books_mod.vector_db_path(book_id))
    except Exception as e:
        # Milvus Lite 是单文件锁：web.py 开着时会锁住这本书的库
        if "Open local milvus failed" in str(e) or "DataDirLocked" in str(e):
            raise RuntimeError(
                f"书 {book_id} 的向量库被锁住了 —— web.py 正在占用它。"
                f"先停掉 web.py 再操作（重建索引 / CLI 提问都需要）。"
            ) from e
        raise


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
        print(f"  已创建 collection {COLLECTION} + COSINE/FLAT 索引")


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


def load_chunks(book_id: str):
    path = os.path.join(books_mod.book_dir(book_id), "parsed.jsonl")
    if not os.path.exists(path):
        print(f"  {book_id}: 无 parsed.jsonl（先跑 parse.py）")
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

    V2（Layer 2）：把 heading_path 拼到编码文本头部再 embedding，让 bge 看到章节位置。
    这样"第N章"/"某节标题"这种元信息 token 在向量空间里也能参与相似度匹配。
    定位链**只参与编码，不入库正文**（正文在 parsed.jsonl 里已是干净的）。
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
    ap = argparse.ArgumentParser(description="书块 → embedding → 该书自己的向量库")
    ap.add_argument("--full", action="store_true", help="全量重建（drop 后重灌）")
    ap.add_argument("--book", default=None, help="指定书 id（默认全部）")
    args = ap.parse_args()

    # 书清单以 manifest 为准（books.py 是单一来源），不是扫目录
    book_ids = [args.book] if args.book else books_mod.list_book_ids()
    if not book_ids:
        print("books/ 下没有书（缺 book.json）")
        return

    model = load_model()
    total = 0
    for book_id in book_ids:
        chunks = load_chunks(book_id)
        if not chunks:
            continue
        # 每本书一个独立向量库 —— 客户端在这一行按书打开
        client = get_client(book_id)
        if args.full and client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
            print(f"  [{book_id}] 已清空 collection")
        ensure_collection(client)
        # V1 无增量（全量重建幂等）: 该书 collection 全清后重灌
        n = index_chunks(client, model, chunks)
        total += n
        print(f"  [{book_id}] 入库 {n} 块 → {books_mod.vector_db_path(book_id)}")

    print("=" * 50)
    print(f"完成: {len(book_ids)} 本书, {total} 块")


if __name__ == "__main__":
    main()
