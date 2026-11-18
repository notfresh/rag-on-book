#!/usr/bin/env python3
"""
迁移脚本：把"多书共库"拆成"一书一库"（2026-11-18）

旧结构：.vector/book_chunks.db 里用 book 字段区分所有书
新结构：books/<id>/vector/book_chunks.db 每本书一个独立库

关键点：**直接搬 vector，不重新 embedding**（4024 块重新编码要 16 分钟 CPU）。
Milvus 的 query 能带出 vector 字段，upsert 回去即可。

用法：
  python3 migrate_split_vectors.py            # 迁移（自动备份旧库）
  python3 migrate_split_vectors.py --dry-run  # 只看会做什么
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import books as books_mod  # noqa: E402

OLD_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vector", "book_chunks.db")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vector.bak.split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(OLD_DB):
        print(f"旧库不存在: {OLD_DB}")
        return

    from pymilvus import MilvusClient
    from index import COLLECTION, build_schema

    old = MilvusClient(OLD_DB)
    old.load_collection(COLLECTION)

    book_ids = books_mod.list_book_ids()
    print(f"书: {', '.join(book_ids)}")
    print(f"旧库: {OLD_DB}")
    print(f"新位置: books/<id>/{books_mod.VECTOR_SUBDIR}/{books_mod.VECTOR_DB_NAME}")
    if args.dry_run:
        for b in book_ids:
            n = old.query(COLLECTION, filter=f"book == '{b}'", output_fields=["count(*)"])
            print(f"  [{b}] 将迁移 {n[0]['count(*)']} 块")
        return

    # 备份旧库
    if not os.path.exists(BACKUP_DIR):
        print(f"备份旧库 → {BACKUP_DIR}")
        shutil.copytree(os.path.join(os.path.dirname(OLD_DB)), BACKUP_DIR)

    total = 0
    for b in book_ids:
        rows = old.query(
            COLLECTION,
            filter=f"book == '{b}'",
            output_fields=["pk", "book", "part", "chapter", "chapter_title",
                           "section", "subsection", "heading_path", "source",
                           "content", "vector"],
            limit=100000,
        )
        if not rows:
            print(f"  [{b}] 旧库里没有块，跳过")
            continue

        # 新库：建 collection + 索引
        from index import get_client, ensure_collection
        client = get_client(b)
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
        ensure_collection(client)

        # pk 改成书内自增（旧 pk 是 slot 段位，跨书不冲突但没必要）
        batch = []
        for i, r in enumerate(rows):
            row = {k: r.get(k) for k in (
                "pk", "book", "part", "chapter", "chapter_title",
                "section", "subsection", "heading_path", "source", "content")}
            row["pk"] = i            # 书内自增
            row["vector"] = r["vector"]
            batch.append(row)
            if len(batch) >= 64:
                client.upsert(COLLECTION, batch)
                batch = []
        if batch:
            client.upsert(COLLECTION, batch)

        total += len(rows)
        print(f"  [{b}] 迁移 {len(rows)} 块 → {books_mod.vector_db_path(b)}")

    print("=" * 50)
    print(f"完成: {total} 块已拆分到 {len(book_ids)} 个独立库")
    print(f"旧库保留在 {OLD_DB}（备份 {BACKUP_DIR}）")


if __name__ == "__main__":
    main()
