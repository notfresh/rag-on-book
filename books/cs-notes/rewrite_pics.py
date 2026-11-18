#!/usr/bin/env python3
"""把 CS-Notes 语料里的相对图片路径改写成 GitHub raw 绝对 URL。

原因：pics/ 目录 112MB，不适合进 git 仓；但 md 里写的是相对路径
![](pics/xxx.png)，拷到本地后图片全 404。改成指向 upstream raw 地址，
图片照样能显示，仓库保持轻量。

用法：python3 rewrite_pics.py <notes_dir>
"""
import os
import re
import sys

RAW_BASE = "https://raw.githubusercontent.com/CyC2018/CS-Notes/master/notes"

# 只改 src="pics/xxx" 这种相对引用（html img 和 markdown 图片两种写法）
PATTERNS = [
    (re.compile(r'(<img[^>]*\ssrc=")(pics/[^"]+)(")'), r'\1' + RAW_BASE + r'/\2\3'),
    (re.compile(r'(!\[[^\]]*\]\()(pics/[^)]+)(\))'), r'\1' + RAW_BASE + r'/\2\3'),
]


def main():
    notes_dir = sys.argv[1]
    changed = 0
    total = 0
    for fn in sorted(os.listdir(notes_dir)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(notes_dir, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        orig = text
        for pat, rep in PATTERNS:
            text = pat.sub(rep, text)
        total += 1
        if text != orig:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            changed += 1
    print(f"处理 {total} 个 md，改写 {changed} 个")


if __name__ == "__main__":
    main()
