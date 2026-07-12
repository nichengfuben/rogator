#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件内容合并工具
功能：将指定目录（包括子目录）下所有UTF-8编码的文件内容合并到一个文档中
"""

import os
import sys
from pathlib import Path
import argparse
import mimetypes
from datetime import datetime

def is_text_file(file_path):
    """判断文件是否为文本文件"""
    text_extensions = {
        '.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv',
        '.md', '.rst', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.log',
        '.sh', '.bat', '.sql', '.java', '.cpp', '.c', '.h', '.hpp',
        '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.ts', '.jsx', '.tsx', ".vue"
    }
    if file_path.suffix.lower() in text_extensions:
        return True
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type is not None and mime_type.startswith('text/')

def read_file_content(file_path, encoding='utf-8'):
    """读取文件内容，自动处理编码问题"""
    for enc in [encoding, 'utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None
    return None

def should_exclude_file(file_path, exclude_patterns):
    """判断是否应该排除��文件"""
    filename = file_path.name
    default_excludes = {
        'upload.txt', 'output.txt', 'merged.txt',
        '.gitignore', '.DS_Store', 'Thumbs.db'
    }
    if filename in default_excludes:
        return True
    return any(pattern in filename for pattern in exclude_patterns)

def generate_file_header(file_path, root_path):
    """生成文件头部信息"""
    relative_path = file_path.relative_to(root_path)
    return f"# {relative_path}\n"

def _should_process_file(file_path, output_path, script_path, include_hidden, exclude_patterns, only_text_files, stats):
    """判断��件是否应该被处理，返回 True 表示应该处理"""
    if file_path.is_dir():
        return False
    if file_path.resolve() == output_path.resolve():
        return False
    if script_path and file_path.resolve() == script_path.resolve():
        return False
    if not include_hidden and any(part.startswith('.') for part in file_path.parts):
        stats['excluded_files'] += 1
        return False
    if should_exclude_file(file_path, exclude_patterns):
        stats['excluded_files'] += 1
        return False
    if only_text_files and not is_text_file(file_path):
        stats['skipped_files'] += 1
        if stats['skipped_files'] <= 10:
            print(f"跳过非文本文件: {file_path.relative_to(Path(stats['root']))}")
        return False
    return True

def _write_file_content(output, file_path, root, stats):
    """处理单个文件��内容写入"""
    stats['total_files'] += 1
    content = read_file_content(file_path)
    if content is None:
        stats['encoding_errors'] += 1
        print(f"警告: 无法读取文件编码 {file_path.relative_to(root)}")
        return
    output.write(generate_file_header(file_path, root))
    output.write(content)
    output.write("\n\n" + "-" * 50 + "\n\n")
    stats['processed_files'] += 1
    stats['total_size'] += len(content.encode('utf-8'))
    if stats['processed_files'] <= 20:
        print(f"已处理: {file_path.relative_to(root)} ({len(content)} 字符)")
    elif stats['processed_files'] == 21:
        print("...")

def merge_files_to_document(root_path=".", output_file="upload.txt",
                            include_hidden=False, exclude_patterns=None,
                            only_text_files=True, script_path=None):
    """将目录下所有文件合并到一个文档���"""
    if exclude_patterns is None:
        exclude_patterns = []

    root = Path(root_path).resolve()
    output_path = root / output_file
    stats = {
        'total_files': 0, 'processed_files': 0, 'skipped_files': 0,
        'excluded_files': 0, 'encoding_errors': 0, 'total_size': 0,
        'root': str(root),
    }
    print(f"开始处理目录: {root}")
    print(f"输出文件: {output_path}")
    print("-" * 50)

    try:
        with open(output_path, 'w', encoding='utf-8') as output:
            output.write(f"文件内容合并��档\n")
            output.write(f"源目录: {root}\n")
            output.write("=" * 60 + "\n\n")
            for file_path in root.rglob('*'):
                if _should_process_file(file_path, output_path, script_path, include_hidden, exclude_patterns, only_text_files, stats):
                    _write_file_content(output, file_path, root, stats)
        print("-" * 50)
        print("处理完成!")
    except Exception as e:
        print(f"处理过程中出错: {e}")

    return stats

def print_statistics(stats):
    """打印统计信息"""
    print(f"\n统计信息:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  已处理文件: {stats['processed_files']}")
    print(f"  跳过文件: {stats['skipped_files']}")
    print(f"  排除文件: {stats['excluded_files']}")
    print(f"  编码错误: {stats['encoding_errors']}")
    size_mb = stats['total_size'] / (1024 * 1024)
    print(f"  总��符数: {stats['total_size']:,}")
    if size_mb > 0:
        print(f"  估计大小: {size_mb:.2f} MB")

def merge_directory(directory_path, output_file="merged_output.txt"):
    """对外封装的函��：传入指定目录路径，合并其下所有文件"""
    if not Path(directory_path).is_dir():
        raise ValueError(f"指定路径���是目录: {directory_path}")
    script_path = Path(__file__).resolve()
    stats = merge_files_to_document(
        root_path=directory_path, output_file=output_file,
        include_hidden=False, exclude_patterns=[],
        only_text_files=True, script_path=script_path
    )
    print_statistics(stats)
    return stats

def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将目录下所有UTF-8文件内容合并到一��文档中",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python file_merger.py                           # 基本使用
  python file_merger.py -o merged.txt             # 指定输出文件名
  python file_merger.py -a                        # 包含隐藏文件
  python file_merger.py -e .log .tmp              # 排除特定文件
  python file_merger.py -b                        # 包含二进制文件
  python file_merger.py /path/to/directory        # 指定目录
        """
    )
    parser.add_argument("path", nargs="?", default=".", help="要处理的目录路径 (默认: 当前目录)")
    parser.add_argument("-o", "--output", default="upload.txt", help="输出文件名 (默认: upload.txt)")
    parser.add_argument("-a", "--all", action="store_true", help="包含隐藏文件和目��")
    parser.add_argument("-e", "--exclude", nargs="*", default=[], help="排除文件模式 (可以指定多个)")
    parser.add_argument("-b", "--binary", action="store_true", help="包含二进制文件 (默认只处理文本文件)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    return parser.parse_args()

def _validate_path(path):
    """验证路径有效性"""
    if not os.path.exists(path):
        print(f"错误: 路径 '{path}' 不存在")
        sys.exit(1)
    if not os.path.isdir(path):
        print(f"错误: '{path}' 不是一个目录")
        sys.exit(1)

def _handle_output(path, output_file, stats):
    """处理输出结果"""
    output_path = Path(path) / output_file
    if stats['processed_files'] > 0:
        print(f"\n输出文件已生成: {output_path.absolute()}")
    else:
        print(f"\n没有找到可处��的文件")
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass

def main():
    args = _parse_args()
    _validate_path(args.path)
    script_path = Path(__file__).resolve()
    stats = merge_files_to_document(
        root_path=args.path, output_file=args.output,
        include_hidden=args.all, exclude_patterns=args.exclude,
        only_text_files=not args.binary, script_path=script_path
    )
    print_statistics(stats)
    _handle_output(args.path, args.output, stats)

if __name__ == "__main__":
    main()
