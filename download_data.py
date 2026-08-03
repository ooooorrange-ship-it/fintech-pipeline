#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SCOW 赛题数据下载脚本（支持断点续传 / 掉线自动重连）
# 使用方法: python download_data.py

import os
import sys
import time

# 配置
SERVER_HOST = "fintechpf.yunzhengdata.com"
SFTP_PORT = 2222
USERNAME = "download"
PASSWORD = os.environ.get("SCOW_DOWNLOAD_PASSWORD", "")
REMOTE_FILE = "/09-智能风控与量化建模赛道-江苏银行-基于资金图谱的涉诈账户发现与可疑链路解释.zip"
LOCAL_FILE = "09-智能风控与量化建模赛道-江苏银行-基于资金图谱的涉诈账户发现与可疑链路解释.zip"
MAX_RETRIES = 10
CHUNK_SIZE = 1024 * 1024  # 1MB

print("=" * 50)
print("SCOW 赛题数据下载工具（支持断点续传/自动重连）")
print("=" * 50)
print(f"服务器: {SERVER_HOST}:{SFTP_PORT}")
print(f"文件: {REMOTE_FILE}")
print("=" * 50)

try:
    import paramiko
except ImportError:
    print("\n[错误] 需要先安装 paramiko 库")
    print("请运行: pip install paramiko")
    input("按回车键退出...")
    sys.exit(1)


def connect():
    transport = paramiko.Transport((SERVER_HOST, SFTP_PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    transport.set_keepalive(15)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def download_with_resume():
    if not PASSWORD:
        print("[错误] 请先设置 SCOW_DOWNLOAD_PASSWORD 环境变量")
        sys.exit(1)
    transport, sftp = connect()
    total_size = sftp.stat(REMOTE_FILE).st_size
    sftp.close()
    transport.close()

    downloaded = os.path.getsize(LOCAL_FILE) if os.path.exists(LOCAL_FILE) else 0
    if downloaded > total_size:
        # 本地文件比远端还大，说明是脏数据（比如换了一份新赛题包），重新下载
        downloaded = 0

    print(f"\n文件大小: {total_size / 1024 / 1024:.1f} MB")
    if downloaded:
        print(f"检测到本地已下载 {downloaded / 1024 / 1024:.1f} MB，将从断点继续下载")
    print("开始下载...\n")

    def print_progress(curr):
        percent = (curr / total_size) * 100 if total_size else 100
        bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
        print(f"\r{bar} {percent:.1f}% ({curr/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end="", flush=True)

    attempt = 0
    while downloaded < total_size:
        attempt += 1
        if attempt > MAX_RETRIES:
            raise RuntimeError(
                f"重试 {MAX_RETRIES} 次仍未下载完成，请检查网络后重新运行本脚本"
                f"（已支持断点续传，重新运行不会从头开始）"
            )
        try:
            transport, sftp = connect()
            remote_f = sftp.open(REMOTE_FILE, 'rb')
            remote_f.seek(downloaded)
            mode = 'r+b' if os.path.exists(LOCAL_FILE) else 'wb'
            with open(LOCAL_FILE, mode) as local_f:
                local_f.seek(downloaded)
                while downloaded < total_size:
                    data = remote_f.read(CHUNK_SIZE)
                    if not data:
                        break
                    local_f.write(data)
                    downloaded += len(data)
                    print_progress(downloaded)
            remote_f.close()
            sftp.close()
            transport.close()
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"\n[网络中断，第 {attempt}/{MAX_RETRIES} 次重试，{wait}秒后继续] {e}")
            time.sleep(wait)

    print("\n\n✓ 下载完成!")
    print(f"文件保存为: {os.path.abspath(LOCAL_FILE)}")


if __name__ == "__main__":
    try:
        download_with_resume()
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        input("按回车键退出...")
