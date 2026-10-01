"""Run with python -m scripts.configure_deepseek_key; never prints the key."""
from __future__ import annotations

import subprocess

from src.core.credentials import save_deepseek_key


def main() -> None:
    # Standard Additions shows a native masked dialog. Capture its return value
    # in memory, never in a terminal, process argument, or repository file.
    script = '''text returned of (display dialog "输入 DeepSeek 官方 API key。仅保存到 macOS 钥匙串的 arxiv-research-agent 条目；不会写入聊天、仓库或日志。输入不会触发 API 测试。" default answer "" with hidden answer buttons {"取消", "保存到钥匙串"} default button "保存到钥匙串" cancel button "取消" with title "ARA — DeepSeek API key")'''
    result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True)
    if result.returncode:
        print("Key setup cancelled or native dialog unavailable. No key was saved.")
        return
    try:
        save_deepseek_key(result.stdout.strip())
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(1) from None
    finally:
        result.stdout = ""
    print("DeepSeek key saved to the ARA-only macOS Keychain item. No API request was made.")


if __name__ == "__main__":
    main()
