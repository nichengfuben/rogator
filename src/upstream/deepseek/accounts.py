from __future__ import annotations

"""DeepSeek 上游账号池。"""

from core.session.accounts import (
    Account,
    accounts_csv_path,
    accounts_for_upstream,
)

ACCOUNTS_CSV_PATH = accounts_csv_path("deepseek")
ACCOUNTS = accounts_for_upstream("deepseek")

__all__ = ["ACCOUNTS", "ACCOUNTS_CSV_PATH", "Account", "accounts_for_upstream"]
