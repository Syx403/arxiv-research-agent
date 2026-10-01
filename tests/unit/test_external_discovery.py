from datetime import date
from src.core.research_policy import date_window, date_mode


def test_date_window_accepts_chinese_year_but_never_interprets_arxiv_id_as_year():
    assert date_window("2026年的MoE论文", date(2026, 9, 15)) == ("2026-01-01", "2026-09-15")
    assert date_window("Explain arxiv:2009.12345", date(2026, 9, 15))[0] is None
    assert date_window("latest agent memory", date(2026, 9, 15))[0] == "2025-09-15"
    assert date_window("last year", date(2026, 9, 15)) == ("2025-01-01", "2025-12-31")

def test_freshness_preference_and_explicit_dates_are_different():
    assert date_mode("最新 DeepSeek KV cache 压缩进展") == "preferred"
    assert date_mode("2026年 KV cache 论文") == "required"
    assert date_mode("最近三个月的论文") == "required"
    assert date_window("最近三个月的论文", date(2026, 9, 15)) == ("2026-06-15", "2026-09-15")
    assert date_window("last 1 month", date(2026, 3, 31)) == ("2026-02-28", "2026-03-31")
