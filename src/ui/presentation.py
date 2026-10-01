"""Only delivered claims and their bound evidence enter the public response."""
from __future__ import annotations

from src.agent.results import paper_url as paper_url, present_state as present_state


STAGES = {
    "control_research": "判断下一步", "expand_reading": "按证据缺口补查",
    "decompose": "理解问题", "retrieve": "检索论文", "multi_hop": "扩展引用",
    "synthesize": "整理回答", "self_rag": "核验证据", "finalize": "交付结果",
    "reflect": "保存会话",
    "discover": "论文发现与获取",
    "plan_papers": "理解需求并设计查询", "search_papers": "搜索 arXiv 论文",
    "assess_papers": "筛选论文并检查覆盖", "present_papers": "整理论文简析",
    "read_papers": "按需读取原文",
}
STOP_LABELS = {
    "clarification_required": "需要明确研究范围",
    "search_limit": "本轮达到搜索预算，召回仍不充分",
    "search_exhausted": "已尝试当前可用查询，尚无足够匹配结果",
    "turn_timeout": "本轮达到时间上限",

    "answer_verified": "已完成本轮证据核验",
    "evidence_incomplete": "部分问题缺少证据",
    "verification_retry_limit": "达到核验重试上限",
    "verification_failed": "部分结论未通过核验",
    "no_evidence": "未找到足够证据",
    "evidence_judgment_failed": "部分证据判断未完成；不代表没有相关论文",
    "synthesis_incomplete": "回答生成不完整",
    "provider_unavailable": "模型服务暂时不可用",
    "budget_exhausted": "达到调用预算或请求次数上限",
    "budget_wait_timeout": "等待其他请求结算超时，本轮已停止",
    "verification_time_guard": "剩余时间不足以继续核验修复，已保留核验内容",
    "cancelled": "已停止；已发出的请求可能仍产生费用",
    "interrupted": "服务曾中断，本轮未完成",
    "runtime_error": "本轮运行失败，请检查本地服务和配置后重试",
    "external_discovery_incomplete": "外部论文获取未完成，详见本轮过程",
    "external_search_unavailable": "外部论文搜索失败",
    "external_discovery_timeout": "外部论文获取超时",
    "no_matching_papers": "本次查询未取得匹配论文",
    "full_text_unavailable": "已找到候选，全文获取未完成",
    "specified_paper_unavailable": "指定论文的信息获取失败，尚未读取正文",
    "candidate_assessment_failed": "论文筛选服务未完成，已保留候选",
    "candidate_output_invalid": "论文筛选输出校验失败，已保留候选",
    "search_plan_invalid": "搜索计划格式校验失败，尚未查询论文",
    "search_planning_failed": "搜索计划解析失败，尚未完成论文查询",
    "verification_unavailable": "部分核验未完成，已保留可交付内容",
    "papers_found": "已完成本轮论文筛选与摘要简析",
    "paper_search_partial": "已保留论文结果，仍有未完成的查询或待补充方向",
}






def usage_delta(snapshot: dict, start_index: int) -> dict:
    rows = snapshot["requests"][start_index:]
    return {
        "requests": len(rows),
        "estimated_usd": sum(r["estimated_usd"] for r in rows if r["status"] == "settled"),
        "uncertain_usd": sum(r["estimated_usd"] for r in rows if r["status"] != "settled"),
    }
