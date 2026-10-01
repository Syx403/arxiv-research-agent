"""Validate links in historical saved answers. Not a discovery provider."""
import hashlib
from urllib.parse import urlsplit


def canonical_report_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.query or parsed.fragment or ".." in parsed.path or "%" in parsed.path:
        raise ValueError("Invalid official report URL")
    if not parsed.path.lower().endswith(".pdf"):
        raise ValueError("Official evidence must be a PDF report")
    if parsed.netloc == "huggingface.co" and parsed.path.startswith("/deepseek-ai/"):
        if "/blob/" not in parsed.path and "/resolve/" not in parsed.path:
            raise ValueError("Invalid publisher report path")
        return "https://huggingface.co" + parsed.path.replace("/blob/", "/resolve/", 1)
    if parsed.netloc == "fe-static.deepseek.com":
        return url
    raise ValueError("Report URL is outside registered publishers")

def report_id(url: str) -> str:
    return "report:" + hashlib.sha256(canonical_report_url(url).encode()).hexdigest()[:32]
