"""Parse in a short-lived process so cancelling acquisition stops PDF work too."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from src.corpus.chunking import Chunk, chunk_sections
from src.corpus.pdf_loader import load_pdf
from src.corpus.html_loader import load_html


async def parse_chunks(path: Path) -> list[Chunk]:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "src.corpus.parse_worker", str(path.resolve()),
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await process.communicate()
        if process.returncode:
            raise ValueError("Original text parsing failed")
        return [Chunk(**chunk) for chunk in json.loads(stdout)]
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


if __name__ == "__main__":
    path = Path(sys.argv[1])
    sections = load_html(path.read_text()) if path.suffix == ".html" else load_pdf(path)
    chunks = chunk_sections(sections)
    print(json.dumps([asdict(chunk) for chunk in chunks]))
