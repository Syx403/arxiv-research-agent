"""Repair section metadata from local PDFs without changing text, IDs, or embeddings.

Dry-run by default. --apply saves an old/new section journal before a transaction.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from src.core.db import _psycopg_conninfo
from src.corpus.pdf_loader import load_pdf


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def infer_section(text, sections):
    normalized = normalize(text)
    # Multiple exact anchors detect chunks that crossed previously missed headings.
    anchors = [normalized[i:i + 100] for i in range(0, max(1, len(normalized) - 99), 160)]
    anchors = [a for a in anchors if len(a) >= 50]
    # Repeated captions, prompts, or headers cannot identify a section. Only an
    # anchor unique to one section name contributes to its classification.
    matched_names = set()
    for anchor in anchors:
        names = {name for name, body in sections if anchor in body}
        if len(names) == 1:
            matched_names.update(names)
    found = list(dict.fromkeys(name for name, _ in sections if name in matched_names))
    if not found:
        return None
    return found[0] if len(found) == 1 else "Mixed: " + " / ".join(found)


async def main(args):
    updates, skipped = [], []
    async with await AsyncConnection.connect(_psycopg_conninfo(), row_factory=dict_row) as conn:
        await conn.execute('SET TRANSACTION READ ONLY')
        cur = await conn.execute("SELECT paper_id,pdf_path FROM papers WHERE ingestion_status='complete' ORDER BY paper_id")
        papers = await cur.fetchall()
        if args.paper_id:
            papers = [p for p in papers if p['paper_id'] in args.paper_id]
        for index, paper in enumerate(papers):
            path = Path(paper['pdf_path'] or '')
            if not path.is_file():
                skipped.append(paper['paper_id'])
                continue
            sections = [(s.name, normalize(s.text)) for s in load_pdf(path)]
            cur = await conn.execute('SELECT chunk_id,section,text FROM chunks WHERE paper_id=%s ORDER BY ord', (paper['paper_id'],))
            for row in await cur.fetchall():
                section = infer_section(row['text'], sections)
                if section and section != row['section']:
                    updates.append({'chunk_id': row['chunk_id'], 'paper_id': paper['paper_id'],
                                    'old': row['section'], 'new': section})
            if (index + 1) % 10 == 0:
                print(json.dumps({'papers_scanned':index+1,'metadata_changes':len(updates)}), flush=True)
    journal = {'papers_scanned':len(papers), 'missing_pdfs':skipped, 'updates':updates}
    output = Path(args.output) if args.output else Path('data/eval_outputs') / ('section-repair-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(journal, handle, ensure_ascii=False, indent=2)
    if args.apply:
        async with await AsyncConnection.connect(_psycopg_conninfo()) as conn:
            for row in updates:
                cur = await conn.execute('UPDATE chunks SET section=%s WHERE chunk_id=%s AND section IS NOT DISTINCT FROM %s',
                                         (row['new'], row['chunk_id'], row['old']))
                if cur.rowcount != 1:
                    raise RuntimeError('Section changed concurrently; transaction rolled back')
    print(json.dumps({'applied':args.apply, 'changes':len(updates), 'journal':str(output),
                      'changed_papers':dict(Counter(u['paper_id'] for u in updates)), 'missing_pdfs':skipped}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--output')
    asyncio.run(main(parser.parse_args()))
