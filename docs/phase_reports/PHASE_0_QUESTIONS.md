# Phase 0 Questions

## Question 1: GitHub repo creation command is rejected by installed gh CLI

**Context**: Phase 0 task 13 requires running `gh repo create arxiv-research-agent --private --source=. --remote=origin --license=mit` exactly, before moving plan/review artifacts and committing.
**Problem**: The installed `gh` CLI rejects that flag combination before making a network request: `the --source option is not supported with --clone, --template, --license, or --gitignore`. The repo has been initialized locally and the branch was renamed to `main`, but no remote has been created or set.
**Options considered**:
1. Run `gh repo create arxiv-research-agent --private --source=. --remote=origin` without `--license` — pros: preserves source/remote behavior and avoids creating remote content; cons: deviates from the exact command in §5.2 task 13.
2. Run `gh repo create arxiv-research-agent --private --license=mit`, then manually add `origin` with the returned URL — pros: preserves the license flag; cons: creates remote license content and deviates from source/remote behavior.
3. Human owner creates the private GitHub repo manually, then Codex runs `git remote add origin <repo-url>` — pros: avoids Codex guessing around `gh` behavior; cons: requires owner action outside the phase.
**Recommendation**: Option 1. It best matches §3.2's stated intent that the remote is empty because no content is pushed, while keeping the local `LICENSE` as source of truth.
