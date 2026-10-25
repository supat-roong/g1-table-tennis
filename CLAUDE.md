# CLAUDE.md

## Commit Style

- Do NOT include `Co-Authored-By: Claude` or any AI co-author lines in commit messages.
- Use atomic commits: each commit should represent one logical change only.
- Commit frequently — after each meaningful, working unit of change.
- Commit messages must follow Conventional Commits format: `<type>(<scope>): <subject>` (e.g. `feat(cli): add --debug flag`, `fix(pipeline): handle empty image list`). Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`.
- Only commit to the `main` branch. 
- Do NOT push or open PRs at the end of a session unless explicitly instructed.