---
name: github-repo-link-and-push
description: "Link a local folder to a GitHub repository and push all contents safely. Use when setting a remote, committing pending changes, syncing branch upstream, and verifying push completion without destructive git commands."
argument-hint: "GitHub repo URL and push scope (all changes, clean only, or selected paths)"
user-invocable: true
disable-model-invocation: false
---

# GitHub Repo Link And Push

## What This Skill Produces
- A local git repo linked to the requested GitHub remote.
- A committed snapshot based on user-selected push scope.
- An upstream branch pushed and verified.
- A short report of what was pushed and any follow-up actions.

## When To Use
- You have a project folder and want it published to GitHub.
- You need to update or verify origin remote URL.
- You want a safe, non-destructive push workflow with explicit user confirmation.

## Inputs
- GitHub repository URL.
- Push scope:
  - all current changes (tracked + untracked), or
  - clean committed files only, or
  - selected paths only.
- Target branch (default to current branch unless user specifies another).

## Procedure
1. Validate repository state.
- Run:
  - git rev-parse --is-inside-work-tree
  - git branch --show-current
  - git remote -v
  - git status --short
  - git rev-list --count HEAD
- If not a git repo, run git init and re-check branch.

2. Confirm safety-critical decision points.
- If working tree is dirty, ask user what scope to push.
- If remote already exists and differs from requested URL, ask whether to replace with:
  - git remote set-url origin <url>
- If no remote exists, add:
  - git remote add origin <url>

3. Prepare commit according to scope.
- All changes:
  - git add -A
- Selected paths:
  - git add <path1> <path2> ...
- Clean only:
  - skip staging/committing new local edits.

4. Ensure there is a commit to push.
- If staged changes exist, commit:
  - git commit -m "Sync workspace snapshot"
- If no commits exist yet, create initial commit (after staging chosen scope).

5. Push and set upstream.
- Push current branch:
  - git push -u origin <current-branch>
- If push is rejected due to non-fast-forward, stop and ask user whether to pull/rebase or use another branch.

6. Verify result.
- Run:
  - git status --short
  - git log --oneline -n 3
  - git remote -v
- Report remote URL, branch, last commit hash/message, and whether working tree is clean.

## Decision Points And Branching
- Dirty tree detected:
  - branch to user choice: all, clean-only, or selected paths.
- Remote mismatch:
  - branch to keep existing remote or replace URL.
- No commits:
  - branch to initial commit path.
- Push rejection:
  - branch to conflict resolution flow (pull/rebase/new branch) with user confirmation.

## Quality Criteria
- No destructive commands used (no reset --hard, no checkout --, no force push unless explicitly approved).
- Remote URL exactly matches user-provided URL.
- Upstream set on pushed branch.
- Push outcome verified with post-push checks.
- User receives a concise summary of what changed and what was pushed.

## Completion Checklist
- [ ] Repository status checked.
- [ ] Remote verified/updated.
- [ ] Push scope confirmed.
- [ ] Commit created if needed.
- [ ] Branch pushed with upstream.
- [ ] Verification checks completed.
- [ ] Results summarized to user.
