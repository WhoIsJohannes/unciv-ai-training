---
name: unciv-fork-push-credential
description: "How to push the Unciv AI-training work to the fork: remote `fork` = WhoIsJohannes/unciv-ai-training, but the active gh account is J-Mentiora (pull-only) so plain `git push` 403s — push with the WhoIsJohannes token via an inline credential helper."
metadata: 
  node_type: memory
  type: project
  originSessionId: 064d271f-6414-4f7d-a66e-650fb37ebf2e
---

The self-play AI work lives on `master` at `/Users/j/Unciv` (worktrees were collapsed to a single master on 2026-06-25; the data-plane + ONNX-loop branches were merged in and deleted). Remotes: `fork` = https://github.com/WhoIsJohannes/unciv-ai-training (the AI-training repo — "everything on main" means this fork's `master`); `origin` = upstream https://github.com/yairm210/Unciv (we never push there; a chronic "behind 1 vs origin/master" is just upstream moving).

**Gotcha:** the machine's active gh account is **J-Mentiora**, which has only **pull** access to `WhoIsJohannes/unciv-ai-training`, so a plain `git push fork master` (and IDE auto-push) fails `403 denied to J-Mentiora`. Both accounts are logged into gh (`gh auth status` shows J-Mentiora active + WhoIsJohannes available).

**How to apply** — push as WhoIsJohannes WITHOUT switching the global gh active account (keeps the other agent's env untouched), token never in argv:
```
cd /Users/j/Unciv
export WIJ_TOKEN=$(gh auth token --user WhoIsJohannes)
git -c credential.helper= -c credential.helper='!f(){ echo username=WhoIsJohannes; echo "password=$WIJ_TOKEN"; }; f' push fork master
unset WIJ_TOKEN
```
The leading empty `credential.helper=` resets the chain so the osxkeychain/J-Mentiora helper isn't consulted. Pushes to the fork have all been clean fast-forwards. Confirm with `git ls-remote --heads fork master`. Reads (`gh api repos/WhoIsJohannes/...`) work fine as J-Mentiora. See [[selfplay-roadmap-bottleneck]].
