# Contributions and Git identity

Current team: Lakshya, Nakul and Nikhil. Lakshya handles data/model work, final integration and UI delivery; Nakul handles the directed graph, HMM/Viterbi and backend; Nikhil supports data analysis, review and verification. Responsibilities describe current ownership and do not rewrite historical authorship.

Codex and Claude Code assisted implementation, debugging and documentation. Members are responsible for reviewing and understanding the submission.

Before genuine new work, each member configures their own verified Git identity:

```bash
git config --local user.name "YOUR ACTUAL NAME"
git config --local user.email "YOUR VERIFIED GITHUB EMAIL OR GITHUB NOREPLY EMAIL"
git var GIT_AUTHOR_IDENT
```

Review and commit actual changes under their real author. Preserve historical provenance; copying a file or being responsible for pushing it does not establish original authorship. A new repository does not automatically correct historical email associations. Coordinate any verified identity correction with the actual author before rewriting shared history.
