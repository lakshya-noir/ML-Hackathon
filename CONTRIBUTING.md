# Contributions and Git identity

Our team consists of Lakshya, Nakul, and Nikhil. Lakshya handles data and model work, final integration, and UI delivery. Nakul manages the directed graph, HMM/Viterbi, and backend. Nikhil supports data analysis, review, and verification. These responsibilities reflect our current division of labor, and they do not alter historical authorship.

Before tackling any new code, each of us configures our local Git identity:

```bash
git config --local user.name "YOUR ACTUAL NAME"
git config --local user.email "YOUR VERIFIED GITHUB EMAIL OR GITHUB NOREPLY EMAIL"
git var GIT_AUTHOR_IDENT
```

We commit all real changes under our respective authors. We preserve historical provenance, meaning that copying a file or pushing it to a remote branch does not establish original authorship. A fresh repository also does not automatically fix past email associations, so we always coordinate any identity corrections with the actual author before rewriting shared history.
