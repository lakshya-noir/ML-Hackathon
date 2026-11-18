# Nikhil and Lakshya: map inspection findings

Status: Codex reviewed generated static images and measured three specific arc comparisons while Nikhil was unavailable. Detailed findings are in `outputs/evaluation_1130/READ_FIRST.md` and `disagreement_details.json`. This is not Nikhil's personal review and does not validate all labels.

Recorded examples: trip 52 point 4 (classifier scoring disagreement), trip 56 point 214 (candidate cutoff and distance ties), trip 65 point 267 (nine closer arcs exclude the surrogate road from eight candidates). See `outputs/evaluation_1130/04_disagreement_closeups.png`.

For each of the three maps, record at least one concrete finding (including a normal-looking example if appropriate).

## Finding template

- Trip ID:
- Observation index / timestamp:
- Layers enabled:
- What we saw:
- Screenshot path:
- Possible explanation (hypothesis, not a fact):
- Needs code investigation, or documentation only:
- Reviewed by / time:

Check loops/intersections, parallel roads, opposing road directions, red disagreements and black candidate misses. Green and orange road segments can overlap; switch layers on/off to compare. Direction differences may require checking the underlying arc node order in the CSV/data.
