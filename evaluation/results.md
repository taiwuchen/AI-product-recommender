# Search evaluation

Run: 2026-09-07T22:09:19.443760+00:00

Provisional labels authored by the coding assistant from catalog titles and descriptions before running retrieval. Not independently human-reviewed; this is a small diagnostic set, not proof of general search quality.

66 products; 24 queries. See the JSON report for every ranked result.

| Method | P@5 | Recall@5 | nDCG@5 | MRR@5 | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| keyword | 0.283 | 0.847 | 0.773 | 0.758 | 0.819 | 2.723 |
| semantic | 0.308 | 0.964 | 0.900 | 0.887 | 431.844 | 500.005 |
| boosted | 0.317 | 0.978 | 0.892 | 0.868 | 431.876 | 500.051 |

Search time includes query embedding and retrieval, excluding setup, image downloads, and rendering. Semantic and boosted modes share the same measured query embedding cost. One timed pass; latency is illustrative, not a load test.

P@5 always divides by five, even when fewer than five relevant products exist. Recall and nDCG help interpret those cases. Unlisted products count as nonrelevant under the provisional labels.

## Lowest-ranked cases

### keyword

- **a jacket that shines back when light hits it** — nDCG@5 0.000. Top three: LIGHT PADDED JACKET (not labeled relevant); LIGHT PADDED JACKET (not labeled relevant); FAUX SUEDE JACKET (not labeled relevant).
- **a short jean jacket** — nDCG@5 0.000. Top three: FAUX SUEDE JACKET (not labeled relevant); FAUX LEATHER JACKET (not labeled relevant); LIGHT PADDED JACKET (not labeled relevant).
- **a coat with two rows of buttons** — nDCG@5 0.000. Top three: TECHNICAL TRENCH COAT (not labeled relevant); COTTON JACKET (not labeled relevant); 100% WOOL COAT - LIMITED EDITION (not labeled relevant).

### semantic

- **a lightweight jacket with a hood** — nDCG@5 0.202. Top three: LIGHTWEIGHT BOMBER JACKET (not labeled relevant); UTILITY JACKET WITH POCKETS (not labeled relevant); HOODED TECHNICAL JACKET (not labeled relevant).
- **a jacket with a corduroy collar** — nDCG@5 0.387. Top three: COTTON JACKET (not labeled relevant); JACKET WITH CONTRAST TOPSTITCHING (not labeled relevant); FAUX SUEDE JACKET (not labeled relevant).
- **a sleeveless linen top** — nDCG@5 0.500. Top three: LINEN - COTTON OVERSHIRT (not labeled relevant); 100% LINEN JACKET (not labeled relevant); 100% LINEN WAISTCOAT (relevant).

### boosted

- **a lightweight jacket with a hood** — nDCG@5 0.416. Top three: UTILITY JACKET WITH POCKETS (not labeled relevant); LIGHTWEIGHT BOMBER JACKET (not labeled relevant); REFLECTIVE EFFECT JACKET (relevant).
- **a sleeveless linen top** — nDCG@5 0.500. Top three: LINEN - COTTON OVERSHIRT (not labeled relevant); 100% LINEN JACKET (not labeled relevant); 100% LINEN WAISTCOAT (relevant).
- **a jacket with a corduroy collar** — nDCG@5 0.500. Top three: COTTON JACKET (not labeled relevant); JACKET WITH CONTRAST TOPSTITCHING (not labeled relevant); RUBBERISED JACKET - LIMITED EDITION (relevant).
