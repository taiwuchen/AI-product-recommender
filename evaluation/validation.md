# Verification record

Verified locally on September 7, 2026, using Python 3.12 and the dependencies in `requirements.lock`.

- **Pass:** 13 unit regression checks (`python -m unittest discover -s tests -v`).
- **Pass:** Streamlit AppTest checks for keyword ranking, results surviving example-button reruns, unknown-query empty state, and blank-query validation.
- **Pass:** Headless Playwright checks for keyword search and image upload/search. The reference faux suede bomber was returned first; repeating image search succeeded.
- **Pass:** 390 × 844 mobile layout inspection, with no horizontal document overflow. Desktop and mobile screenshots are under the ignored `output/playwright/` directory.
- **Pass:** Final browser session reported zero console errors or warnings.
- **Pass:** Three real CLIP inference runs across successive worker threads returned the correct product first. Run `python -m evaluation.image_smoke`; see `image-smoke.json`.
- **Partial:** The keyword evaluation ran on all 24 provisional queries. Live semantic and boosted evaluation could not run because Google rejected the saved credentials (`invalid_grant`). The Google client has mocked contract tests, not a successful live verification in this session.

The image index included 55 of 66 catalog images. Eleven archival image URLs failed; their product IDs are recorded in `image-smoke.json`. The exact-image smoke check verifies retrieval wiring, not image relevance on unseen photos.

A native PyTorch convolution crash was reproduced when reusing the model across worker threads with FAISS loaded. A fixed one-thread CPU inference budget resolved the reproducer and the repeated browser flow on this Mac. This was verified after restarting with the final dependency lock.
