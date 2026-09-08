# Contributing

Bug reports, questions, and pull requests are welcome.

## Set up

```bash
uv sync --extra image
uv run streamlit run product_search/app/streamlit_app.py
```

Semantic search needs an OpenRouter key in `.env`; keyword search and the tests do not.

## Before opening a pull request

Run the tests. They run offline and mock every network call.

```bash
uv run python -m unittest discover -s tests -v
```

Keep changes small and focused. If you change retrieval behavior, rerun the evaluation with `uv run python -m evaluation.run` and include the updated report so the numbers in the README stay truthful.

## Bring your own catalog

The app reads any CSV with the columns `product_name`, `link`, `product_images`, and `details`. Point `CATALOG_PATH` at it. The relevance labels in `evaluation/queries.json` belong to the bundled ZARA catalog, so write your own query set before evaluating another dataset.
