import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

from product_search.config import INDEXES_DIR, ROOT, catalog_path
from product_search.search.explanations import product_evidence
from product_search.search.indexes import catalog_fingerprint, load_or_build
from product_search.search.keyword import KeywordSearch
from product_search.utils.data_loader import ProductDataLoader
from product_search.utils.images import load_image

load_dotenv(ROOT / ".env")
DATA_PATH = catalog_path()
MODES = ["Semantic", "Semantic + keywords", "Keyword"]

st.set_page_config(page_title="Semantic Product Search", layout="wide")
st.html(Path(__file__).with_name("style.css"))


@st.cache_resource
def catalog(fingerprint):
    loader = ProductDataLoader(DATA_PATH)
    df = loader.preprocess_data()
    return df, loader, KeywordSearch(df["text_for_embedding"].tolist())


@st.cache_resource
def text_model():
    from product_search.models.text_embedding import TextEmbeddingGenerator
    return TextEmbeddingGenerator()


@st.cache_resource
def image_model():
    try:
        from product_search.models.image_embedding import ImageEmbeddingGenerator
    except ImportError as exc:
        raise ImportError("Image search needs the optional dependencies. Run `uv sync --extra image`.") from exc
    return ImageEmbeddingGenerator()


@st.cache_resource
def search_index(kind, fingerprint):
    if kind == "text":
        from product_search.models.text_embedding import MODEL_ID
        factory = text_model
    else:
        from product_search.models.image_embedding import MODEL_ID
        factory = image_model
    df, _, _ = catalog(fingerprint)
    return load_or_build(DATA_PATH, df, INDEXES_DIR, kind, MODEL_ID, factory)


@st.cache_data(max_entries=100)
def catalog_image(url):
    try:
        return load_image(url, INDEXES_DIR / "images")
    except Exception:
        return None


def show_product(product, rank=None, query=""):
    image = catalog_image(product["image_url"])
    if image is not None:
        st.image(ImageOps.pad(image, (600, 760), color="#ECEBE5"), width="stretch")
    else:
        st.markdown('<div class="image-missing">Image unavailable</div>', unsafe_allow_html=True)
    name = product["name"].capitalize()
    st.markdown(f"#### {f'{rank}. ' if rank else ''}{name}")
    with st.expander("View details"):
        if query:
            label, excerpts = product_evidence(product, query)
            st.caption(label)
            for excerpt in excerpts:
                st.text(excerpt)
            st.divider()
        st.text(product["details"] or "No description is available.")
        st.link_button("Original listing", product["link"])


def show_results(products, query=""):
    for start in range(0, len(products), 3):
        for offset, (column, product) in enumerate(zip(st.columns(3), products[start:start + 3])):
            with column:
                show_product(product, start + offset + 1, query)


def main():
    fingerprint = catalog_fingerprint(DATA_PATH)
    df, loader, keyword = catalog(fingerprint)
    st.title("Semantic Product Search")
    st.caption(f"Find a jacket by description or photo · {len(df)} pieces")
    text_tab, image_tab = st.tabs(["Describe it", "Use a photo"])
    with text_tab:
        with st.form("text_search", border=False):
            field, options, action = st.columns([6, 2, 1.5], vertical_alignment="bottom")
            with field:
                query = st.text_input("Describe a jacket", key="query", placeholder="e.g. a cropped denim jacket")
            with options:
                with st.popover("Search options", width="stretch"):
                    mode = st.radio("Search method", MODES)
            with action:
                submitted = st.form_submit_button("Search", type="primary", width="stretch")
        if submitted:
            st.session_state.pop("text_results", None)
            if not query.strip():
                st.warning("Describe a product to start searching.")
            else:
                try:
                    if mode == "Keyword":
                        start = time.perf_counter()
                        ids = keyword.search(query)
                    else:
                        with st.spinner("Preparing semantic search on first use…"):
                            db, _ = search_index("text", fingerprint)
                            model = text_model()
                        start = time.perf_counter()
                        with st.spinner("Finding matches…"):
                            embedding = model.generate_text_embedding(query)
                            _, indices = db.search_by_text(embedding, query_text=query, keyword_boost=mode == "Semantic + keywords")
                            ids = indices[0].tolist()
                    st.session_state.text_results = {"ids": ids, "query": query, "mode": mode,
                                                     "ms": (time.perf_counter() - start) * 1000}
                except Exception as exc:
                    st.error(f"Text search could not finish. {exc}")
        results = st.session_state.get("text_results")
        if results:
            st.subheader(f"Results for “{results['query']}”")
            st.caption(f"{len(results['ids'])} pieces · {results['mode']}")
            if not results["ids"]:
                st.info("No matching words found. Try a different description or search method.")
            show_results(loader.get_product_details(results["ids"]), results["query"])
    with image_tab:
        with st.form("image_search", border=False):
            uploaded = st.file_uploader("Reference photo", type=["jpg", "jpeg", "png"], max_upload_size=15)
            submitted_image = st.form_submit_button("Find similar pieces", type="primary")
        if submitted_image:
            st.session_state.pop("image_results", None)
            if uploaded is None:
                st.warning("Choose a reference photo first.")
            else:
                try:
                    with Image.open(uploaded) as source:
                        reference = ImageOps.exif_transpose(source).convert("RGB")
                    with st.spinner("Preparing image search on first use…"):
                        db, report = search_index("image", fingerprint)
                        model = image_model()
                    start = time.perf_counter()
                    embedding = model.generate_embedding_from_pil_image(reference)
                    _, ids = db.search_by_image(embedding)
                    st.session_state.image_results = {"ids": ids[0].tolist(), "reference": reference,
                                                      "ms": (time.perf_counter() - start) * 1000, "report": report}
                except Exception as exc:
                    st.error(f"Image search could not finish. {exc}")
        results = st.session_state.get("image_results")
        if results:
            st.image(results["reference"], width=160, caption="Your reference")
            st.subheader("Visually similar pieces")
            st.caption(f"{results['report']['indexed']} of {len(df)} catalog photos searchable")
            show_results(loader.get_product_details(results["ids"]))
    st.divider()
    st.caption("Taiwu Chen · Archival ZARA catalog · Availability not tracked · Not affiliated with ZARA")


if __name__ == "__main__":
    main()
