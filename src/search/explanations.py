import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def product_evidence(product, query=""):
    terms = set(re.findall(r"[a-z]+", query.lower())) - ENGLISH_STOP_WORDS
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", product["details"]) if part.strip()]
    candidates = [product["name"], *sentences]
    ranked = sorted(candidates, key=lambda value: -len(terms & set(re.findall(r"[a-z]+", value.lower()))))
    matches = [value for value in ranked if terms & set(re.findall(r"[a-z]+", value.lower()))]
    if matches:
        return "Matching words in the listing", matches[:2]
    if not sentences:
        return "Listing details", ["No description is available for this product."]
    return "Listing details", sentences[:2]
