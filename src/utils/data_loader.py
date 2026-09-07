import pandas as pd


class ProductDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def preprocess_data(self):
        df = pd.read_csv(self.data_path).fillna("")
        required = {"product_name", "link", "product_images", "details"}
        if not required.issubset(df.columns) or df.empty:
            raise ValueError("Catalog must contain products with name, link, image, and details columns.")
        if not df["product_name"].str.strip().all() or not df["link"].is_unique:
            raise ValueError("Products need nonempty names and unique links.")
        df["image_url"] = df["product_images"].str.strip()
        df["details"] = df["details"].str.replace("View more", "", regex=False).str.strip()
        df["text_for_embedding"] = df["product_name"] + ". " + df["details"]
        self.df = df.reset_index(drop=True)
        return self.df

    def get_product_details(self, product_indices):
        if self.df is None:
            self.preprocess_data()
        products = []
        for idx in product_indices:
            if not 0 <= idx < len(self.df):
                raise ValueError(f"Unknown product ID: {idx}")
            row = self.df.iloc[idx]
            products.append({"id": int(idx), "name": row["product_name"], "details": row["details"],
                             "image_url": row["image_url"], "link": row["link"]})
        return products
