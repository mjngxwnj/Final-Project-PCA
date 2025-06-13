import numpy as np
import pandas as pd
from tf_idf_module import TfidfVectorizer
from PIL import Image


class FeatureVectorizer:
    """
    A class to vectorize text, images, and tables into a unified feature vector.
    """

    def __init__(self):
        self.text_vectorizer = TfidfVectorizer()
        self._is_text_vectorizer_fitted = False

    def fit_texts(self, full_text: str):
        """
        Fit the TF-IDF vectorizer using lines from a single large text string.
        """
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("No valid lines to fit.")
        self.text_vectorizer.fit(lines)
        self._is_text_vectorizer_fitted = True

    def _text_vectorizer(self, text_input: str) -> np.ndarray:
        """
        Transform a single text string by splitting into lines and applying TF-IDF.
        Returns an array of shape (num_lines, num_vocab)
        """
        if not self._is_text_vectorizer_fitted:
            raise RuntimeError("Text vectorizer must be fitted first.")

        lines = [line.strip() for line in text_input.splitlines() if line.strip()]
        if not lines:
            raise ValueError("No valid lines to vectorize.")

        line_vectors = []
        for line in lines:
            vec = self.text_vectorizer.transform(line)
            vec_array = vec.toarray()[0] if hasattr(vec, "toarray") else np.asarray(vec)[0]
            line_vectors.append(vec_array)

        return np.vstack(line_vectors)  # shape: (num_lines, num_vocab)

    def _image_vectorizer(self, image: Image.Image) -> np.ndarray:
        if not isinstance(image, Image.Image):
            raise TypeError("Each image must be a PIL.Image.Image object.")

        image = image.convert("L").resize((64, 64))  # Grayscale
        image_array = np.array(image)

        if image_array.shape != (64, 64):
            raise ValueError("Image must be resized to (64, 64).")

        return image_array.flatten()  # shape (4096,)

    def _table_vectorizer(self, table_data: pd.DataFrame) -> np.ndarray:
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("Each table must be a pandas DataFrame.")

        array_data = table_data.to_numpy(dtype=np.float32)

        min_vals = np.min(array_data, axis=0)
        max_vals = np.max(array_data, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

        normalized = (array_data - min_vals) / denom
        return normalized.flatten()

    def vectorize(self, data_info: dict) -> np.ndarray:
        if not isinstance(data_info, dict):
            raise TypeError("data_info must be a dict.")

        for key in ['type', 'content', 'meta']:
            if key not in data_info:
                raise ValueError(f"data_info must contain '{key}' key.")

        content_types = data_info['type']
        content = data_info['content']
        feature_vectors = []

        if isinstance(content, dict):
            if "text" in content_types and content.get("text"):
                vec = self._text_vectorizer(content["text"])
                feature_vectors.append(vec)

            if "images" in content_types and content.get("images"):
                for img in content["images"]:
                    vec = self._image_vectorizer(img)
                    feature_vectors.append(vec)

            if "tables" in content_types and content.get("tables"):
                for tbl in content["tables"]:
                    try:
                        vec = self._table_vectorizer(tbl)
                        feature_vectors.append(vec)
                    except Exception as e:
                        print("Skipping table due to error:", e)
        else:
            if "text" in content_types and isinstance(content, str):
                vec = self._text_vectorizer(content)
                feature_vectors.append(vec)

            elif "images" in content_types and isinstance(content, Image.Image):
                vec = self._image_vectorizer(content)
                feature_vectors.append(vec)

            elif "tables" in content_types and isinstance(content, pd.DataFrame):
                try:
                    vec = self._table_vectorizer(content)
                    feature_vectors.append(vec)
                except Exception as e:
                    print("Skipping table due to error:", e)

        if not feature_vectors:
            raise ValueError("No valid content to vectorize.")

        # Flatten all feature vectors and concatenate
        flat_vectors = []
        for vec in feature_vectors:
            if vec.ndim == 1:
                flat_vectors.append(vec.reshape(1, -1))  # shape (1, D)
            else:
                flat_vectors.append(vec)  # keep shape (num_lines, D)
        return np.vstack(flat_vectors)
