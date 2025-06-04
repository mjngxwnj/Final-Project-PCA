import numpy as np
import pandas as pd
from src.Vectorizer.vectorizer import FeatureVectorizer



#image samle
image_sample = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)

#table sample
table_sample = pd.DataFrame({
    'Col1': np.random.rand(10),
    'Col2': np.random.rand(10),
    'Col3': np.random.rand(10)
})

# text sample
text_sample = """This is a sample text. It contains multiple sentences. Each sentence will be tokenized.\
                    This is another sentence.\
                    This is the last sentence. \
                    This is a sample text for vectorization."""

data_info_1 = {
    'type': 'text',
    'content': text_sample,
    'metadata': {}
}

data_info_2 = {
    'type': 'image',
    'content': image_sample,
    'metadata': {}
}

data_info_3 = {
    'type': 'table',
    'content': table_sample,
    'metadata': {}
}



vectorizer_1 = FeatureVectorizer(data_info_1).vectorize()
vectorizer_2 = FeatureVectorizer(data_info_2).vectorize()
vectorizer_3 = FeatureVectorizer(data_info_3).vectorize()

print(f"text after vectorization: {vectorizer_1}")
print(f"image after vectorization: {vectorizer_2}")
print(f"table after vectorization: {vectorizer_3}")


# Output
"""
text after vectorization: 
[[0.         0.         0.08109302 0.         0.21972246 0.21972246
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.21972246 0.
  0.08109302]
 [0.         0.         0.         0.44793987 0.         0.
  0.44793987 0.         0.44793987 0.44793987 0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.13862944 0.         0.         0.         0.         0.
  0.         0.35835189 0.         0.         0.35835189 0.
  0.35835189 0.         0.         0.         0.         0.35835189
  0.        ]
 [0.1732868  0.         0.10136628 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.44793987 0.         0.
  0.10136628]
 [0.13862944 0.35835189 0.08109302 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.35835189 0.         0.         0.
  0.08109302]
 [0.         0.         0.05792359 0.         0.15694461 0.15694461
  0.         0.         0.         0.         0.         0.25596564
  0.         0.25596564 0.         0.         0.15694461 0.
  0.05792359]]

image after vectorization: [180.   3. 132. ... 186. 184.  74.]

table after vectorization: 
[[0.40958321 1.         0.57221585]
 [1.         0.         1.        ]
 [0.43602945 0.81065966 0.35661498]
 [0.5294588  0.85135712 0.28451989]
 [0.86907137 0.39351025 0.97602873]
 [0.33602462 0.51466692 0.63705656]
 [0.24204313 0.76895405 0.        ]
 [0.63363129 0.64288431 0.12198617]
 [0.17424785 0.43491333 0.49546231]
 [0.         0.63450677 0.47280603]]

"""

