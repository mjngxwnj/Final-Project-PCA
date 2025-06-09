import os
import numpy as np
import pandas as pd
from src.Vectorizer.vectorizer import FeatureVectorizer
from src.Reader.reader_agent import DataExpander


# Path to the files
txt_path = os.path.join("data_test_for_reader", "report.txt")
image_path = os.path.join("data_test_for_reader", "cat.jpg")
table_path = os.path.join("data_test_for_reader", "data.csv")



def main():
    # 0. Call modules
    reader = DataExpander()
    vectorizer = FeatureVectorizer()

    # 1. Read files
    text_data = reader.expand(txt_path)
    image_data = reader.expand(image_path)
    table_data = reader.expand(table_path)

    # 2. Vectorize data
    text_vector = vectorizer.vectorize(text_data)
    image_vector = vectorizer.vectorize(image_data)
    table_vector = vectorizer.vectorize(table_data)

    # 3. Print results
    print("text after vectorization: \n", text_vector)
    print("image after vectorization:", image_vector)
    print("table after vectorization: \n", table_vector)

if __name__ == "__main__":
    main()


# Output
"""
text after vectorization: 
 [[0.         0.         0.         ... 0.         0.         0.        ]
 [0.01455649 0.         0.         ... 0.         0.         0.        ] 
 [0.02375427 0.         0.         ... 0.         0.         0.        ] 
 ...
 [0.         0.         0.         ... 0.         0.         0.        ] 
 [0.01162171 0.         0.         ... 0.         0.         0.        ] 
 [0.         0.         0.         ... 0.         0.         0.        ]]
table after vectorization:
 [[0.         0.         0.         0.        ]
 [0.18181818 0.03225806 0.5        0.0248911 ]
 [0.27272727 0.09677419 1.         0.04667082]
 [0.18181818 0.03225806 0.15       0.06036092]
 [0.18181818 0.03225806 1.         0.0746733 ]
 [0.45454545 0.22580645 1.         0.08867455]
 [0.27272727 0.09677419 0.5        0.07156192]
 [0.45454545 0.22580645 1.         0.10267579]
 [0.63636364 0.22580645 0.75       0.14779091]
 [0.63636364 0.48387097 1.         0.25668948]
 [0.63636364 0.22580645 0.75       0.26913503]
 [0.63636364 0.22580645 0.55       0.3282514 ]
 [0.63636364 0.48387097 0.75       0.36714375]
 [0.63636364 0.22580645 0.75       0.34038581]
 [0.63636364 0.22580645 0.75       0.42470442]
 [1.         0.22580645 1.         0.44026136]
 [1.         0.48387097 0.75       0.53329185]
 [0.63636364 0.48387097 0.15       0.54449284]
 [0.63636364 0.48387097 0.15       0.5955196 ]
 [0.63636364 0.48387097 0.9        0.65805849]
 [1.         0.48387097 0.75       0.65805849]
 [1.         0.48387097 1.         0.86963286]
 [1.         1.         1.         0.87554449]
 [1.         1.         0.7        1.        ]]

"""