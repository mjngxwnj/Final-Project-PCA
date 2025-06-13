import os
import numpy as np
import pandas as pd
from src.Vectorizer.vectorizer import FeatureVectorizer
from src.Reader.reader_agent import DataExpander

# Path to the files
txt_path   = os.path.join("data_test_for_reader", "report.txt")
image_path = os.path.join("data_test_for_reader", "cat.jpg")
table_path = os.path.join("data_test_for_reader", "Network.csv")
pdf_path   = os.path.join("data_test_for_reader", "test_pdf.pdf")


def main():
    # 0. Call modules
    reader = DataExpander()
    vectorizer = FeatureVectorizer()

    # 1. Read files
    #text_data = dict(reader.expand(txt_path)[0])
    #image_data = reader.expand(image_path)
    table_data = reader.expand(table_path)
    #pdf_data = reader.expand(pdf_path)

    # 2. Vectorize data
    #text_vector = vectorizer.vectorize(text_data)
    #print(text_vector.shape)
    #image_vector = vectorizer.vectorize(image_data)
    table_vector = vectorizer.vectorize(table_data)
    #pdf_vector = vectorizer.vectorize(pdf_data)


    # 3. Print results
    #print("text after vectorization: \n", text_vector)
    #print("image after vectorization:", image_vector)
    print("table after vectorization: \n", table_vector)
    #print("pdf after vectorization: \n", pdf_vector)

if __name__ == "__main__":
    main()

