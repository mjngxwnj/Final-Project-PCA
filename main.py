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
docx_path  = os.path.join("data_test_for_reader", "report.docx")


def main():
    # 0. Call modules
    reader = DataExpander()
    vectorizer = FeatureVectorizer()

    # 1. Read files
    #text_data = dict(reader.expand(txt_path)[0])
    #image_data = reader.expand(image_path)
    #table_data = reader.expand(table_path)
    #pdf_data = reader.expand(pdf_path)
    docx_data = reader.expand(docx_path)
    print(docx_data)

    # 2. Vectorize data
    #text_vector = vectorizer.vectorize(text_data)
    #print(text_vector.shape)
    #image_vector = vectorizer.vectorize(image_data)
    #table_vector = vectorizer.vectorize(table_data)
    #pdf_vector = vectorizer.vectorize(pdf_data)
    docx_vector = vectorizer.vectorize(docx_data)

    # 3. Print results
    #print("text after vectorization: \n", text_vector)
    #print("image after vectorization:", image_vector)
    #print("table after vectorization: \n", table_vector)
    #print("pdf after vectorization: \n", pdf_vector)
    print("docx after vectorization: \n", docx_vector)

if __name__ == "__main__":
    main()

"""
[{'type': 'text', 'content': 'Cái này là ảnh con mèo nè\nCòn về văn bản hả. Mọi 
người còn trông chờ gì hơn ngoài dòng này >>>>>>', 'meta': {'original_filename':
 'report.docx', 'original_extension': '.docx'}}, {'type': 'table', 'content':   
Cái này là bảng nè  Label1                Label2     Label3 Label4
0               Row1   Sdvsd  Dsvadf585623^$**^$%$  Dsvsdv xc    123
1               Row2  Sdvsdv             524456354      Sdfsd    52r
2                R\n  dsvsdv             Sdvsdvsdv       &*$@  dfbsf
3               \n\n                                                , 'meta': {'
original_filename': 'report.docx', 'original_extension': '.docx', 'source': 'tab
le_1'}}, {'type': 'image', 'content': array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       ...,

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], shape=(1430, 1430, 3), dtype=uint8), 'meta': {'original_fil
ename': 'report.docx', 'original_extension': '.docx', 'source': 'image_1', 'size
_pixels': (1430, 1430, 3)}}]
docx after vectorization: 
 [array([[0.05776227, 0.05776227, 0.        , 0.        , 0.05776227,
        0.        , 0.        , 0.05776227, 0.05776227, 0.05776227,
        0.        , 0.        , 0.05776227, 0.05776227, 0.        ,
        0.05776227, 0.        , 0.05776227, 0.        , 0.        ],
       [0.        , 0.        , 0.06931472, 0.06931472, 0.        ,
        0.        , 0.06931472, 0.        , 0.        , 0.        ,
        0.06931472, 0.06931472, 0.        , 0.        , 0.06931472,
        0.        , 0.06931472, 0.        , 0.        , 0.06931472]]), array([],
 shape=(4, 0), dtype=float64), array([0., 0., 0., ..., 0., 0., 0.], shape=(20449
00,))]
"""
