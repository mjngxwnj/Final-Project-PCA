from src.Vectorizer.vectorizer import FeatureVectorizer


sample_text = """Machine learning is a method of data analysis that automates analytical model building. \
                It is a branch of artificial intelligence based on the idea that systems can learn from \
                data, identify patterns, and make decisions with minimal human intervention. \
                Machine learning is used in a wide range of applications, such as email filtering and \
                computer vision."""


data_info = {
    'type': 'text',
    'content': sample_text,
    'metadata': {}
}

vectorizer = FeatureVectorizer(data_info)
print(vectorizer.text_vectorizer())




# Dữ liệu: có thể 1 hoặc nhiều câu/documents
