
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import jieba
import re
import io
import streamlit as st


def recommender(view_product, dictionary, tfidf, index, products):
    st.write("- **Convert search words into Sparse vectors**")
    view_product = view_product.lower().split()
    st.write("- **Convert search words into Sparse Vectors**")
    kw_vector = dictionary.doc2bow(view_product)
    st.write("View product's vector:")
    st.write(kw_vector)
    st.write("- **Similarity calculation**")
    sim = index[tfidf[kw_vector]]

    st.write("- **print result**")
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])

    df_result = pd.DataFrame({'id': list_id, 'score': list_score})

    # 5 highest scores
    five_highest_score = df_result.sort_values(by='score',
                                               ascending=False).head(6)
    st.write('Five highest score:')
    st.write(five_highest_score)
    st.write('Ids to list:')
    idTolist = list(five_highest_score['id'])
    st.write(idTolist)

    products_find = products[products.index.isin(idTolist)]
    results = products_find[['index', 'item_id', 'name']]
    results = pd.concat([results, five_highest_score],
                        axis=1).sort_values(by='score',
                                            ascending=False)
    return results


def loadContentBasedGenSim():

    st.markdown("""
        # Đọc dữ liệu
    """, unsafe_allow_html=True)

    products = pd.read_csv('Products.csv')
    reviews = pd.read_csv('Reviews.csv', lineterminator='\n')

    st.write(products.head(2))

    # st.write(df.info()) issue https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100/2
    buffer = io.StringIO()
    products.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.markdown("""
        # Underthesea
    """, unsafe_allow_html=True)
    #
    st.write("- **word_tokenize, pos_tag, sent_tokenize**")
    #
    #     link: https://github.com/undertheseanlp/underthesea

    products = products[products['name'].notnull()]

    products['name_description'] = products.name + products.description

    products = products[products['name_description'].notnull()]

    products['name_description_pre'] = products['name_description'].apply(
        lambda x: word_tokenize(x, format='text'))

    st.text(type(products))

    st.text(products.shape)

    products = products.reset_index()

    st.markdown("""
        # Gensim
    """, unsafe_allow_html=True)
    st.write(
        """
        - https://pypi.org/project/gensim/ -Là một thư viện Python chuyên xác định sự tương tự về ngữ nghĩa giữa hai tài liệu thông qua mô hình không gian vector và bộ công cụ mô hình hóa chủ đề.
        - Có thể xử lý kho dữ liệu văn bản lớn với sự trợ giúp của việc truyền dữ liệu hiệu quả và các thuật toán tăng cường
        - Tốc độ xử lý và tối ưu hóa việc sử dụng bộ nhớ tốt
        - Tuy nhiên, Gensim có ít tùy chọn tùy biến cho các function
    """
    )

    st.markdown("""
        #### Tham khảo:
    """, unsafe_allow_html=True)
    st.markdown(
        """
            - link https://www.tutorialspoint.com/gensim/index.htm
            - link https://www.machinelearningplus.com/nlp/gensim-tutorial/
        """
    )

    st.write("- **Tokenize(split) the sentences into words**")
    intro_products = [[text for text in x.split()]
                      for x in products.name_description_pre]

    st.text(len(intro_products))

    st.write(intro_products[:1])

    st.write("- **Stop words**")
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'

    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()

    stop_words = stop_words.split('\n')

    st.write("- **remove some special elements in texts**")
    intro_products_re = [[re.sub('[0-9]+', '', e) for e in text]
                         for text in intro_products]  # số
    intro_products_re = [[t.lower()
                          for t in text if not t in [
        '', ' ', ',', '.', '...', '-', ':',
        ';', '?', '%', '(', ')', '+', '/']]
        for text in intro_products_re]  # ký tự đặc biệt
    intro_products_re = [[t for t in text if not t in stop_words]
                         for text in intro_products_re]  # stopword

    st.write(intro_products_re[:1])

    st.write("- **Obtain the number of features based on dictionary:**")
    st.write("- **Use corpora.Dictionary**")
    dictionary = corpora.Dictionary(intro_products_re)

    st.write("- **List od features in dictionary**")
    st.text(dictionary.token2id)

    st.write(len(dictionary.token2id))

    st.write("- **Numbers of features (word) in dictionary**")
    feature_cnt = len(dictionary.token2id)
    st.write(feature_cnt)

    st.write("- **Obtain corpus based on dictionary (dense matrix)**")
    corpus = [dictionary.doc2bow(text) for text in intro_products_re]

    st.text(corpus[0])

    st.write("- **Use TF-IDF to process corpous, obtaning index**")
    tfidf = models.TfidfModel(corpus)
    st.write("- **Tính toán sự tương tự trong ma trận thưa thớt**")
    index = similarities.SparseMatrixSimilarity(
        tfidf[corpus], num_features=feature_cnt)

    st.write("- **When user choose one product**")
    st.write("- **Giả sử là chọn sản phẩm đầu tiên để xem, index=0**")
    product_ID = 10001355
    product = products[products.item_id == product_ID].head(1)

    st.text(type(product['name_description_pre']))

    st.write(product[['index', 'item_id', 'name_description_pre']])

    st.write("- **Sản phẩm đang xem**")
    name_description_pre = product['name_description_pre'].to_string(
        index=False)

    name_description_pre

    st.write("- **Dề xuất các sản phẩm đang xem**")

    results = recommender(name_description_pre,
                          dictionary, tfidf, index, products)
    st.write(results)

    st.write("- **Recommender 5 similarity products for the selected product**")
    st.write("- **Check and remove the selected product from the results**")
    results = results[results.item_id != product_ID]
    st.write(results)

    st.write(
        "- **Lưu lại các tham số dictionary, tfidf, index để có thể đọc ở bất cứ đâu**")
    st.text(type(dictionary))

    st.text(type(tfidf))

    st.text(type(index))
