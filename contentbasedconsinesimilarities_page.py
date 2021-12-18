
from wordcloud import WordCloud
import re
import jieba
from gensim import corpora, models, similarities
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import io
warnings.filterwarnings('ignore', category=FutureWarning)


def item(id, products):
    return products.loc[products['item_id'] == id]['name'].to_list()[0].split('-')[0]

    st.write("- **Thông tin sản phẩm gợi ý**")


def recommend(item_id, num, results, products):
    st.write('Recommending ' +
             str(num) +
             ' products similar to ' +
             item(item_id, products) + '...')
    st.write('* '*40)
    recs = results[item_id][:num]
    for rec in recs:
        print(rec[1])
        st.write('Recommended: product id:' +
                 str(rec[1]) + ', ' +
                 item(rec[1], products) + '(score:'
                 + str(rec[0]) + ')')


def get_product_text(item_id, num, results, products):
    rcmd_ids = [r[1] for r in results[item_id]] + [item_id]
    text = (products[products.item_id.isin(rcmd_ids)])
    return ' '.join(text.name + text.description)


def loadContentBasedConsineSimilarities():
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
    st.write("- link: https://github.com/undertheseanlp/underthesea")

    products = products[products['name'].notnull()]

    products['name_description'] = products.name + products.description

    products = products[products['name_description'].notnull()]

    products['name_description_pre'] = products['name_description'].apply(
        lambda x: word_tokenize(x, format='text'))

    st.text(type(products))

    st.text(products.shape)

    st.write(products.head(2))

    products = products.reset_index()

    st.write("- **Stop words**")
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'

    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()

    stop_words = stop_words.split('\n')

    st.write("- **TF - IFD**")
    tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words=stop_words)

    tfidf_matrix = tf.fit_transform(products.name_description_pre)

    st.markdown("""
        # Consine Similarities
    """, unsafe_allow_html=True)

    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    st.text(cosine_similarities)

    st.text(cosine_similarities.shape)

    st.write("- **với mỗi sản phẩm, lấy 10 sản phẩm tương quan nhất**")
    results = {}

    for idx, row in products.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-10:-1]
        similar_items = [(cosine_similarities[idx][i],
                          products['item_id'][i]) for i in similar_indices]
        results[row['item_id']] = similar_items[1:]

    st.write("- **các sản phẩm có tương quan với product_id = 38458616**")
    st.write(results[38458616])

    st.write("- **Lấy thông tin sản phẩm**")

    recommend(1059892, 5, results, products)

    st.markdown("""
        # Word cloud
    """, unsafe_allow_html=True)

    wordcloud_text = get_product_text(1059892, 5, results, products)

    wc = WordCloud(stopwords=stop_words).generate(wordcloud_text)
    fig = plt.figure()
    plt.imshow(wc)
    st.pyplot(fig, clear_figure=True)
    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Do sử dụng tên và phần mô tả để so sánh đặc tính của sản phẩm, độ chính xác của hệ thống không bị ảnh hưởng khi tên và phần mô tả không chính xác hoặc thiếu thông tin.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""
        # Lưu kết quả
    """, unsafe_allow_html=True)

    info = []
    for p_id, v in results.items():
        for item in v:
            info.append({
                'product_id': p_id,
                'rcmd_product_id': item[1],
                'score': item[0]
            })
    content_based_df = pd.DataFrame(info)

    content_based_df.to_csv('CB.csv')
