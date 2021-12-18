import warnings
import pandas_profiling as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import io
warnings.filterwarnings('ignore', category=FutureWarning)


def load_preprocessing():
    st.title("Preprocessing")
    st.write("### 1. Product")

    products = pd.read_csv('ProductRaw.csv')

    st.write(products.head())

    st.write('- *Số lượng sản phẩm*')
    st.write('Số lượng sản phẩm:', products.shape[0])

    st.write('- *Kiểu dữ liệu*')
    # st.write(df.info()) issue https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100/2
    buffer = io.StringIO()
    products.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write('- *Bỏ dữ liệu trùng*')
    st.write('Trước khi droping, records =', products.shape[0])
    products.drop_duplicates(inplace=True)
    st.write('Sau khi droping, records =', products.shape[0])

    st.write('- *Kiểm tra dữ liệu null*')
    st.text(products.isnull().any())

    st.write('- *Lưu kết quả*')
    products.to_csv('Products.csv', index=False)

    st.write("### 2. Review")

    reviews = pd.read_csv('ReviewRaw.csv')

    st.write(reviews.head(3))

    st.write('- *Số lượng sản phẩm*')
    st.write('Số lượng sản phẩm:', reviews.shape[0])

    st.write('- *Kiểu dữ liệu*')
    # st.write(df.info()) issue https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100/2
    buffer = io.StringIO()
    reviews.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write('- *Bỏ dữ liệu trùng*')
    st.write('Trước khi droping, records =', reviews.shape[0])
    reviews.drop_duplicates(inplace=True)
    st.write('Sau khi droping, records =', reviews.shape[0])

    st.write('- *Kiểm tra dữ liệu null*')
    st.text(reviews.isnull().any())

    m = []
    for c in ['full_name', 'created_time']:
        m.append({
            'feature': c,
            'MissingVal': reviews[reviews[c].isnull()].shape[0],
            'Percentage': reviews[reviews[c].isnull()].shape[0]/reviews.shape[0]*100
        })
    df = pd.DataFrame(m)
    df
    st.markdown(
        """
            #### Nhận xét:
            <div class="boxed">
                <ol class="s">
                    <li>created_time feature có dữ liệu null khá lớn.</li>
                    <li>full_name feature có dữ liệu null tương đối 9.5%.</li>
                    <li>name và full_name(nếu có) khá giống nhau.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True
    )

    st.write('- *Bỏ feature full_name và created_time*')
    reviews.drop(['full_name', 'created_time'], axis=1, inplace=True)

    st.write('- *lưu kết quả*')
    reviews.to_csv('Reviews.csv', index=False)

    st.write("### 3. Product - Review")

    st.write('- *sản phẩm không có trong review*')
    reviews[~reviews.product_id.isin(products.item_id)]

    st.write('- *loại bỏ review có mã sản phẩm không tồn tại*')
    reviews = reviews[reviews.product_id.isin(products.item_id)]

    reviews.reset_index(drop=True, inplace=True)

    st.write('- *lưu kết quả*')
    reviews.to_csv('Reviews.csv', index_label='id')
    products.to_csv('Products.csv', index=False)
