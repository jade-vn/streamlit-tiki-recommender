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


def loadEDA():
    pd.options.display.float_format = '{:.2f}'.format

    st.markdown("""# <font color='F00FF'>Đọc dữ liệu</font>""",
                unsafe_allow_html=True)

    reviews = pd.read_csv('Reviews.csv', lineterminator='\n')

    products = pd.read_csv('Products.csv')

    st.markdown("""# <font color='F00FF'>Product</font>""",
                unsafe_allow_html=True)

    # pr = ProfileReport(products)
    # st_profile_report(pr)
    st.write(products.head(2))

    st.write("- **Số lượng sản phẩm**")
    st.write('Số lượng sản phẩm:', products.shape[0])

    st.write("- **Kiểu dữ liệu**")
    # st.write(df.info()) issue https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100/2
    buffer = io.StringIO()
    products.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write(products[['price', 'list_price']].describe().T)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>rating dao động trong khoảng từ 0 đến 5</li>
                <li>Khoảng giá giao động rất rộng từ 7000.00 đến 62690000.00</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""## <font color='orange'>Giá</font>""",
                unsafe_allow_html=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    products.price.plot(kind='box', ax=ax[0])
    products.price.plot(kind='hist', bins=20, ax=ax[1])
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Khoảng giá của sản phẩm giao động rất rộng từ 7000.00 đến 62690000.00</li>
                <li>Phần lớn giá của sản phẩm đều tập trung < 3000000.00</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""## <font color='orange'>Thương hiệu</font>""",
                unsafe_allow_html=True)

    st.write("- **Sản phẩm theo thương hiệu**")
    brands = products.groupby(
        'brand')['item_id'].count().sort_values(ascending=False)
    st.text(brands)

    st.write("- **Top 10 thương hiệu có số lượng mã sản phẩm cao nhất**")
    brands[1:11].plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Product Items by brand')
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Sản phẩm có thương hiệu Samsung chiếm phần lớn.</li>
                <li>Các thương hiệu còn lại có số lượng tương đương nhau.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.write("- **Giá bán theo thương hiệu**")
    price_by_brand = products.groupby(by='brand').mean()['price']
    price_by_brand.sort_values(ascending=False)[:10].plot(kind='bar')
    plt.ylabel('Price')
    plt.title('Average Price by brand')
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Hitachi là thưng hiệu có giá bán trung bình cao nhất.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""## <font color='orange'>**Rating**</font>""",
                unsafe_allow_html=True)

    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=products, x='rating', kde=False, alpha=0.8)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Sản phẩm có rating 0 và 5 cao nhất.</li>
                <li>Phần lớn sán phẩm có rating > 4</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.write("- **Nhóm và xem xét rating trong dataset product**")
    st.text(products.groupby(['rating'])['item_id'].count())

    st.write("- **Xem xét product rating trong review của khác hàng**")

    avg_price_customer = reviews.groupby(by='product_id').mean()[
        'rating'].to_frame().reset_index()
    avg_price_customer.rename({'rating': 'avg_rating'}, axis=1, inplace=True)
    st.write(avg_price_customer.head())

    products = products.merge(
        avg_price_customer, left_on='item_id', right_on='product_id', how='left')

    fig_01 = plt.figure(figsize=(10, 4))
    sns.histplot(data=products, x='rating', kde=False, alpha=0.8)
    st.pyplot(fig_01, clear_figure=True)
    fig_02 = plt.figure(figsize=(10, 4))
    sns.histplot(data=products, x='avg_rating', kde=False, alpha=0.8)
    st.pyplot(fig_02, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Rating của sản phẩm trong review của khách hàng > 0.</li>
                <li>Có thể kết luận điểm rating =0 trong product là do thiếu dữ liệu.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""# <font color='F00FF'>Review</font>""",
                unsafe_allow_html=True)

    # pr = ProfileReport(reviews)
    # st_profile_report(pr)
    st.write(reviews.head())

    st.write("- **Số lượng sản phẩm**")
    st.write('Số lượng sản phẩm:', reviews.shape[0])

    st.write("- **Kiểu dữ liệu**")
    # st.write(df.info()) issue https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100/2
    buffer = io.StringIO()
    reviews.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    fig = plt.figure(figsize=(5, 5))
    ax = sns.histplot(data=reviews, x='rating', stat="density", kde=True,
                      alpha=0)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 5)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Phần lớn khác hàng phản hồi tích cực về sản phẩm.</li>
                <li>Sản phẩm có chất lương tốt hoặc khách hàng dễ tính.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.text(reviews.groupby(['rating']).size())

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Phần lớn đánh giá rating là 5</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""## <font color='orange'>**Top 20 sản phẩm được đánh giá nhiều nhất**</font>""",
                unsafe_allow_html=True)

    plt.figure(figsize=(16, 8))
    top_products = reviews.groupby('product_id').count(
    )['customer_id'].sort_values(ascending=False)[:20]
    top_products.index = products[products.item_id.isin(
        top_products.index)]['name'].str[:25]
    top_products.plot(kind='bar')
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        ▶ Nhận xét:
        <div class="boxed">
            <ol class="s">
                <li>Phụ kiện điện thoại máy tính.</li>
                <li>Chuột không dây Logitech được đánh giá nhiều nhất</li>
            </ol>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("""## <font color='orange'>**Top 20 khách hàng thực hiện đánh giá nhiều nhất**</font>""",
                unsafe_allow_html=True)

    st.write("- **Top 20 customer thực hiện đánh giá nhiều nhất**")
    top_rating_customers = reviews.groupby('customer_id').count()[
        'product_id'].sort_values(ascending=False)[:20]

    plt.figure(figsize=(16, 8))
    plt.bar(x=[str(x) for x in top_rating_customers.index],
            height=top_rating_customers.values)
    plt.xticks(rotation=70)
    st.pyplot(fig, clear_figure=True)
