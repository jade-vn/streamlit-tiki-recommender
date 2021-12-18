
from pyspark.sql.functions import col, explode
from pyspark.sql.functions import isnull, when, count, col
from pyspark.sql.types import IntegerType, DoubleType
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pyspark
import findspark
import os
import streamlit as st
import io

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.0-bin-hadoop2.7"
findspark.init()


def loadCollaborativeFiltering_Recommender():
    # Đề xuất cho người dùng các sản phẩm dựa vào sự tương quan người dùng với nhau
    #
    st.write("- **Part2: Đề xuất cho 1 người dùng cụ thể**")

    SparkContext.setSystemProperty('spark.executor.memory', '12g')
    sc = SparkContext(master='local', appName='Recommendation_Beauty')

    spark = SparkSession(sc)

    recommenders = spark.read.parquet('user_recs.parquet')

    st.text(recommenders.printSchema())

    st.text(recommenders.show(5))

    st.write("- **Recommendation for customer_id = 6177374**")
    customer_id = 6177374
    find_user_rec = recommenders.filter(
        recommenders['customer_id'] == customer_id)
    st.text(find_user_rec.show(truncate=False))

    st.write("- **Solution 1: Consim**")

    rec = find_user_rec.select(find_user_rec.customer_id,
                               explode(find_user_rec.recommendations))

    st.text(rec.show())

    rec = rec.withColumn('product_id',
                         rec.col.getField('product_id'))\
        .withColumn('rating',
                    rec.col.getField('rating'))
    st.text(rec.show())

    st.write("- **Filter all products having rating > 4.0**")
    st.text(rec.filter(rec.rating >= 4.0).show())

    st.write("- **Solution 2: Gemsim**")
    result = ''
    for user in find_user_rec.collect():
        lst = []
        for row in user['recommendations']:
            st.text(row)
            lst.append((row['product_id'],
                        row['rating']))
        dic_user_rec = {'customer_id': user.customer_id,
                        'recommendations': lst}
        result = dic_user_rec

    st.write('Recommendation for: ', '6177374')
    st.text(result)
