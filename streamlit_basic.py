# streamlit run streamlit_basic.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 샘플 데이터 생성
data = {'선거구': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        '성별': ['남', '여', '남', '여', '남', '여', '남', '여', '남']}
df = pd.DataFrame(data)

# Streamlit 앱
st.title("📊 Pandas DataFrame & Seaborn Plot")

# DataFrame 출력
st.subheader("📋 데이터 테이블")
st.dataframe(df)

# 시각화
st.subheader("📈 Seaborn 그래프")
fig, ax = plt.subplots()
sns.countplot(data=df, x='선거구', hue='성별', palette={'남': 'navy', '여': 'orange'}, ax=ax)
st.pyplot(fig)