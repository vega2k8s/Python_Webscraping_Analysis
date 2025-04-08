import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
data = pd.read_csv('data/data_draw_korea.csv')
data = data.drop('Unnamed: 0', axis=1)
plt.rc('font', family="Malgun Gothic")

# Streamlit UI 구성
st.title("📊 대한민국 광역시도 데이터 분석")

# 광역시도 목록 가져오기
sido_list = data['광역시도'].unique()

# 광역시도 선택
sido_name = st.selectbox("조회할 광역시도를 선택하세요", sido_list)

# 선택된 광역시도의 데이터 필터링
sido_df = data[data['광역시도'] == sido_name][['행정구역', '인구수', '면적']].reset_index(drop=True)

if sido_df.empty:
    st.error("❌ 해당 광역시도의 데이터를 찾을 수 없습니다.")
else:
    # 데이터 표시
    st.subheader(f"📊 {sido_name}의 DataFrame")
    st.dataframe(sido_df)

    # 인구수 그래프
    st.subheader(f"📈 {sido_name}의 인구수 현황")
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(x='행정구역', y='인구수', data=sido_df.sort_values(by='인구수', ascending=False), ax=ax, hue='행정구역')
    ax.set_title(f'{sido_name} 행정구역별 인구수')
    st.pyplot(fig)

    # 면적 그래프
    st.subheader(f"📈 {sido_name}의 면적 현황")
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(x='행정구역', y='면적', data=sido_df.sort_values(by='면적', ascending=False), ax=ax, hue='행정구역')
    ax.set_title(f'{sido_name} 행정구역별 면적')
    st.pyplot(fig)
