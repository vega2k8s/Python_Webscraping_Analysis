# streamlit run streamlit_korea_func.py
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

data = pd.read_csv('data/data_draw_korea.csv')
data = data.drop('Unnamed: 0', axis=1)
plt.rc('font', family="Malgun Gothic")

# 광역시도 이름을 인자로 받아서 인구수와 면적을 그려주는 함수
def show_pop_area(sido_name):
    sido_df = data.loc[data['광역시도'] == sido_name,['행정구역','인구수','면적']].reset_index(drop=True)
    
    if sido_df.empty:
        st.error("❌ 해당 광역시도 데이터를 찾을 수 없습니다. 정확한 이름을 입력해주세요.")
        return
        
    st.subheader(f"📊 {sido_name}의 DataFrame")
    st.dataframe(sido_df)

    st.subheader(f"📈 {sido_name}의 인구수 현황")

    fig, (axes1,axes2) = plt.subplots(nrows=2, ncols=1,figsize=(18, 12))
    pop_plot = sns.barplot(x='행정구역', y='인구수', data=sido_df.sort_values(by='인구수',ascending=False), ax=axes1,hue='행정구역')
    pop_plot.set_title(f'{sido_name} 행정구역별 인구수')
    
    area_plot = sns.barplot(x='행정구역', y='면적', data=sido_df.sort_values(by='면적',ascending=False), ax=axes2,hue='행정구역')
    area_plot.set_title(f'{sido_name} 행정구역별 면적')
    st.pyplot(fig)

# ✅ Streamlit UI 구성
st.title("📊 대한민국 광역시도 데이터 분석")

# 🔹 사용자 입력 받기
sido_name = st.text_input("조회할 광역시도를 입력하세요 (예: 서울특별시, 부산광역시)")
button_clicked = st.button("📌 조회")

# 🔹 버튼 클릭 시 함수 실행
if button_clicked:
    show_pop_area(sido_name)