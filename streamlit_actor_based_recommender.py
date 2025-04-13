# streamlit run actor_based_recommender.py
'''
핵심 추천 로직 (TF-IDF + Cosine Similarity)
1. 배우 입력 → 출연작 목록 필터링

사용자가 출연작 중 하나 선택

해당 콘텐츠의 설명(Description), 장르, 유형(Type) 등을 기반으로 콘텐츠 벡터화

Cosine Similarity를 활용해 유사한 콘텐츠 추천

Streamlit UI로 결과 출력
'''
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df['cast'] = df['cast'].fillna('No Data')
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    return df

df = load_data()

st.title("🎬 출연작 기반 콘텐츠 추천 시스템")
st.write("선호 배우 → 출연작 선택 → 유사한 넷플릭스 콘텐츠 추천")

# 1. 배우 이름 입력
actor = st.text_input("🔍 좋아하는 배우를 입력하세요 (예: Leonardo DiCaprio)")

if actor:
    # 2. 해당 배우 출연작 추출
    actor_movies = df[df['cast'].str.contains(actor, case=False, na=False)]
    
    if actor_movies.empty:
        st.warning("😢 해당 배우의 출연작이 없습니다.")
    else:
        st.success(f"{actor} 출연작 {len(actor_movies)}편 발견!")
        # 3. 출연작 중 하나 선택
        selected_title = st.selectbox("🎞 추천 기준으로 삼을 작품을 선택하세요", actor_movies['title'].unique())

        # 4. TF-IDF 기반 콘텐츠 벡터화
        df['content'] = df['type'] + " " + df['listed_in'] + " " + df['description']
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])

        # 5. 선택한 작품 인덱스
        target_idx = df[df['title'] == selected_title].index[0]

        # 6. 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()

        # 7. 유사도 높은 콘텐츠 상위 5개 추천 (자기 자신 제외)
        similar_indices = cosine_sim.argsort()[::-1][1:6]
        recommendations = df.iloc[similar_indices][['title', 'type', 'release_year', 'listed_in']]

        st.subheader("🎁 추천 콘텐츠 TOP 5")
        st.dataframe(recommendations)
