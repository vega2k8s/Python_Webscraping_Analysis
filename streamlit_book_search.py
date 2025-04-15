# streamlit run streamlit_book_search.py

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="네이버 도서 검색", layout="wide")
st.title("네이버 도서 검색 애플리케이션")

# .env 파일에서 환경 변수 로드
load_dotenv()
client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

headers = {
    'X-Naver-Client-Id': client_id,
    'X-Naver-Client-Secret': client_secret,
}

# 만약 환경 변수가 없다면 사이드바에서 입력받도록 함
if not client_id or not client_secret:
    st.sidebar.header("API 키 설정")
    client_id = st.sidebar.text_input("NAVER_CLIENT_ID 입력", type="password")
    client_secret = st.sidebar.text_input("NAVER_CLIENT_SECRET 입력", type="password")
    
    # API 키 정보가 필요함을 알림
    if not client_id or not client_secret:
        st.warning("네이버 API 사용을 위해 CLIENT_ID와 CLIENT_SECRET을 입력해주세요.")

def search_naver_api(endpoint, query, display=50):
    """네이버 API 검색 함수"""
    payload = {
        'query': query,
        'display': display,
        'sort': 'sim'
    }
    url = f'https://openapi.naver.com/v1/search/{endpoint}.json'
    
    try:
        res = requests.get(url, params=payload, headers=headers)
        res.raise_for_status()  # 에러 발생 시 예외 처리
        return res.json().get('items', [])
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 오류가 발생했습니다: {e}")
        return []

def save_json(data, filepath):
    """JSON 파일 저장 함수"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    st.success(f"데이터가 {filepath}에 저장되었습니다.")

def filter_and_sort_books(df, min_discount=20000):
    """
    할인 금액이 min_discount 이상인 도서 필터링 후 정렬
    
    Parameters:
        df (DataFrame): 도서 데이터프레임
        min_discount (int): 최소 할인 금액 기준 (기본값 20000)
    Returns:
        DataFrame: 필터링 및 정렬된 결과
    """
    if df.empty:
        return pd.DataFrame()
    
    # discount 열이 문자열일 경우 숫자로 변환
    if df['discount'].dtype == 'object':
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce')
    
    return (
        df.loc[df['discount'] >= min_discount, ['title', 'author', 'discount', 'publisher', 'pubdate']]
          .sort_values(by='discount', ascending=False)
          .reset_index(drop=True)
    )

def filter_books_by_publisher(df, publisher_name):
    """
    특정 출판사가 포함된 도서만 필터링 (image, description 컬럼 제외)
    Parameters:
        df (DataFrame): 도서 데이터프레임
        publisher_name (str): 포함할 출판사 이름
    Returns:
        DataFrame: 필터링된 결과
    """
    if df.empty or publisher_name == "":
        return pd.DataFrame()
    
    # 컬럼 목록 필터링 ('image'와 'description' 제외)
    columns_to_show = [col for col in df.columns if col not in ['image', 'description']]
    
    return (
        df.loc[df['publisher'].str.contains(publisher_name, na=False), columns_to_show]
          .reset_index(drop=True)
    )

# 사이드바에 검색 옵션 구성
st.sidebar.header("검색 옵션")
search_query = st.sidebar.text_input("검색어", "파이썬")
display_count = st.sidebar.slider("검색 결과 수", 10, 100, 50)

# 검색/저장 버튼
search_button = st.sidebar.button("검색하기")
save_button = st.sidebar.button("결과 저장하기")

# 필터링 옵션
st.sidebar.header("필터링 옵션")
min_discount = st.sidebar.number_input("최소 할인 금액", 0, 100000, 20000, 1000)
publisher_filter = st.sidebar.text_input("출판사 필터", "")

# 메인 영역
tab1, tab2, tab3 = st.tabs(["전체 결과", "할인 필터링", "출판사 필터링"])

# 데이터를 저장할 상태 변수
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
    st.session_state.books_df = pd.DataFrame()

# 검색 버튼 클릭
if search_button and client_id and client_secret:
    with st.spinner('검색 중...'):
        st.session_state.search_results = search_naver_api('book', search_query, display_count)
        if st.session_state.search_results:
            st.session_state.books_df = pd.DataFrame(st.session_state.search_results)
            st.success(f"{len(st.session_state.search_results)}개의 도서를 찾았습니다.")
        else:
            st.warning("검색 결과가 없습니다.")

# 저장 버튼 클릭
if save_button and st.session_state.search_results:
    filepath = f"data/{search_query}_books.json"
    save_json(st.session_state.search_results, filepath)

# 탭 1: 전체 결과
with tab1:
    if not st.session_state.books_df.empty:
        st.write("전체 검색 결과")
        st.dataframe(st.session_state.books_df, use_container_width=True)
    else:
        st.info("검색 결과가 없습니다. 검색 버튼을 클릭하여 결과를 불러오세요.")

# 탭 2: 할인 필터링
with tab2:
    if not st.session_state.books_df.empty:
        filtered_by_discount = filter_and_sort_books(st.session_state.books_df, min_discount)
        if not filtered_by_discount.empty:
            st.write(f"할인 금액 {min_discount}원 이상 도서")
            st.dataframe(filtered_by_discount, use_container_width=True)
        else:
            st.info(f"할인 금액이 {min_discount}원 이상인 도서가 없습니다.")
    else:
        st.info("검색 결과가 없습니다. 검색 버튼을 클릭하여 결과를 불러오세요.")

# 탭 3: 출판사 필터링
with tab3:
    if not st.session_state.books_df.empty and publisher_filter:
        filtered_by_publisher = filter_books_by_publisher(st.session_state.books_df, publisher_filter)
        if not filtered_by_publisher.empty:
            st.write(f"출판사 '{publisher_filter}'가 포함된 도서")
            st.dataframe(filtered_by_publisher, use_container_width=True)
        else:
            st.info(f"출판사 이름에 '{publisher_filter}'가 포함된 도서가 없습니다.")
    else:
        st.info("검색 결과가 없거나 출판사 필터가 입력되지 않았습니다.")