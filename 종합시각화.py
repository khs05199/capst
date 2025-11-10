#===================================================================
#1. 라이브러기 가져오기
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os 
#===================================================================
#2. 페이지 설정
st.set_page_config(
    page_title='대구시 공영주차장 태양광 & 혼잡도 통합 대시보드',
    page_icon='☀️⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)
#===================================================================
#3. 데이터 경로 설정 (GitHub/로컬 폴더 기준)
MAIN_DATA_PATH = '태양광_일사량 및 주차 구획수.xlsx'
CONGESTION_DATA_PATH = '혼잡도_요일별_시간별_요약.xlsx'
#===================================================================
#4. 고정 파라미터

## 하루 평균 충전 차량 수
EV_COUNT_PER_DAY = 4
##EV 평균 배터리 용량
EV_BATTERY_KWH = 80
##태양광으로 충당할 전체 충전량 비율
PV_TARGET_RATIO = 0.30
#E#SS round-trip efficiency
ESS_RTE = 0.85
##태양광 모듈 효율
PV_EFFICIENCY = 0.18
##인버터 및 시스템 손실 반영
SYSTEM_LOSS = 0.80 
##한 주차구획 면적 (m²)
PARKING_AREA_PER_SLOT = 12.5
## 1년
DAYS_PER_YEAR = 365
#===================================================================
#5. 태양광 일사량 적합도 분류
def calculate_pv_requirements(file_path):
    # 데이터 불러오기
    df = pd.read_excel(file_path)
    
    #하루 목표 태양광 발전량 (ESS 효율 반영)
    daily_ev_demand = EV_COUNT_PER_DAY * EV_BATTERY_KWH
    target_pv_energy = daily_ev_demand * PV_TARGET_RATIO
    required_pv_output = target_pv_energy / ESS_RTE  # kWh/day
    
    #주차장별 계산 수행
    df["㎡당_일평균_발전량(kWh/m²/day)"] = (
        df["㎡당 연간 일사량(kWh/m²/yr)"] * PV_EFFICIENCY * SYSTEM_LOSS / DAYS_PER_YEAR
    )
    
    df["필요패널면적(m²)"] = required_pv_output / df["㎡당_일평균_발전량(kWh/m²/day)"]
    df["필요구획수"] = df["필요패널면적(m²)"] / PARKING_AREA_PER_SLOT

    #적합/부적합 기준 분류
    df["태양광 적합 여부"] = df.apply(
        lambda row: (
            "부적합" if (row["필요구획수"] < 80 and row["필요구획수"] > row["총주차면수"] * 0.5)
            else "적합"
        ),
        axis=1
    )
    
    #정리
    result = df[
        [
            "주차장_ID", "지번주소", "주차장명", "총주차면수",
            "㎡당 연간 일사량(kWh/m²/yr)",
            "필요패널면적(m²)", "필요구획수", "태양광 적합 여부",
            "위도", "경도"
        ]
    ]
    
    return result.round(2)
#===================================================================
#6. 혼잡도 상태 분류
def classify_congestion(pv_df, congestion_file_path):
    #혼잡도 엑셀 파일의 모든 시트 읽기 (월~일)
    sheets = pd.read_excel(congestion_file_path, sheet_name=None, index_col=0)

    #모든 요일 시트의 합계를 계산
    total_congestion = None
    for day, df_day in sheets.items():
        # % 기호 제거 및 float 변환
        df_day = df_day.replace('%', '', regex=True).astype(float)
        
        if total_congestion is None:
            total_congestion = df_day
        else:
            total_congestion += df_day
    
    #주차장별 일주일 총합 평균 (시간별 평균을 통해)
    weekly_avg_congestion = total_congestion.mean(axis=0)  # axis=0 → 주차장별 평균
    
    #0~1 정규화
    min_val, max_val = weekly_avg_congestion.min(), weekly_avg_congestion.max()
    normalized = (weekly_avg_congestion - min_val) / (max_val - min_val)
    
    #혼잡도 라벨링
    def congestion_label(x):
        if pd.isna(x):
            return np.nan
        elif x < 0.7:
            return '여유'
        elif x < 0.9:
            return '보통'
        else:
            return '혼잡'
    
    congestion_labels = normalized.apply(congestion_label)
    
    #DataFrame으로 변환
    congestion_df = pd.DataFrame({
        '주차장_ID': normalized.index,
        '정규화_혼잡도': normalized.values,
        '혼잡도': congestion_labels.values
    })
    
    # 7️⃣ 태양광 부적합 주차장은 혼잡도 NaN 처리
    merged = pv_df.merge(congestion_df, on='주차장_ID', how='left')
    merged.loc[merged['태양광 적합 여부'] == '부적합', ['정규화_혼잡도', '혼잡도']] = np.nan
    
    return merged
#===================================================================
#7. 최종 선별 데이터 프레임 
#태양광 및 ESS 관련 계산
pv_df = calculate_pv_requirements('태양광_일사량 및 주차 구획수.xlsx')

#혼잡도 데이터 기반 차량 흐름 분류
car_df = classify_congestion(pv_df, CONGESTION_DATA_PATH)

columns_to_display = [
    '주차장_ID', '주차장명', '지번주소', '총주차면수',
    '㎡당 연간 일사량(kWh/m²/yr)', '필요패널면적(m²)', '필요구획수',
    '태양광 적합 여부', '정규화_혼잡도', '혼잡도', '위도', '경도'
]

# 컬럼 순서 정리
final_df = car_df[columns_to_display]

# 인덱스 초기화
final_df.reset_index(drop=True, inplace=True)

#===================================================================
# 8. 시각화 대시보드
# 🧭 사이드바 설정
# =========================================================
st.sidebar.header("📍 필터 선택")

gu_list = ['전체'] + sorted(
    final_df['지번주소'].astype(str)
    .str.extract(r'대구광역시 (\w+)')[0]
    .dropna()
    .unique()
)

selected_gu = st.sidebar.selectbox('담당구 선택', gu_list)

if selected_gu != '전체':
    filtered_df = final_df[final_df['지번주소'].str.contains(selected_gu, na=False)]
else:
    filtered_df = final_df.copy()

parking_list = ['전체'] + list(filtered_df['주차장명'].unique())
selected_parking = st.sidebar.selectbox('주차장 선택', parking_list)

solar_options = ['전체', '적합', '부적합']
selected_solar = st.sidebar.selectbox('태양광 적합 여부', solar_options)

cong_options = ['전체', '여유', '보통', '혼잡']
selected_cong = st.sidebar.selectbox('혼잡도 상태', cong_options)

# 필터 적용
if selected_solar != '전체':
    filtered_df = filtered_df[filtered_df['태양광 적합 여부'] == selected_solar]
if selected_cong != '전체':
    filtered_df = filtered_df[filtered_df['혼잡도'] == selected_cong]
if selected_parking != '전체':
    filtered_df = filtered_df[filtered_df['주차장명'] == selected_parking]

# =========================================================
# 🎨 색상 매핑
# =========================================================
color_map = {'여유': '#2ecc71', '보통': '#f39c12', '혼잡': '#e74c3c'}

# =========================================================
# 🗺️ 지도 시각화
# =========================================================
st.markdown("## ☀️⚡ 대구시 공영주차장 태양광 & 혼잡도 통합 대시보드")
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    title_text = "🗺️ 대구시 공영주차장 태양광 설치 적합도 지도" \
        if selected_gu == '전체' else f"🗺️ {selected_gu} 공영주차장 태양광 설치 지도"
    st.subheader(title_text)

    if filtered_df.empty:
        st.warning("선택 조건에 해당하는 데이터가 없습니다.")
    else:
        # 적합/부적합 분리
        suitable_df = filtered_df[filtered_df['태양광 적합 여부'] != '부적합']
        unsuitable_df = filtered_df[filtered_df['태양광 적합 여부'] == '부적합']

        # 지도 시각화
        fig = px.scatter_mapbox(
            suitable_df,
            lat='위도',
            lon='경도',
            hover_name='주차장명',
            hover_data=['지번주소', '태양광 적합 여부', '혼잡도', '주차장_ID'],
            color='혼잡도',
            color_discrete_map=color_map,
            zoom=11,
            height=650,
            size_max=30
        )

        # 부적합 주차장은 ❌ 표시
        fig.add_trace(go.Scattermapbox(
        lat=unsuitable_df['위도'],
        lon=unsuitable_df['경도'],
        mode='markers+text',
        text=["❌"]*len(unsuitable_df),
        textfont=dict(size=20, color="red"),
        marker=dict(size=0, color='rgba(0,0,0,0)'),  # 투명 점
        hovertext=unsuitable_df['주차장명'],
        hoverinfo="text"
    ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 35.8714, "lon": 128.6014},
            margin={"r":0,"t":20,"l":0,"b":0},
            legend_title_text="혼잡도 상태",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ➡️ 오른쪽 상세 정보
# =========================================================
with col2:
    st.subheader("📊 선택 주차장 상세 정보")
    
    if selected_parking != '전체' and not filtered_df.empty:
        info = filtered_df.iloc[0]
        st.markdown(f"**주차장명:** {info['주차장명']}")
        st.markdown(f"**주차장 ID:** {info['주차장_ID']}")
        st.markdown(f"**주소:** {info['지번주소']}")
        st.markdown(f"**총 주차면수:** {info['총주차면수']}대")
        st.markdown(f"**필요 태양광 패널 면적:** {info['필요패널면적(m²)']:.2f} m²")
        st.markdown(f"**필요 주차 구획수:** {info['필요구획수']}대")
        
        # 총 주차면수 대비 필요한 패널 면적 비율
        total_parking_area = info['총주차면수'] * PARKING_AREA_PER_SLOT
        pv_fill_ratio = info['필요패널면적(m²)'] / total_parking_area
        st.markdown(f"**태양광 패널로 충당 비율:** {pv_fill_ratio*100:.1f}%")
        st.markdown(f"**연간 일사량:** {info['㎡당 연간 일사량(kWh/m²/yr)']} kWh/m²/yr")
        st.markdown(f"**태양광 적합 여부:** {info['태양광 적합 여부']}")

        # 혼잡도 상태 바
        st.markdown("**혼잡도 상태:**")
        if not pd.isna(info['정규화_혼잡도']):
            st.progress(int(info['정규화_혼잡도']*100))
            st.markdown(f"**혼잡도:** {info['혼잡도']}")
        else:
            st.markdown("**혼잡도:** 표시 불가 (태양광 부적합)")

        # ==============================
        # 시간별 혼잡도 그래프
        # ==============================
        # 엑셀 불러오기 (모든 시트)
        sheets = pd.read_excel(CONGESTION_DATA_PATH, sheet_name=None, index_col=0)

        # 요일 선택
        selected_day = st.selectbox("📅 요일 선택", list(sheets.keys()))

        df_day = sheets[selected_day].copy()
        df_day = df_day.replace('%', '', regex=True).astype(float)

        if info['주차장_ID'] in df_day.columns:
            y_values = df_day[info['주차장_ID']] * 100  # 0~100%
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_day.index,
                y=y_values,
                mode='lines+markers',
                name=info['주차장_ID'],
                text=[f"{v:.1f}%" for v in y_values],
                hovertemplate='%{text}<extra></extra>'
            ))
            fig.update_layout(
                title=f"{selected_day} {info['주차장명']} 시간별 혼잡도 (%)",
                xaxis_title="시간",
                yaxis_title="혼잡도 (%)",
                hovermode="x unified",
                template="plotly_white"
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_center={"lat": 35.8714, "lon": 128.6014},
                margin={"r":0,"t":20,"l":0,"b":0},
            )

            # 지도 출력
            st.plotly_chart(fig, use_container_width=True)

            # 🧭 지도 아래에 커스텀 범례 추가
            legend_html = """
            <div style="text-align:center; font-size:16px; margin-top:-20px;">
                <span style="color:#2ecc71; font-weight:bold;">● 여유</span>　
                <span style="color:#f39c12; font-weight:bold;">● 보통</span>　
                <span style="color:#e74c3c; font-weight:bold;">● 혼잡</span>　
                <span style="color:red; font-weight:bold;">❌ 태양광 부적합</span>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        
        else:
            st.info("해당 주차장은 선택한 요일 데이터가 없습니다.")

    else:
        st.info("지도 또는 사이드바에서 주차장을 선택해주세요.")