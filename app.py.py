import pickle
import numpy as np
import streamlit as st
from Orange.data import Table, Domain

# 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측기",
    page_icon="🚦",
    layout="centered"
)

# 1. 오렌지3 모델 로드 함수 (캐싱 적용으로 속도 최적화)
@st.cache_resource
def load_orange_model(model_path="traffic_casualty_model.pkcls"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_orange_model()
except FileNotFoundError:
    st.error("🚨 'traffic_casualty_model.pkcls' 파일을 찾을 수 없습니다. 깃허브 저장소 루트 경로에 파일이 있는지 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 모델의 Domain(변수 정보)에서 '시도' 목록 추출
feature_vars = {var.name: var for var in model.domain.attributes}
sido_var = feature_vars.get("시도")

if sido_var and hasattr(sido_var, "values") and sido_var.values:
    sido_list = list(sido_var.values)
else:
    # 모델 메타데이터에 목록이 없을 경우 기본 전국 시도 목록 사용
    sido_list = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기도", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]

# 타이틀 및 설명
st.title("🚦 어린이 보호구역 교통사고 위험도 예측")
st.markdown("지역과 사고 발생 지표를 조절하여 **예상 사망자 수**와 **위험 등급(신호등)**을 실시간으로 확인하세요.")
st.divider()

# 2. 사용자 입력 UI (실시간 슬라이더 및 선택창)
st.subheader("📊 사고 조건 입력")

col1, col2 = st.columns(2)

with col1:
    selected_sido = st.selectbox("📍 시도 선택", options=sido_list, index=0)
    accidents = st.slider("🚗 사고건수 (건)", min_value=0, max_value=100, value=5, step=1)
    minor_injuries = st.slider("🩹 경상자수 (명)", min_value=0, max_value=100, value=3, step=1)

with col2:
    severe_injuries = st.slider("🚑 중상자수 (명)", min_value=0, max_value=50, value=1, step=1)
    reported_injuries = st.slider("📋 부상신고자수 (명)", min_value=0, max_value=50, value=0, step=1)

# 3. 오렌지3 모델 예측 실행 함수
def predict_fatalities(sido, acc, severe, minor, reported):
    # 모델이 학습한 특성(Attributes) 순서에 맞게 입력 배열 생성
    input_domain = Domain(model.domain.attributes)
    row_values = []
    
    for attr in input_domain.attributes:
        if attr.name == "시도":
            # 범주형 변수는 해당 문자열의 인덱스 값으로 변환
            idx = attr.values.index(sido) if sido in attr.values else 0
            row_values.append(float(idx))
        elif attr.name == "사고건수":
            row_values.append(float(acc))
        elif attr.name == "중상자수":
            row_values.append(float(severe))
        elif attr.name == "경상자수":
            row_values.append(float(minor))
        elif attr.name == "부상신고자수":
            row_values.append(float(reported))
        else:
            row_values.append(0.0)
            
    data_table = Table.from_numpy(input_domain, np.array([row_values]))
    prediction = model(data_table)
    
    # 예측값이 음수로 나올 경우 0으로 보정
    pred_value = max(0.0, float(prediction[0]))
    return pred_value

# 실시간 예측 계산
predicted_deaths = predict_fatalities(
    selected_sido, accidents, severe_injuries, minor_injuries, reported_injuries
)

st.divider()

# 4. 예측 결과 및 대형 신호등 / 안전 수칙 카드 출력
st.subheader("🚨 실시간 예측 및 위험도 진단")

# 위험도 구간 설정 (데이터 특성에 따라 기준치를 자유롭게 조정하세요)
if predicted_deaths < 0.2:
    status_color = "#28a745"  # 초록
    bg_color = "#e8f5e9"
    light_emoji = "🟢"
    status_title = "안전 (SAFE)"
    status_desc = "예상 사망자 수가 매우 낮은 안정적인 수준입니다."
    safety_tips = [
        "**시속 30km 이하 서행:** 어린이 보호구역 내 제한속도를 항상 준수하세요.",
        "**전방 주시 철저:** 주정차된 차량 사이에서 어린이가 갑자기 튀어나올 수 있습니다.",
        "**횡단보도 일시정지:** 보행자가 없더라도 횡단보도 앞에서는 일단 멈춤을 생활화하세요."
    ]
elif predicted_deaths < 0.8:
    status_color = "#ffc107"  # 노랑
    bg_color = "#fffde7"
    light_emoji = "🟡"
    status_title = "주의 (CAUTION)"
    status_desc = "사고 발생 시 중대 인명 피해로 이어질 가능성이 있습니다."
    safety_tips = [
        "**스쿨존 불법 주정차 절대 금지:** 불법 주정차는 운전자의 시야를 가려 어린이 사고의 주원인이 됩니다.",
        "**우회전 시 일단멈춤:** 교차로 우회전 시 보행자 신호와 횡단보도 주변을 반드시 확인하세요.",
        "**등·하교 시간대 집중 경계:** 오전 8~9시, 오후 1~4시 사이에는 더욱 각별한 주의가 필요합니다."
    ]
else:
    status_color = "#dc3545"  # 빨강
    bg_color = "#ffebee"
    light_emoji = "🔴"
    status_title = "위험 (DANGER)"
    status_desc = "사망 사고 발생 위험이 매우 높습니다! 즉각적인 안전 대책과 방어운전이 필요합니다."
    safety_tips = [
        "**최고 수준의 방어운전:** 어린이 보호구역 진입 시 속도를 시속 20km 이하로 대폭 감속하세요.",
        "**스마트폰·내비게이션 조작 금지:** 단 1~2초의 전방 미주시가 치명적인 사고로 직결됩니다.",
        "**어린이 보행자 안전 지도:** 어린이는 차가 오더라도 멈추지 않고 뛰는 특성이 있으므로 완전히 지나갈 때까지 대기하세요."
    ]

# 대형 신호등 및 예측 수치 카드 렌더링
st.markdown(
    f"""
    <div style="
        background-color: {bg_color};
        border: 3px solid {status_color};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="font-size: 80px; line-height: 1.1;">{light_emoji}</div>
        <h2 style="color: {status_color}; margin: 10px 0 5px 0; font-weight: 800;">{status_title}</h2>
        <p style="font-size: 18px; color: #333; margin-bottom: 12px;">
            예상 사망자 수: <strong style="font-size: 26px; color: {status_color};">{predicted_deaths:.2f} 명</strong>
        </p>
        <p style="font-size: 15px; color: #555; margin: 0;">{status_desc}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 어린이 보호구역 안전 수칙 카드
st.markdown("### 🛡️ 어린이 보호구역 필수 안전 수칙")
for tip in safety_tips:
    st.info(tip)