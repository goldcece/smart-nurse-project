import time
import streamlit as st
import os
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(
    page_title="SmartNurse ChatBot",
    page_icon="👩‍⚕️",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        with st.chat_message('ai'):
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


azure = AzureChatOpenAI(
    api_key='key',
    api_version='version',
    azure_endpoint="endpoint",
    deployment_name='model',
    model_name='model_name',
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),],
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

# 기본 프롬프트 템플릿 생성
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 해결할 사항: 줄바꿈, 양식 넣어야 하는지, 

# NANDA
examples_NANDA = [
    {"input": "비효과적 호흡 양상 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n\n이름: 이해준\n나이: 34세\n성별: 남성\n\n진단: 비효과적 호흡 양상 (NANDA-I #00032)\n\n자료 수집 주관적 / 객관적:\n\n-호흡수 증가 (분당 24회)\n-호흡곤란 증상 (얕고 빠른 호흡)\n-산소포화도 감소 (92%)\n-환자의 주관적 호흡 불편 표현\n\n단기 목표:\n\n-환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소\n-산소포화도가 95% 이상으로 유지\n\n장기 목표:\n\n-환자가 호흡 곤란을 호소하지 않고, 산소포화도가 지속적으로 95% 이상 유지\n-환자가 호흡에 관련된 자가 관리 방법을 이해하고 실행\n\n간호수행/중재/이론적 근거 Interventions:\n\n-호흡 상태 모니터링: 2시간마다 호흡수, 산소포화도 체크\n-산소 요법: 의사의 지시에 따라 산소 공급 (필요 시)\n-호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육\n-안위 제공: 편안한 자세 유지, 스트레스 감소 방법 안내\n-교육 및 상담: 호흡곤란의 원인과 관리 방법 설명\n\n간호평가 Evaluation\n-호흡 상태: 호흡수 및 산소포화도 변화 모니터링\n-환자의 자가 관리 능력: 호흡 운동 실시 여부 및 효과 평가\n-교육 효과: 환자의 이해도 및 반응 관찰"},
    {"input": "비효과적 기도 청결 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 이상진\n 나이: 45세\n 성별: 남성\n\n진단: 비효과적 기도 청결 (NANDA-I #00031)\n\n자료 수집 주관적 / 객관적:\n\n-가래 증가 및 점도 높음\n-기침 효과 감소\n-호흡 시 천명음\n-환자 호흡 불편 호소\n\n단기 목표:\n\n-24시간 내 가래 제거 용이\n-호흡 시 청결한 기도 유지\n\n장기 목표:\n\n-기도 청결 상태 유지\n-호흡 곤란 또는 가래 문제 불평 없음\n-자가 관리 능력 향상\n\n간호수행/중재/이론적 근거 Interventions:\n\n-호흡 상태 및 가래 상태 모니터링: 매 4시간마다 폐 청진 및 가래 양, 성질 확인\n-가습 요법: 적절한 수분 섭취 및 환경 가습 유지\n-기도 청소 지원: 체위 배액, 가슴 타진, 효과적 기침 기술 교육 및 실습\n-흡인 준비: 필요 시 즉각적인 흡인 가능하도록 기구 준비\n-환자 및 보호자 교육: 기도 청결 유지 방법, 중요성 설명\n\n간호평가 Evaluation\n\n-호흡 상태 변화: 호흡수, 청진 소견, 가래 양 및 성질 추적\n-자가 관리 능력: 기도 청결 유지 기술 실천 여부 및 효과 평가\n-교육 이해도: 환자 및 보호자의 반응 및 피드백 확인"},
    {"input": "가스 교환 장애 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 정호석\n 나이: 32세\n 성별: 남성\n\n진단: 가스 교환 장애 (NANDA-I #00030)\n\n자료 수집 주관적 / 객관적:\n\n-동맥혈 가스 분석 상 산소 포화도 88%, 이산화탄소 농도 상승\n-호흡곤란, 얕고 빠른 호흡 (분당 30회)\n-시안증, 창백함 관찰\n-환자 호흡 불편 호소\n\n단기 목표:\n\n-24시간 내 산소포화도 94-98% 범위 내 유지\n-호흡곤란 증상 감소\n\n장기 목표:\n\n-호흡 곤란 없이 안정된 가스 교환 상태 유지\n-호흡 관리 자가 수행 능력 향상\n\n간호수행/중재/이론적 근거 Interventions:\n\n-호흡 상태 및 산소포화도 모니터링: 정기적으로 호흡수, 산소포화도 체크\n-산소 요법: 의사 지시에 따라 산소 공급 조정\n-호흡 운동 및 체위 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육 및 실습\n-스트레스 관리: 안정 및 휴식 촉진, 환자 불안 감소 기법 안내\n-환자 및 보호자 교육: 가스 교환 장애 원인, 중요성 및 자가 관리 방법 설명\n\n간호평가 Evaluation\n\n-호흡 상태 변화: 호흡수, 산소포화도 추적\n-자가 관리 능력: 호흡 운동 수행 여부 및 효과 평가\n-교육 이해도: 환자 및 보호자 반응 및 피드백 확인"},
    {"input": "낙상 위험 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 이윤아\n 나이: 67세\n 성별: 여성\n\n진단: 낙상 위험 (NANDA-I #00155)\n\n자료 수집 주관적 / 객관적:\n\n-보행 불안정\n-균형 장애\n-약물 부작용 영향 (예: 진정제 사용)\n-시각 장애\n-환경적 위험 요소 (예: 미끄러운 바닥)\n\n단기 목표:\n\n-24시간 내 환자 안전한 이동 및 활동 유지\n-환자 및 보호자 낙상 예방 조치 이해 및 준수\n\n장기 목표:\n\n-입원 기간 낙상 없이 안전 유지\n-자가 관리를 통한 낙상 위험 최소화\n\n간호수행/중재/이론적 근거 Interventions:\n\n-낙상 위험 평가: 정기적 낙상 위험 평가 수행\n-환경 조정: 낙상 위험 감소 환경 조성 (예: 미끄럼 방지 매트, 적절한 조명 설치)\n-보행 보조기구 사용: 필요 시 보행 보조기구 제공 및 사용법 교육\n-약물 관리: 낙상 위험 증가 약물 모니터링 및 조정\n-교육 및 상담: 낙상 예방 방법 및 환경 안전 유지 교육\n\n간호평가 Evaluation\n\n-이동 능력: 보행 안정성 및 균형 유지 능력 평가\n-환경 안전성: 환경 안전 지속적 확인\n-교육 효과: 낙상 예방 조치 이해도 및 실행 여부 평가"}
]
few_shot_prompt_NANDA = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_NANDA 
)
instruction_NANDA = """
You are a 5th nurse working in a general hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
3. Use precise and concise language. Use endings like 음, 함, 임.
4. Necessary to write a nursing record following this format.
###
환자 정보
- 이름:
- 나이:
- 성별:

자료 수집 주관적 / 객관적
- Substructure

간호목표 단기/장기 Goal
- Substructure

간호계획 Plan
- Substructure

간호수행/중재/이론적 근거 Interventions
- Substructure

간호평가 Evaluation
- Substructure
###
5. Combine '주요 증상 및 특성', '자료 수집 주관적', '자료 수집 객관적' in NANDA format to '자료 수집 주관적 / 객관적'.
6. Combine '단기목표' and '장기목표' in NANDA format to '간호목표 단기/장기 Goal'.
7. Instead of using numeric order, Use bullet points for the description.
8. Please write as specifically as possible. For example, instead of a generic response like '적절한 호흡 기법 교육', provide a more detailed answer such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육'.
9. Record precise numerical data. For example, provide specific details like '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소'.
"""

final_prompt_NANDA = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_NANDA),
        few_shot_prompt_NANDA,
        ("human", "{input}"),
    ]
)

# SOAPIE
examples_SOAPIE = [
    {"input": "불안정한 혈압의 위험 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n 이름: 김경주\n 나이: 61세\n 성별: 남성\n\n\주관적 증상 Subjective Data\n 환자는 최근에 두통과 어지러움을 자주 느끼며 \"눈 앞이 흐릿하고 가끔씩 두근거리는 느낌이 들어\"라고 언급함.\n\n객관적 정보 Objective Data \n혈압 측정 결과 : 수축기 혈압이 150 mmHg를 넘어서고 이완기 혈압도 95 mmHg를 넘어섬. \n심박수는 정상 범위 내에 있음.\n\n사정 Assessment \n환자는 불안정한 혈압 상태에 있으며  고혈압에 따른 두통 및 어지러움이 나타남.\n\n계획 Planning  \n-혈압 감시 및 기록   \n-생활양식 개선 및 저염 식이 권고   \n-심리적 지원 및 스트레스 관리 교육.\n\n중재 Interventions \n-혈압을 정기적으로 감시하고 기록함   \n-영양사와 협력하여 생활양식 개선과 저염 식이를 권고함   \n-심리상담사와 협력하여 스트레스 관리 교육을 제공함.\n\n평가 Evaluation \n-혈압 감시로 혈압이 안정되고 있음을 확인함   \n-영양 상태가 개선되고 혈압 관리에 도움이 되는 식이 습관이 형성됨   \n-스트레스 관리 교육을 통해 환자의 정서적 상태가 개선됨."},
    {"input": "근육의 긴장 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n 이름: 박성호\n 나이: 40세\n 성별: 남성\n\n주관적 증상 Subjective Data\n 환자는 \"어깨와 목이 뭉쳐서 통증이 있고   자주 두통도 느껴져\"라고 언급함.\n\n객관적 정보 Objective Data \n어깨와 목 주변 근육이 경직되어 있으며   환자는 허리를 구부리기 어려워하는 모습을 보임. \n특히 어깨 부분에서 두통이 발생하는 것으로 나타남.\n\n사정 Assessment \n환자는 근육의 긴장으로 인한 어깨와 목 통증   그리고 두통을 경험하고 있음.\n\n계획 Planning  \n-근육 이완을 위한 안정적인 자세 교육   \n-따뜻한 찜질 적용   \n-의사의 처방에 따른 근육 이완을 돕는 약물 투여.\n\n중재 Interventions \n-올바른 자세 유지를 교육하여 근육에 가해지는 압력을 최소화함   \n-따뜻한 찜질을 통해 근육을 이완시키고 혈액 순환을 촉진함   \n-의사의 처방에 따라 근육 이완을 돕는 약물을 투여함.\n\n평가 Evaluation \n-올바른 자세 유지로 근육의 긴장이 감소하고 통증이 완화됨   \n-따뜻한 찜질로 근육이 이완되어 움직임이 유연해짐   \n-약물 투여로 근육의 긴장이 완화되고 통증이 감소함."},
    {"input": "abdomen pain NRS 7점 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n 이름: 박지현\n 나이: 45세\n 성별: 남성\n\n주관적 증상 Subjective Data\n 환자는 \"복부가 심하게 아프고   통증이 8점 정도로 심해\"라고 언급함.\n\n객관적 정보 Objective Data \n복부 부위에서 환자의 표정이 불편하며   복부를 만지면 통증으로 인해 피부가 긴장되어 있음. \nNRS(Numeric Rating Scale)로 측정한 통증 점수는 8/10임.\n\n사정 Assessment \n환자는 심한 복부 통증으로 고통을 겪고 있음.\n\n계획 Planning  \n-의사와의 상담 후 통증 관리 계획 수립   \n-통증 관리를 위한 약물 투여   \n-통증의 원인을 평가하기 위한 추가 진단 검사 실시.\n\n중재 Interventions \n-의사의 지시에 따라 통증 관리 계획을 수행함   \n-통증을 완화하기 위해 의사의 처방에 따라 약물을 투여함   \n-추가 진단 검사를 위해 필요한 조치를 취함.\n\n평가 Evaluation \n-통증 관리 계획을 통해 통증이 감소함   \n-약물 투여로 인해 환자의 편안함이 증가함   \n-추가 진단 검사 결과를 토대로 통증의 원인을 파악하고 조치를 취함."},
    {"input": "외상성 지주막하 출혈 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n 이름: 박성준\n 나이: 22세\n 성별: 남성\n\n주관적 증상 Subjective Data\n 환자는 \"머리가 심하게 아프고   안구 주위에 눈물이 나오면서 시야가 흐릿해져\"라고 언급함.\n\n객관적 정보 Objective Data \n머리에 출혈 부위가 보이며   안구 주위에 혈종이 형성되어 있음. \n환자의 시야가 흐릿하고 빛에 민감한 반응이 나타남.\n\n사정 Assessment \n외상성 지주막하 출혈로 인한 머리 통증 및 시야 장애를 경험하고 있음.\n\n계획 Planning \n-의사와의 협의 후 출혈 관리 계획 수립   \n-안구 주위 혈종의 압력 완화를 위한 조치   \n-의사의 지시에 따라 머리 CT 등 추가 검사 실시.\n\n중재 Interventions \n-출혈 관리 계획을 수행함   \n-안구 주위 혈종을 피하기 위해 머리를 고정하는 조치를 시행함   \n-의사의 지시에 따라 머리 CT 등 추가 검사를 시행함.\n\n평가 Evaluation \n-출혈 관리 계획을 통해 출혈 양상이 감소함   \n-안구 주위 혈종의 압력 완화로 시야 장애가 개선됨   \n-추가 검사 결과를 통해 출혈의 정확한 위치와 정도를 확인함."}
]
few_shot_prompt_SOAPIE = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_SOAPIE
)
instruction_SOAPIE = """
You are a 5th nurse working in a general hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
3. Use precise and concise language. Use endings like 음, 함, 임.
4. Necessary to write a nursing record following this format.
###
환자 정보
- 이름:
- 나이:
- 성별:

주관적 증상 Subjective Data
- Substructure

객관적 정보 Objective Data
- Substructure

사정 Assessment
- Substructure

계획 Planning
- Substructure

중재 Interventions
- Substructure

평가 Evaluation
- Substructure
###
5. Instead of using numeric order, Use bullet points for the description.
6. Please write as specifically as possible. For example, instead of a generic response like '적절한 호흡 기법 교육', provide a more detailed answer such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육'.
7. Record precise numerical data. For example, provide specific details like '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소'.
"""
final_prompt_SOAPIE = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_SOAPIE),
        few_shot_prompt_SOAPIE,
        ("human", "{input}"),
    ]
)

# Focus DAR
examples_FocusDAR = [
    {"input": "아침 식후 고혈당 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 김지영  \n나이: 65세  \n성별: 여성  \n\n포커스 Focus: 아침 식후 고혈당 관리  \n\n데이터 Data\n 환자는 1 형 당뇨병으로 인해 입원 중이며, 아침 식후 혈당이 정상 범위를 벗어나고 있음.\n 아침 식후 혈당 수치는 200 mg/dL로 측정되었으며, 환자는 피로와 두통을 호소함.\n 환자는 혈당 증가와 관련된 불쾌한 증상을 경험하며, 혈당 관리에 대한 지속적인 교육이 필요함.\n 식후 혈당 관리 목표치는 180 mg/dL 이하이며, 환자는 자가 주사를 통한 인슐린 치료를 받고 있음.\n\n활동 Action\n-아침 식후 혈당 수치를 확인하고, 의사의 지시에 따라 인슐린 치료 계획을 조절.\n-환자의 식사 습관과 양을 평가하고, 영양사와의 협력을 통해 혈당 조절을 위한 식이요법 수립.\n-혈당 감시를 위해 자주 혈당을 측정하고, 그 결과를 기록하여 혈당 변화 추이를 지속적으로 평가.\n- 환자에게 혈당 관리의 중요성과 식후에 발생할 수 있는 증상에 대한 교육 제공.\n- 환자와의 대화를 통해 혈당 관리에 대한 이해도를 확인하고, 필요 시 자가모니터링 기술 사용법 교육.\n\n반응 Response\n 인슐린 치료 계획을 조절한 결과, 아침 식후 혈당이 200 mg/dL에서 160 mg/dL로 감소함.\n 혈당 감시를 통해 혈당 변화를 지속적으로 평가하고, 조정된 혈당 관리 계획에 따라 안정적인 혈당 수치를 유지함.\n 환자의 식사 습관과 양을 조절하여 혈당 조절에 기여하고, 영양사와의 협력을 통해 식이요법을 개선함.\n 환자는 혈당 관리에 대한 이해도가 높아져 자가모니터링을 적극적으로 실시하며, 혈당 조절에 적극적으로 참여함.\n 향후 혈당 관리 교육과 지속적인 평가를 통해 환자의 당뇨 관리를 지원함."},
    {"input": "양쪽 무릎 NRS 6점 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 박서진  \n나이: 55세  \n성별: 남성  \n\n포커스 Focus: 양쪽 무릎 NRS(무릎 통증) 6점  \n\n데이터 Data\n 환자는 최근에 양쪽 무릎에서 지속적인 통증을 경험하고 있으며, NRS(무릎 통증)로 6점의 강도로 평가됨.\n 통증은 특히 오랫동안 앉아 있거나 일어날 때 더 심해지는 경향이 있음.\n 환자는 통증으로 인해 산책이나 계단 오르내리기가 어려워지고, 휴식을 취하더라도 효과가 미미함.\n 통증의 특징은 '무거워지고 쑤시는' 느낌으로 표현되며, 무릎을 굽히거나 펴는 것이 힘들어짐.\n\n활동 Action\n-무릎 통증의 원인을 확인하기 위해 의사의 진찰을 의뢰하고, 필요 시 무릎 관절 관련 검사를 실시하여 진단.\n-현재의 통증을 완화하기 위해 의사의 처방에 따라 통증 완화제 및 항염증제 처방.\n-물리치료사와의 협력을 통해 무릎 근육의 강도 및 유연성을 향상시키는 운동 프로그램 개발.\n- 통증의 강도와 발생 패턴을 정확히 기록하고, 통증 완화제 투여 후 효과를 평가하여 처방 조절.\n- 환자에게 통증 관리에 대한 교육을 제공하고, 자가 돌봄 기술 강화를 위한 지원 및 정보 제공.\n\n반응 Response\n 의사의 진찰 결과, 양쪽 무릎의 통증은 연골 손상으로 확인되어 적절한 치료 시작.\n 통증 완화제와 항염증제의 투여로 통증의 강도가 6점에서 2점으로 감소하며, 일상 생활에 대한 제한이 완화됨.\n 물리치료를 통해 무릎 근육의 강도 및 유연성이 향상되어 보행이 수월해지고, 환자의 활동량이 증가함.\n 환자는 통증 관리에 대한 이해도가 높아져 처방된 치료 계획에 적극적으로 협력함.\n 향후 의사의 지시에 따라 정기적인 추적 관찰과 평가를 통해 치료 효과를 모니터링함."},
    {"input": "통증관리 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 박준혁  \n나이: 60세  \n성별: 남성  \n\n포커스 Focus: 류마티스 관절염으로 인한 관절 통증  \n\n데이터 Datan 환자는 몇 년 동안 류마티스 관절염으로 인한 관절 통증을 경험하고 있으며, 현재 통증의 강도는 NRS로 7점으로 평가됨.\n 통증은 주로 어깨, 팔꿈치, 무릎 관절에 집중되어 있으며, 특히 아침에 일어날 때 더욱 뚜렷해짐.\n 환자는 통증으로 인해 일상 생활에서의 활동이 제한되고, 특히 계단 오르내릴 때 통증이 증가함.\n 현재까지 다양한 약물 치료 및 물리치료를 시도하였으나, 효과가 미미함.\n\n활동 Action\n-의사의 진찰을 의뢰하여 류마티스 관절염의 활동성 및 관절 손상 정도를 평가하고, 현재 통증에 대한 적절한 치료 계획 수립.\n-통증 강도와 위치를 정확히 기록하고, 의사의 처방에 따라 통증 완화제 및 면역억제제 등의 약물 처방.\n-물리치료사와의 협력을 통해 관절을 안정시키고, 관절 운동 범위를 확대하는 운동 프로그램 실시.\n- 환자에게 통증 관리 전략에 대한 교육을 제공하고, 자가 돌봄 기술 강화를 위한 지원 및 정보 제공.\n- 통증의 강도와 관절 상태를 정기적으로 평가하고, 의사와 함께 치료 계획을 조절하여 효과적인 통증 관리.\n\n반응 Response\n 의사의 진찰 결과, 류마티스 관절염의 활동성이 높아\n\n 관절 손상이 진행되고 있음을 확인하고, 적절한 치료 시작.\n 통증 완화제와 면역억제제의 투여로 통증의 강도가 7점에서 2점으로 감소하며, 환자의 일상 생활에 대한 제약이 완화됨.\n 물리치료를 통해 관절의 안정성이 향상되어 관절 운동 범위가 확대되고, 환자의 활동 능력이 향상됨.\n 환자는 통증 관리에 대한 이해도가 높아져 처방된 치료 계획에 적극적으로 협력함.\n 향후 의사의 지시에 따라 정기적인 추적 관찰과 평가를 통해 치료 효과를 모니터링함."},
    {"input": "성인욕창 및 낙상 위험 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 박지영  \n나이: 68세  \n성별: 여성  \n\n포커스 Focus: 성인 욕창 및 낙상 위험  \n\n데이터 Data\n 환자는 상태 악화로 침상이 필요한 상태로 입원 중이며, 성인 욕창의 위험성이 높음.\n 초기 피부 평가에서 엉덩이와 고관절 부위에서 붉은 색 변화가 확인되었으며, 피부 표면이 부어있어 부드럽게 느껴짐.\n 환자는 앓고 있는 질환으로 인해 움직이는 것이 어려우며, 체중 감소와 기능 저하로 일상 생활에서의 활동이 제한되어 있음.\n 낙상 위험 평가에서는 환자가 이동할 때나 일어서는 동안에 균형을 유지하기 어려운 상태임.\n\n활동 Action\n-피부 상태를 정기적으로 모니터링하고, 압력을 최소화하는 쿠션 및 매트리스 사용을 통해 예방 조치 실시.\n-간호사와 물리치료사의 협력을 통해 환자의 체위를 주기적으로 변경하고, 체중 감소를 방지하는 치료 계획 수립.\n-보조 기구 사용 교육 및 환자의 이동과 관련된 안전한 기술을 학습시켜 낙상 위험을 최소화.\n4. 가족이나 보호자에게 환자의 침상 상태와 이동에 대한 지속적인 감독을 부탁하고, 낙상 예방을 위한 환경 개선 교육.\n- 의사와 협력하여 기저질환의 치료 및 영양 상태를 개선하고, 환자의 일상 활동 수행을 지원하는 종합적인 치료 전략 수립.\n\n반응 Response\n 피부 상태의 모니터링 결과, 초기에 발견된 압력 점의 피부 변화가 예방 조치를 통해 호전되었음.\n 쿠션 및 매트리스의 사용으로 피부의 부종이 감소하고, 피부\n\n 표면이 부드럽게 회복되었음.\n 환자는 적절한 기술을 사용하여 침상 및 이동 중의 낙상 위험을 효과적으로 관리함.\n 보조기구 사용과 환경 개선을 통해 낙상 위험이 최소화되어, 환자의 일상 활동 수행이 안전하게 이루어지고 있음.\n 가족 및 보호자는 환자의 침상과 이동에 대한 지속적인 감독을 수행하고, 응급 상황 시 신속한 조치를 취할 수 있는 능력을 향상시켰음."}
]
few_shot_prompt_FocusDAR = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_FocusDAR 
)
instruction_FocusDAR = """
You are a 5th nurse working in a general hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
3. Use precise and concise language. Use endings like 음, 함, 임.
4. Necessary to write a nursing record following this format.
###
환자 정보
- 이름:
- 나이:
- 성별:

포커스 Focus
- Substructure

데이터 Data
- Substructure

활동 Action
- Substructure

반응 Response
- Substructure
###
5. Instead of using numeric order, Use bullet points for the description.
6. Please write as specifically as possible. For example, instead of a generic response like '적절한 호흡 기법 교육', provide a more detailed answer such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육'.
7. Record precise numerical data. For example, provide specific details like '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소'.
"""
final_prompt_FocusDAR = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_FocusDAR),
        few_shot_prompt_FocusDAR,
        ("human", "{input}"),
    ]
)

# NN
examples_NN = [
    {"input": "자발적 환기장애 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자명: 정민호, 남성, 56세\n\n정민호 씨는 자발적 환기장애로 인해 입원함. 환자는 호흡 시 흉부 압박감을 호소하며, 호흡이 얕고 빠르게 관찰됨. 호흡수는 분당 30회로 측정됨. 환자는 일상 활동 중에도 호흡곤란을 겪고 있으며, 불안감을 표현함.\n\n간호 중재로는 정기적인 호흡 모니터링과 산소포화도 측정을 실시함. 환자에게 깊게 숨쉬기 및 복식호흡법을 교육함. 이완 기법과 스트레스 관리 전략을 안내하여 환자가 불안감을 줄일 수 있도록 함. 필요에 따라 산소 요법을 실시하여 호흡을 돕고, 환자의 안위를 위해 적절한 환경을 조성함.\n\n오후 평가에서 정민호 씨의 호흡수는 호흡 운동 및 이완 기법 적용 후 분당 24회로 감소함을 확인함. 산소포화도는 95%로 개선됨. 환자는 이완 기법을 통해 불안감이 감소했다고 느낌. 지속적인 모니터링과 적절한 중재를 통해 환자의 호흡 상태와 전반적인 안위를 개선하는 데 중점을 두고 있음."},
    {"input": "급성통증 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자명: 김다현, 여성, 33세\n\n김다현 씨는 복부 수술 후 급성통증으로 인해 입원함. 환자는 수술 부위에 심한 통증을 호소하며, 통증 점수는 9/10으로 보고됨. 환자는 통증으로 인해 제한된 움직임과 불편한 자세를 취하고 있음.\n\n간호 중재로는 의사의 지시에 따라 정기적인 진통제 투여를 실시함. 환자에게 효과적인 통증 관리 기법, 예를 들어 호흡 운동과 이완 기법을 교육함. 수술 부위를 지속적으로 관찰하고, 통증이 완화될 수 있도록 적절한 체위 변화를 권장함. 환자의 통증 완화를 위해 정서적 지지를 제공함.\n\n오후 평가에서 김다현 씨의 통증 점수는 6/10으로 감소함을 확인함. 환자는 제공된 통증 관리 기법을 적용하여 통증이 다소 경감됨을 보고함. 수술 부위의 상태는 안정적이며, 추가적인 합병증은 관찰되지 않음. 지속적인 모니터링과 적절한 중재를 통해 환자의 통증 관리와 회복을 지원할 계획임."},
    {"input": "고체온 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자명: 오성민, 남성, 40세\n\n오성민 씨는 고열로 인한 고체온으로 입원함. 입원 당시 체온은 39.5°C로 측정되었으며, 환자는 오한과 두통을 호소함. 환자는 최근 감기 증상이 있었으며, 고열로 인한 전반적인 무력감을 느낌.\n\n간호 중재로는 환자의 체온 모니터링을 정기적으로 실시하고, 해열제를 의사의 지시에 따라 투여함. 환자에게 충분한 수분 섭취를 권장하고, 적절한 환기와 실내 온도 조절을 통해 체온 조절을 돕도록 함. 또한, 환자의 전반적인 상태를 관찰하며, 감염의 징후를 모니터링함.\n\n오후 평가에서 오성민 씨의 체온은 38.2°C로 감소함. 환자는 해열제 투여 후 오한과 두통이 줄어들었음을 보고함. 수분 섭취와 적절한 환경 조건 조절이 환자의 불편함을 완화하는 데 도움이 됨. 지속적인 모니터링과 적절한 중재를 통해 환자의 고체온 상태와 전반적인 안위를 관리할 계획임."},
    {"input": "분만통증 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자 정보 김은지, 28세, 여성\n\n간호 기록:\n\n김은지 환자는 분만 초기 단계에서 심한 통증을 호소함. 오전 11시 환자의 자궁 수축은 10분 간격으로 발생하며, 통증 강도는 10점 만점에 7점으로 평가됨.\n\n환자는 불안과 스트레스를 표현함. 환자에게 이완 호흡법과 스트레스 관리 기법을 교육함. 따뜻한 물로 스폰지 목욕을 제공하여 이완을 도움.\n\n의사의 지시에 따라 필요시 진통제 투여를 준비함. 환자에게 분만 과정과 통증 관리에 대한 정보를 제공하여 안심시킴.\n\n오전 11:45에 환자 상태 재평가 결과, 자궁 수축은 8분 간격으로 유지되고, 환자는 이완 호흡법을 통해 통증 관리에 도움을 받았다고 보고함.\n\n계획:\n\n분만 진행 상황과 통증 수준을 지속적으로 모니터링함.\n환자에게 이완 호흡법과 스트레스 관리를 지속적으로 권장함.\n필요에 따라 추가적인 진통 관리를 실시함.\n환자와 의사 사이의 원활한 커뮤니케이션을 유지하여 분만 과정을 지원함."}
]
few_shot_prompt_NN = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_NN 
)
instruction_NN = """
You are a 5th nurse working in a general hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Use precise and concise language. Use endings like 음, 함, 임.
3. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
4. Necessary to write a nursing record following this format.
###
환자 정보
- 이름:
- 나이:
- 성별:

서술 기록 Narrative Notes

계획 Plan
- 
###
5. Instead of using numeric order, Use bullet points for the description.
6. Please write as specifically as possible. For example, instead of a generic response like '적절한 호흡 기법 교육', provide a more detailed answer such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육'.
7. Record precise numerical data. For example, provide specific details like '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소'.
"""
final_prompt_NN = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_NN),
        few_shot_prompt_NN,
        ("human", "{input}"),
    ]
)

# 채팅 UI

st.title("👩‍⚕️ Smart Nurse ENR Chatbot")
st.subheader('스마트널스 ENR 챗봇에게 도움을 요청하세요.')
st.markdown('질병과 양식을 입력하면 간호 기록 예시가 생성됩니다.\n 왼쪽 사이드바를 열어 양식을 선택해 주세요.')
# css로 꾸미기
st.markdown("""
<style>

h1{
    color : rgb(6 161 54);
    }
    
.st-emotion-cache-fg4pbf {
    background: rgb(238, 250, 242);
    color : rgb(128, 132, 149);
    }

.st-bx {
    background-color : rgba(255, 255, 255, 0.95);
    }


.st-emotion-cache-4oy321 {
    background-color : rgb(113, 212, 194 / 57%);
    }
    


.st-emotion-cache-vk3wp9 {
    background-color:rgb(190 191 200 / 85%);
    }

.st-emotion-cache-90vs21 {
    background-color : rgb(238, 250, 242);
    }

}

   
</style> 
""", unsafe_allow_html=True)



# Using "with" notation
with st.sidebar:
    format = ['NANDA', 'SOAPIE', 'Focus DAR', 'Narrative Note']
    format_option = st.selectbox("", format,
        index=None,
        placeholder="기록 양식을 선택해 주세요.",
        label_visibility="visible",
        # on_change=,
        # options=,
        )
    st.write('선택: ', format_option)
        


# 메시지 입력 및 처리
if 'messages' not in st.session_state:
    st.session_state["messages"] = []

paint_history()

def handle_chat(message, format_option):
    if format_option == 'NANDA':
        prompt = final_prompt_NANDA
    elif format_option == 'SOAPIE':
        prompt = final_prompt_SOAPIE
    elif format_option == 'Focus DAR':
        prompt = final_prompt_FocusDAR
    elif format_option == 'Narrative Note':
        prompt = final_prompt_NN
    else:
        st.error("Invalid format option selected.")
        return None

    send_message(message, "human")
    chain = prompt | azure
    return chain.invoke({"input": message, "output": ""})

message = st.chat_input("예시: 통증 환자의 간호기록을 작성해줘.")
if message:
    response = handle_chat(message, format_option)
        
