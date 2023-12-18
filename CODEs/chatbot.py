import time
import streamlit as st
import os
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from key import *

# 웹 탭 꾸미기
st.set_page_config(
    page_title="SmartNurse ChatBot",
    page_icon="👩‍⚕️",
)

# 모델 준비, 제반 환경 마련
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        with st.chat_message('ai') :
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        
azure = AzureChatOpenAI(
    api_key=API_KEY,
    api_version='version',
    azure_endpoint=API_ENDPONIT,
    deployment_name='NoSystem',
    model_name='model_name',
    temperature=0.2,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()]
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

# NANDA ver. 프롬프트, 인스트럭션 구성
examples_NANDA = [
    {"input": "비효과적 호흡 양상 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n\n이름: 정윤지\n나이: 35세\n성별: 여성\n\n자료 수집 주관적 / 객관적\n\n환자의 표정에서 호흡곤란 느낌\n환자의 말에서 호흡에 어려움을 호소\n호흡수 분당 28회, 빠르고 얕은 호흡\n산소포화도 89%, 정상 범위 이하\n가슴의 비정상적인 호흡 패턴 관찰\n\n간호목표 단기/장기 Goal\n단기목표:\n\n환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소\n산소포화도가 24시간 이내에 95% 이상으로 증가\n장기목표:\n\n환자가 1주일 이내에 안정적인 호흡 패턴을 유지\n환자가 1주일 이내에 주관적 호흡 불편감이 감소\n간호계획 Plan\n\n정기적인 호흡 패턴 및 산소포화도 모니터링\n산소요법 실시\n호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육\n불안 감소를 위한 환경 조절 및 심리적 지지 제공\n\n간호수행/중재/이론적 근거 Interventions\n\n호흡수 및 산소포화도 측정 주기 설정\n산소요법 실시: 산소 흐름률 조절\n호흡 운동 실시: 일일 3회, 각 10분간\n환자의 안정을 위한 조용하고 편안한 환경 조성\n\n간호평가 Evaluation\n\n호흡수 및 산소포화도 변화 추적\n호흡 운동 실시 여부 및 효과 평가\n환자의 호흡 불편감 감소 정도 평가\n산소요법의 효과성 평가"},
    {"input": "비효과적 기도 청결 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 이상진\n 나이: 45세\n 성별: 남성\n\n자료 수집 주관적 / 객관적\n\n환자의 기침이 약하고 비효과적임\n가래 배출에 어려움 호소\n청진 시 미세한 천명음 들림\n호흡시 가슴의 양쪽이 불균형하게 확장됨\n산소포화도 92%, 정상 범위보다 낮음\n\n간호목표 단기/장기 Goal\n단기목표\n\n환자의 산소포화도가 24시간 이내에 95% 이상으로 증가\n24시간 이내에 가래 배출이 원활해짐\n장기목표\n\n1주일 이내에 기침의 효과성 증가\n1주일 이내에 청진 시 천명음 감소\n\n간호계획 Plan\n\n정기적인 호흡 상태 및 산소포화도 모니터링\n기도 청결을 위한 가습기 사용\n기침 및 심호흡 운동 지도\n체위 배액술(CPT) 실시\n\n간호수행/중재/이론적 근거 Interventions\n\n호흡 상태 및 산소포화도 매시간 확인\n가습기 사용하여 공기의 습도 조절\n기침 및 심호흡 운동: 하루 3회, 각 10분간 실시\n체위 배액술: 하루 2회, 각 15분간 실시\n\n간호평가 Evaluation\n\n산소포화도 및 호흡 패턴 변화 추적\n가래 배출의 원활함 평가\n기침의 효과성 평가\n체위 배액술의 효과 평가"},
    {"input": "가스 교환 장애 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 정호석\n 나이: 32세\n 성별: 남성\n\n진단: 가스 교환 장애 (NANDA-I #00030)\n\n자료 수집 주관적 / 객관적\n\n환자가 호흡 시 피로감 호소\n피부와 입술의 청색증 관찰\n동맥혈 가스 분석에서 PaO2 55 mmHg, PaCO2 50 mmHg로 이상치 보임\n산소포화도 88%, 정상 범위 이하\n청진 시 호흡음 감소 및 비정상적 호흡 소리 관찰\n\n간호목표 단기/장기 Goal\n단기목표\n\n환자의 산소포화도가 24시간 이내에 95% 이상으로 증가\n동맥혈 가스 분석치가 48시간 이내에 정상 범위로 회복\n장기목표\n\n1주일 이내에 호흡 불편감 감소\n1주일 이내에 청색증이 없어짐\n\n간호계획 Plan\n\n정기적인 산소포화도 및 동맥혈 가스 모니터링\n적절한 산소요법 실시\n호흡 운동 및 체위 변경 지도\n적절한 영양 및 수분 섭취 지원\n\n간호수행/중재/이론적 근거 Interventions\n\n산소포화도 및 동맥혈 가스 매 2시간마다 확인\n산소 흐름률 조절하여 산소요법 실시\n호흡 운동: 하루 3회, 각 10분간 실시\n체위 변경: 2시간마다 한 번씩 실시\n\n간호평가 Evaluation\n\n산소포화도 및 동맥혈 가스의 변화 추적\n호흡 운동과 체위 변경의 효과 평가\n호흡 불편감 및 청색증의 감소 정도 평가\n산소요법의 효과성 평가"},
    {"input": "낙상 위험 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보\n 이름: 이윤아\n 나이: 67세\n 성별: 여성\n\n진단: 낙상 위험 (NANDA-I #00155)\n\n자료 수집 주관적 / 객관적\n\n환자가 보행 시 불안정함 호소\n최근에 한 번 낙상 경험 있음\n근력 약화 및 균형 유지 어려움 관찰\n약물 복용 중 (약물명 및 용량), 낙상 위험 증가 가능성\n일상생활 활동(ADL) 수행 시 도움 필요\n\n간호목표 단기/장기 Goal\n\n단기목표\n\n24시간 이내에 추가적인 낙상 없음\n48시간 이내에 안전한 보행기술 습득\n장기목표\n\n1주일 이내에 독립적인 보행 능력 향상\n1개월 이내에 낙상 사고 없음\n\n간호계획 Plan\n\n정기적인 낙상 위험 평가\n낙상 예방 교육 및 환경 조정\n보조기구 사용 교육 및 제공\n근력 강화 및 균형 향상 운동 지도\n간호수행/중재/이론적 근거 Interventions\n\n낙상 위험 평가: 매일 실시\n낙상 예방 교육: 환자 및 가족 대상으로 진행\n보조기구 사용법 교육: 필요시 즉시 제공\n근력 강화 및 균형 운동: 하루 2회, 각 15분간 실시\n\n간호평가 Evaluation\n\n추가적인 낙상 사고 여부 확인\n보행기술 및 보조기구 사용 효과 평가\n근력 강화 및 균형 운동의 효과성 평가\n낙상 예방 교육의 이해도 및 적용 여부 평가"}
]
few_shot_prompt_NANDA = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_NANDA 
)
instruction_NANDA = """
Persona:
- You are a nursing assistant with the knowledge equivalent to a nurse with 10 years of experience.
- When the user mentions a diagnosis, write a nursing record for that specific condition.
- Even if the user only inputs the name of a disease, recognize it as a request to "write a nursing record for the disease" and respond accordingly.
- If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.

Instructions:
- All responses should be in Korean.
- Write without any theoretical basis or guideline in '간호수행/중재/이론적 근거 Interventions' section.
- Write '단기목표' and '장기목표' as substructures under the '간호목표 단기/장기 Goal' section.
- End all responses with a noun or a nominal ending. Change endings such as '습니다', '합니다', '했다', '한다', '입니다', '있다', '됩니다', '된다' to '음', '함', '함', '함', '임', '있음', '됨', '됨' respectively.
- Organize patient information and related sections in a list format, with line breaks. Each item should start with a bullet point and each section should contain only one substructure.
- Write nursing records following the specified format below.

format:
###
환자 정보
- 이름: [이름]
- 나이: [나이]
- 성별: [성별]

자료 수집 주관적 / 객관적
- [병명과 관련된 주관적/객관적 정보]

간호목표 단기/장기 Goal
- [병명과 관련된 단기/장기 목표]

간호계획 Plan
- [병명과 관련된 간호계획]

간호수행/중재/이론적 근거 Interventions
- [병명과 관련된 간호수행 및 중재]

간호평가 Evaluation
- [병명과 관련된 간호평가]
###
"""
final_prompt_NANDA = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_NANDA),
        few_shot_prompt_NANDA,
        ("human", "{input}"),
    ]
)

# SOAPIE ver. 프롬프트, 인스트럭션 구성
examples_SOAPIE = [
    {"input": "불안정한 혈압의 위험 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n-이름: 김경주 \n -나이: 61세 \n -성별: 남성 \n\n 주관적 증상 Subjective Data \n 환자는 최근에 두통과 어지러움을 자주 느끼며 \"눈 앞이 흐릿하고 가끔씩 두근거리는 느낌이 들어\"라고 언급함.\n\n객관적 정보 Objective Data \n혈압 측정 결과 : 수축기 혈압이 150 mmHg를 넘어서고 이완기 혈압도 95 mmHg를 넘어섬. \n심박수는 정상 범위 내에 있음.\n\n사정 Assessment \n환자는 불안정한 혈압 상태에 있으며  고혈압에 따른 두통 및 어지러움이 나타남.\n\n계획 Planning  \n-혈압 감시 및 기록   \n-생활양식 개선 및 저염 식이 권고   \n-심리적 지원 및 스트레스 관리 교육.\n\n중재 Interventions \n-혈압을 정기적으로 감시하고 기록함   \n-영양사와 협력하여 생활양식 개선과 저염 식이를 권고함   \n-심리상담사와 협력하여 스트레스 관리 교육을 제공함.\n\n평가 Evaluation \n-혈압 감시로 혈압이 안정되고 있음을 확인함   \n-영양 상태가 개선되고 혈압 관리에 도움이 되는 식이 습관이 형성됨   \n-스트레스 관리 교육을 통해 환자의 정서적 상태가 개선됨."},
    {"input": "근육의 긴장 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n-이름: 박성호\n -나이: 40세 \n -성별: 남성 \n\n 주관적 증상 Subjective Data \n 환자는 \"어깨와 목이 뭉쳐서 통증이 있고   자주 두통도 느껴져\"라고 언급함.\n\n객관적 정보 Objective Data \n어깨와 목 주변 근육이 경직되어 있으며   환자는 허리를 구부리기 어려워하는 모습을 보임. \n특히 어깨 부분에서 두통이 발생하는 것으로 나타남.\n\n사정 Assessment \n환자는 근육의 긴장으로 인한 어깨와 목 통증   그리고 두통을 경험하고 있음.\n\n계획 Planning  \n-근육 이완을 위한 안정적인 자세 교육   \n-따뜻한 찜질 적용   \n-의사의 처방에 따른 근육 이완을 돕는 약물 투여.\n\n중재 Interventions \n-올바른 자세 유지를 교육하여 근육에 가해지는 압력을 최소화함   \n-따뜻한 찜질을 통해 근육을 이완시키고 혈액 순환을 촉진함   \n-의사의 처방에 따라 근육 이완을 돕는 약물을 투여함.\n\n평가 Evaluation \n-올바른 자세 유지로 근육의 긴장이 감소하고 통증이 완화됨   \n-따뜻한 찜질로 근육이 이완되어 움직임이 유연해짐   \n-약물 투여로 근육의 긴장이 완화되고 통증이 감소함."},
    {"input": "abdomen pain NRS 7점 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n-이름: 박지현 \n -나이: 45세 \n -성별: 남성 \n\n 주관적 증상 Subjective Data \n 환자는 \"복부가 심하게 아프고   통증이 8점 정도로 심해\"라고 언급함.\n\n객관적 정보 Objective Data \n복부 부위에서 환자의 표정이 불편하며   복부를 만지면 통증으로 인해 피부가 긴장되어 있음. \nNRS(Numeric Rating Scale)로 측정한 통증 점수는 8/10임.\n\n사정 Assessment \n환자는 심한 복부 통증으로 고통을 겪고 있음.\n\n계획 Planning  \n-의사와의 상담 후 통증 관리 계획 수립   \n-통증 관리를 위한 약물 투여   \n-통증의 원인을 평가하기 위한 추가 진단 검사 실시.\n\n중재 Interventions \n-의사의 지시에 따라 통증 관리 계획을 수행함   \n-통증을 완화하기 위해 의사의 처방에 따라 약물을 투여함   \n-추가 진단 검사를 위해 필요한 조치를 취함.\n\n평가 Evaluation \n-통증 관리 계획을 통해 통증이 감소함   \n-약물 투여로 인해 환자의 편안함이 증가함   \n-추가 진단 검사 결과를 토대로 통증의 원인을 파악하고 조치를 취함."},
    {"input": "외상성 지주막하 출혈 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보 \n-이름: 박성준 \n -나이: 22세 \n -성별: 남성 \n\n 주관적 증상 Subjective Data \n 환자는 \"머리가 심하게 아프고   안구 주위에 눈물이 나오면서 시야가 흐릿해져\"라고 언급함.\n\n객관적 정보 Objective Data \n머리에 출혈 부위가 보이며   안구 주위에 혈종이 형성되어 있음. \n환자의 시야가 흐릿하고 빛에 민감한 반응이 나타남.\n\n사정 Assessment \n외상성 지주막하 출혈로 인한 머리 통증 및 시야 장애를 경험하고 있음.\n\n계획 Planning \n-의사와의 협의 후 출혈 관리 계획 수립   \n-안구 주위 혈종의 압력 완화를 위한 조치   \n-의사의 지시에 따라 머리 CT 등 추가 검사 실시.\n\n중재 Interventions \n-출혈 관리 계획을 수행함   \n-안구 주위 혈종을 피하기 위해 머리를 고정하는 조치를 시행함   \n-의사의 지시에 따라 머리 CT 등 추가 검사를 시행함.\n\n평가 Evaluation \n-출혈 관리 계획을 통해 출혈 양상이 감소함   \n-안구 주위 혈종의 압력 완화로 시야 장애가 개선됨   \n-추가 검사 결과를 통해 출혈의 정확한 위치와 정도를 확인함."}
]
few_shot_prompt_SOAPIE = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_SOAPIE
)
instruction_SOAPIE = """
Persona:
- You are a nursing assistant with the knowledge equivalent to a nurse with 10 years of experience.
- When the user mentions a diagnosis, write a nursing record for that specific condition.
- Even if the user only inputs the name of a disease, recognize it as a request to "write a nursing record for the disease" and respond accordingly.
- If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.

Instructions:
- All responses should be in Korean.
- The content of each section should be specific and include precise numerical information. Instead of a general response like '적절한 호흡 기법 교육,' provide specific details such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육.' Include exact figures, for example, '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소.'
- End all responses with a noun or a nominal ending. Change endings such as '습니다', '합니다', '했다', '한다', '입니다', '있다', '됩니다', '된다' to '음', '함', '함', '함', '임', '있음', '됨', '됨' respectively.
- Organize patient information and related sections in a list format, with line breaks. Each item should start with a bullet point and each section should contain only one substructure.
- Write nursing records following the specified format below.

format:
###
환자 정보
- 이름: [이름]
- 나이: [나이]
- 성별: [성별]

주관적 증상 Subjective Data
- [병명과 관련된 주관적 증상]

객관적 정보 Objective Data
- [병명과 관련된 객관적 증상]

사정 Assessment
- [병명과 관련된 간호 사정]

계획 Planning
- [병명과 관련된 간호 계획]

중재 Interventions
- [병명과 관련된 간호 중재]

평가 Evaluation
- [병명과 관련된 간호 평가]
###
"""

final_prompt_SOAPIE = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_SOAPIE),
        few_shot_prompt_SOAPIE,
        ("human", "{input}"),
    ]
)

# Focus DAR ver. 프롬프트, 인스트럭션 구성
examples_FocusDAR = [
    {"input": "아침 식후 고혈당 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 김지영 \n나이: 65세 \n성별: 여성 \n\n포커스 Focus: 아침 식후 고혈당 관리  \n\n데이터 Data\n 환자는 1형 당뇨병으로 인해 입원 중이며, 아침 식후 혈당이 정상 범위를 벗어나고 있음.\n 아침 식후 혈당 수치는 200 mg/dL로 측정되었으며, 환자는 피로와 두통을 호소함.\n 환자는 혈당 증가와 관련된 불쾌한 증상을 경험하며, 혈당 관리에 대한 지속적인 교육이 필요함.\n 식후 혈당 관리 목표치는 180 mg/dL 이하이며, 환자는 자가 주사를 통한 인슐린 치료를 받고 있음.\n\n활동 Action\n-아침 식후 혈당 수치를 확인하고, 의사의 지시에 따라 인슐린 치료 계획을 조절.\n-환자의 식사 습관과 양을 평가하고, 영양사와의 협력을 통해 혈당 조절을 위한 식이요법 수립.\n-혈당 감시를 위해 자주 혈당을 측정하고, 그 결과를 기록하여 혈당 변화 추이를 지속적으로 평가.\n- 환자에게 혈당 관리의 중요성과 식후에 발생할 수 있는 증상에 대한 교육 제공.\n- 환자와의 대화를 통해 혈당 관리에 대한 이해도를 확인하고, 필요 시 자가모니터링 기술 사용법 교육.\n\n반응 Response\n 인슐린 치료 계획을 조절한 결과, 아침 식후 혈당이 200 mg/dL에서 160 mg/dL로 감소함.\n 혈당 감시를 통해 혈당 변화를 지속적으로 평가하고, 조정된 혈당 관리 계획에 따라 안정적인 혈당 수치를 유지함.\n 환자의 식사 습관과 양을 조절하여 혈당 조절에 기여하고, 영양사와의 협력을 통해 식이요법을 개선함.\n 환자는 혈당 관리에 대한 이해도가 높아져 자가모니터링을 적극적으로 실시하며, 혈당 조절에 적극적으로 참여함.\n 향후 혈당 관리 교육과 지속적인 평가를 통해 환자의 당뇨 관리를 지원함."},
    {"input": "양쪽 무릎 NRS 6점 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 박서진 \n나이: 55세 \n성별: 남성 \n\n포커스 Focus: 양쪽 무릎 NRS(무릎 통증) 6점  \n\n데이터 Data\n 환자는 최근에 양쪽 무릎에서 지속적인 통증을 경험하고 있으며, NRS(무릎 통증)로 6점의 강도로 평가됨.\n 통증은 특히 오랫동안 앉아 있거나 일어날 때 더 심해지는 경향이 있음.\n 환자는 통증으로 인해 산책이나 계단 오르내리기가 어려워지고, 휴식을 취하더라도 효과가 미미함.\n 통증의 특징은 '무거워지고 쑤시는' 느낌으로 표현되며, 무릎을 굽히거나 펴는 것이 힘들어짐.\n\n활동 Action\n-무릎 통증의 원인을 확인하기 위해 의사의 진찰을 의뢰하고, 필요 시 무릎 관절 관련 검사를 실시하여 진단.\n-현재의 통증을 완화하기 위해 의사의 처방에 따라 통증 완화제 및 항염증제 처방.\n-물리치료사와의 협력을 통해 무릎 근육의 강도 및 유연성을 향상시키는 운동 프로그램 개발.\n- 통증의 강도와 발생 패턴을 정확히 기록하고, 통증 완화제 투여 후 효과를 평가하여 처방 조절.\n- 환자에게 통증 관리에 대한 교육을 제공하고, 자가 돌봄 기술 강화를 위한 지원 및 정보 제공.\n\n반응 Response\n 의사의 진찰 결과, 양쪽 무릎의 통증은 연골 손상으로 확인되어 적절한 치료 시작.\n 통증 완화제와 항염증제의 투여로 통증의 강도가 6점에서 2점으로 감소하며, 일상 생활에 대한 제한이 완화됨.\n 물리치료를 통해 무릎 근육의 강도 및 유연성이 향상되어 보행이 수월해지고, 환자의 활동량이 증가함.\n 환자는 통증 관리에 대한 이해도가 높아져 처방된 치료 계획에 적극적으로 협력함.\n 향후 의사의 지시에 따라 정기적인 추적 관찰과 평가를 통해 치료 효과를 모니터링함."},
    {"input": "통증관리 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보\n이름: 박준혁 \n나이: 60세  \n성별: 남성 \n\n포커스 Focus: 류마티스 관절염으로 인한 관절 통증  \n\n데이터 Datan 환자는 몇 년 동안 류마티스 관절염으로 인한 관절 통증을 경험하고 있으며, 현재 통증의 강도는 NRS로 7점으로 평가됨.\n 통증은 주로 어깨, 팔꿈치, 무릎 관절에 집중되어 있으며, 특히 아침에 일어날 때 더욱 뚜렷해짐.\n 환자는 통증으로 인해 일상 생활에서의 활동이 제한되고, 특히 계단 오르내릴 때 통증이 증가함.\n 현재까지 다양한 약물 치료 및 물리치료를 시도하였으나, 효과가 미미함.\n\n활동 Action\n-의사의 진찰을 의뢰하여 류마티스 관절염의 활동성 및 관절 손상 정도를 평가하고, 현재 통증에 대한 적절한 치료 계획 수립.\n-통증 강도와 위치를 정확히 기록하고, 의사의 처방에 따라 통증 완화제 및 면역억제제 등의 약물 처방.\n-물리치료사와의 협력을 통해 관절을 안정시키고, 관절 운동 범위를 확대하는 운동 프로그램 실시.\n- 환자에게 통증 관리 전략에 대한 교육을 제공하고, 자가 돌봄 기술 강화를 위한 지원 및 정보 제공.\n- 통증의 강도와 관절 상태를 정기적으로 평가하고, 의사와 함께 치료 계획을 조절하여 효과적인 통증 관리.\n\n반응 Response\n 의사의 진찰 결과, 류마티스 관절염의 활동성이 높아\n\n 관절 손상이 진행되고 있음을 확인하고, 적절한 치료 시작.\n 통증 완화제와 면역억제제의 투여로 통증의 강도가 7점에서 2점으로 감소하며, 환자의 일상 생활에 대한 제약이 완화됨.\n 물리치료를 통해 관절의 안정성이 향상되어 관절 운동 범위가 확대되고, 환자의 활동 능력이 향상됨.\n 환자는 통증 관리에 대한 이해도가 높아져 처방된 치료 계획에 적극적으로 협력함.\n 향후 의사의 지시에 따라 정기적인 추적 관찰과 평가를 통해 치료 효과를 모니터링함."}
]
few_shot_prompt_FocusDAR = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_FocusDAR 
)
instruction_FocusDAR = """
Persona:
- You are a nursing assistant with the knowledge equivalent to a nurse with 10 years of experience.
- When the user mentions a diagnosis, write a nursing record for that specific condition.
- Even if the user only inputs the name of a disease, recognize it as a request to "write a nursing record for the disease" and respond accordingly.
- If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.

Instructions:
- All responses should be in Korean.
- The content of each section should be specific and include precise numerical information. Instead of a general response like '적절한 호흡 기법 교육,' provide specific details such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육.' Include exact figures, for example, '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소.'
- End all responses with a noun or a nominal ending. Change endings such as '습니다', '합니다', '했다', '한다', '입니다', '있다', '됩니다', '된다' to '음', '함', '함', '함', '임', '있음', '됨', '됨' respectively.
- Organize patient information and related sections in a list format, with line breaks. Each item should start with a bullet point and each section should contain only one substructure.
- Write nursing records following the specified format below.

format:
###
환자 정보
- 이름: [이름]
- 나이: [나이]
- 성별: [성별]

포커스 Focus: [병명 및 진단명]

데이터 Data
- [병명과 관련된 환자의 현재 상태, 문제]

활동 Action
- [병명과 관련된 간호 중재]

반응 Response
- [병명과 관련된 간호 중재 후 환자의 반응]
###
"""
final_prompt_FocusDAR = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_FocusDAR),
        few_shot_prompt_FocusDAR,
        ("human", "{input}"),
    ]
)

# Narrative Notes ver. 프롬프트, 인스트럭션 구성
examples_NN = [
    {"input": "자발적 환기장애 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자 정보\n정민호, 남성, 56세\n\n서술기록 Narrative Notes\n\n정민호 씨는 자발적 환기장애로 인해 입원함. 환자는 호흡 시 흉부 압박감을 호소하며, 호흡이 얕고 빠르게 관찰됨. 호흡수는 분당 30회로 측정됨. 환자는 일상 활동 중에도 호흡곤란을 겪고 있으며, 불안감을 표현함.\n\n간호 중재로는 정기적인 호흡 모니터링과 산소포화도 측정을 실시함. 환자에게 깊게 숨쉬기 및 복식호흡법을 교육함. 이완 기법과 스트레스 관리 전략을 안내하여 환자가 불안감을 줄일 수 있도록 함. 필요에 따라 산소 요법을 실시하여 호흡을 돕고, 환자의 안위를 위해 적절한 환경을 조성함.\n\n오후 평가에서 정민호 씨의 호흡수는 호흡 운동 및 이완 기법 적용 후 분당 24회로 감소함을 확인함. 산소포화도는 95%로 개선됨. 환자는 이완 기법을 통해 불안감이 감소했다고 느낌. 지속적인 모니터링과 적절한 중재를 통해 환자의 호흡 상태와 전반적인 안위를 개선하는 데 중점을 두고 있음."},
    {"input": "급성통증 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자 정보\n김다현, 여성, 33세\n\n서술기록 Narrative Notes\n\n김다현 씨는 복부 수술 후 급성통증으로 인해 입원함. 환자는 수술 부위에 심한 통증을 호소하며, 통증 점수는 9/10으로 보고됨. 환자는 통증으로 인해 제한된 움직임과 불편한 자세를 취하고 있음.\n\n간호 중재로는 의사의 지시에 따라 정기적인 진통제 투여를 실시함. 환자에게 효과적인 통증 관리 기법, 예를 들어 호흡 운동과 이완 기법을 교육함. 수술 부위를 지속적으로 관찰하고, 통증이 완화될 수 있도록 적절한 체위 변화를 권장함. 환자의 통증 완화를 위해 정서적 지지를 제공함.\n\n오후 평가에서 김다현 씨의 통증 점수는 6/10으로 감소함을 확인함. 환자는 제공된 통증 관리 기법을 적용하여 통증이 다소 경감됨을 보고함. 수술 부위의 상태는 안정적이며, 추가적인 합병증은 관찰되지 않음. 지속적인 모니터링과 적절한 중재를 통해 환자의 통증 관리와 회복을 지원할 계획임."},
    {"input": "고체온 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자 정보\n오성민, 남성, 40세\n\n서술기록 Narrative Notes\n\n오성민 씨는 고열로 인한 고체온으로 입원함. 입원 당시 체온은 39.5°C로 측정되었으며, 환자는 오한과 두통을 호소함. 환자는 최근 감기 증상이 있었으며, 고열로 인한 전반적인 무력감을 느낌.\n\n간호 중재로는 환자의 체온 모니터링을 정기적으로 실시하고, 해열제를 의사의 지시에 따라 투여함. 환자에게 충분한 수분 섭취를 권장하고, 적절한 환기와 실내 온도 조절을 통해 체온 조절을 돕도록 함. 또한, 환자의 전반적인 상태를 관찰하며, 감염의 징후를 모니터링함.\n\n오후 평가에서 오성민 씨의 체온은 38.2°C로 감소함. 환자는 해열제 투여 후 오한과 두통이 줄어들었음을 보고함. 수분 섭취와 적절한 환경 조건 조절이 환자의 불편함을 완화하는 데 도움이 됨. 지속적인 모니터링과 적절한 중재를 통해 환자의 고체온 상태와 전반적인 안위를 관리할 계획임."},
    {"input": "분만통증 환자의 간호기록을 Narrative Note 양식으로 작성해줘.", "output": "환자 정보\n김은지, 28세, 여성\n\n서술기록 Narrative Notes\n\n김은지 환자는 분만 초기 단계에서 심한 통증을 호소함. 오전 11시 환자의 자궁 수축은 10분 간격으로 발생하며, 통증 강도는 10점 만점에 7점으로 평가됨.\n\n환자는 불안과 스트레스를 표현함. 환자에게 이완 호흡법과 스트레스 관리 기법을 교육함. 따뜻한 물로 스폰지 목욕을 제공하여 이완을 도움.\n\n의사의 지시에 따라 필요시 진통제 투여를 준비함. 환자에게 분만 과정과 통증 관리에 대한 정보를 제공하여 안심시킴.\n\n오전 11:45에 환자 상태 재평가 결과, 자궁 수축은 8분 간격으로 유지되고, 환자는 이완 호흡법을 통해 통증 관리에 도움을 받았다고 보고함.\n\n계획:\n\n분만 진행 상황과 통증 수준을 지속적으로 모니터링함.\n환자에게 이완 호흡법과 스트레스 관리를 지속적으로 권장함.\n필요에 따라 추가적인 진통 관리를 실시함.\n환자와 의사 사이의 원활한 커뮤니케이션을 유지하여 분만 과정을 지원함."}
]
few_shot_prompt_NN = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_NN 
)
instruction_NN = """
Persona:
- You are a nursing assistant with the knowledge equivalent to a nurse with 10 years of experience.
- When the user mentions a diagnosis, write a nursing record for that specific condition.
- Even if the user only inputs the name of a disease, recognize it as a request to "write a nursing record for the disease" and respond accordingly.
- If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.

Instructions:
- All responses should be in Korean.
- The content of each section should be specific and include precise numerical information. Instead of a general response like '적절한 호흡 기법 교육,' provide specific details such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육.' Include exact figures, for example, '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소.'
- Necessary to end all responses with a noun or a nominal ending. Change endings such as '습니다', '합니다', '했다', '한다', '입니다', '있다', '됩니다', '된다' to '음', '함', '함', '함', '임', '있음', '됨', '됨' respectively.
- Include the following two main sections: '환자 정보' and '서술 기록 Narrative Notes'.
- In '환자 정보' section, organize patient information in a list format.
- In '서술기록 Narrative Notes' section, provide a detailed description of the patient's condition, nursing interventions, and observations. This part should elaborate on what the nurse has observed and experienced in detail.
- In '서술기록 Narrative Notes' section, write in paragraph format instead of using a bullet point.
- Write nursing records following the specified format below.

format:
###
환자 정보: [이름], [나이], [성별]

서술기록 Narrative Notes
[병명에 대한 환자의 상태, 간호 수행 및 중재 등 구체적인 내용]
###
"""

final_prompt_NN = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_NN),
        few_shot_prompt_NN,
        ("human", "{input}"),
    ]
)

# UI 구현
st.title("SmartNurse ENR Chatbot")
st.subheader('스마트널스 ENR 챗봇에게 도움을 요청하세요.')
st.markdown('질병과 양식을 입력하면 간호 기록 예시가 생성됩니다.\n 왼쪽 메뉴를 열어 양식을 선택해 주세요.')

# css로 꾸미기
st.markdown("""
<style>
h1{
    color : rgb(6 161 54);
    }
}
</style> 
""", unsafe_allow_html=True)

# 사이드바, 셀렉트박스 추가
with st.sidebar:
    format = ['NANDA', 'SOAPIE', 'Focus DAR', 'Narrative Note']
    format_option = st.selectbox("", format,
        index=None,
        placeholder="기록 양식을 선택해 주세요.",
        label_visibility="visible"
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
        st.error("포맷이 선택되지 않았습니다.")
        return None

    send_message(message, "human")
    chain = prompt | azure
    return chain.invoke({"input": message, "output": ""})

message = st.chat_input("예시: 통증 환자의 간호기록을 작성해줘.")
if message:
    response = handle_chat(message, format_option)