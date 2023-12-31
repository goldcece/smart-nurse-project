# -*- coding: utf-8 -*-
"""fine_tuning_chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iXoB0Bk6-PPQ2XhUlYCjySeUCGmi4YLz
"""

!pip install --upgrade openai

!pip install langchain

!pip install transformers

import os
import pandas as pd
from openai import AzureOpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
from transformers import GenerationConfig, pipeline

# Azure OpenAI 객체 생성
azure = AzureChatOpenAI(
    api_key='key',
    api_version='version',
    azure_endpoint='endpoint',
    azure_deployment='model', # pre-trained model 불러오기
    model_name='model_name', # model_name 복붙

)

# prompt tuning
system_instrucion = """
You are a nurse with five years of experience in a tertiary care hospital or university hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
3. Use precise and concise language. Use endings like 음, 함, 임.
4. Fill out patient information in order
5. NANDA(Necessary to include '주관적자료/객관적자료', '단기목표/장기목표', '간호계획' '간호중재', '간호평가'), SOAPIE(Subject, Object, Assessment, Plan, Intervention, Evaluation), Focus DAR(Focus, Data, Action, Response) 등 간호기록 양식에 맞게 작성 해 주세요. \
6. Create a fictional patient's information.
7. Be as specific as possible.
8. Record precise measurements.
9. Use bullet points for the description instead of using numeric order.
10. Fill in the patient's name, age, and gender, but no other personal information.
"""

# 이상적 답변 예시
examples = [
    {"input": "비효과적 호흡 양상 환자의 간호기록 NANDA 양식으로 작성해줘.", "output": "환자 정보:\n\n이름: 이해준\n나이: 34세\n성별: 남성\n\n진단: 비효과적 호흡 양상 (NANDA-I #00032)\n\n주요 증상 및 특성:\n\n*호흡수 증가 (분당 24회)\n*호흡곤란 증상 (얕고 빠른 호흡)\n*산소포화도 감소 (92%)\n*환자의 주관적 호흡 불편 표현\n\n단기 목표:\n\n*환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소\n*산소포화도가 95% 이상으로 유지\n\n장기 목표:\n\n*환자가 호흡 곤란을 호소하지 않고, 산소포화도가 지속적으로 95% 이상 유지\n*환자가 호흡에 관련된 자가 관리 방법을 이해하고 실행\n\n간호 계획 및 중재:\n\n*호흡 상태 모니터링: 2시간마다 호흡수, 산소포화도 체크\n*산소 요법: 의사의 지시에 따라 산소 공급 (필요 시)\n*호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육\n*안위 제공: 편안한 자세 유지, 스트레스 감소 방법 안내\n*교육 및 상담: 호흡곤란의 원인과 관리 방법 설명\n\n간호 평가:\n*호흡 상태: 호흡수 및 산소포화도 변화 모니터링\n*환자의 자가 관리 능력: 호흡 운동 실시 여부 및 효과 평가\n*교육 효과: 환자의 이해도 및 반응 관찰"},
    {"input": "고체온 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보: \n 이름: 정수현\n 나이: 32세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 체온이 계속해서 높아지며 "'계속 땀이 나고 몸이 쑤셔'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n체온이 38.8°C로 상승\n피부는 뜨거워 보이고 땀이 많이 나고 있음. \n식욕 감소 및 체중 감소도 관찰됨.\n\nA: 평가 (Assessment): \n환자는 고체온 상태에 있음\n고체온이 식욕 감소와 체중 감소를 유발함.\n\nP: 계획 (Plan):  \n1. 적절한 수분 공급과 냉각을 위한 간호 중재   \n2. 체온 조절을 위한 의사와의 협력   \n3. 영양 상태 평가 및 영양제 투여 검토.\n\nI: 간호 중재 (Intervention): \n1. 적절한 수분 공급 및 냉각을 위해 쿨링 테크닉 적용   \n2. 의사의 지시에 따라 체온 조절을 위한 치료 시행   \n3. 영양사와 협력하여 환자의 영양 상태를 평가하고 필요 시 영양제 투여.\n\nE: 평가 (Evaluation): \n1. 쿨링 테크닉과 수분 공급으로 체온이 안정되고 땀이 감소함  \n 2. 의사의 치료로 체온이 정상 범위로 회복됨   \n3. 영양 상태가 향상되고 체중이 증가함."},
    {"input": "분만통증 환자의 간호기록을 Focus DAR 양식으로 작성해줘.", "output": "환자 정보:\n이름: 김지은  \n나이: 28세  \n성별: 여성  \n\nFocus: 분만 통증 관리  \n\nData:\n 환자는 현재 38주에 입성한 임신부로, 자연분만을 통한 분만을 계획하고 있음.\n 환자는 1시간 간격으로 자발적으로 시작된 자궁 수축이 진행 중이며, 수축 간격이 점차 감소하고 있음.\n 분만 도중에 통증이 강하게 느껴지며, 자극을 받을 때와 수축 중에 가장 강한 통증을 느끼고 있음.\n 통증의 강도는 Visual Analog Scale (VAS)로 8점으로 평가되며, 통증과 함께 호흡 곤란 및 안절부절 못하는 느낌을 호소함.\n 산모는 자연분만을 원하며, 진통제에 대한 개입을 최소화하고자 함.\n\nAction:\n1. 간호진은 산모의 통증 경험을 정확히 기록하고, 자연분만을 위한 통증 관리 계획 수립.\n2. 통증의 강도 및 느낌을 정확히 기록하고, 수축과 통증 간의 관계를 분석하여 최적의 관리 전략을 찾음.\n3. 호흡 및 안정적인 자세 유지에 대한 교육을 실시하여 산모가 효과적인 통증 관리를 위해 적절한 방법을 활용하도록 지원.\n4. 산모의 통증에 대한 이해도를 높이고, 산모와 동반자에게 협력적인 통증 관리에 대한 지침을 제공.\n5. 필요한 경우 의사와 협력하여 추가적인 통증 관리 옵션을 검토하고, 분만 진행 상황에 따라 관리 계획을 조절.\n\nResponse:\n 산모는 호흡 및 안정적인 자세 유지를 통해 통증을 일부 완화하고, 적절한 방법을 활용하여 통증에 대처함.\n 통증의 강도와 느낌을 정확히 기록하여 간호진은 효과적인 통증 관리 계획을 수립하고 조절함.\n 산모는 진통제에 대한 개입을 최소화하고, 자연분만의 경험을 중요시하여 통증을 더욱 관리 가능한 수준으로 유지함.\n 간호진의 교육과 지원을 통해 산모와 동반자는 통증 관리에 대한 이해도가 향상되어 협력적으로 참여함.\n 분만 도중에는 산모의 통증 관리를 계속 모니터링하고, 필요에 따라 의사와 협력하여 추가적인 조치를 취함."},
    {"input": "가스 교환 장애 환자의 간호기록 Narrative Note 양식으로 작성해줘.", "output": "환자 정보: \n 이름: 이철수\n 나이: 65세\n 성별: 남성\n\n환자 이철수 씨는 불안정한 혈압으로 입원하였음. 입원 당시 혈압은 160/100 mmHg로 측정되었으며, 과거력으로 고혈압 및 당뇨병이 있음. 환자는 현기증과 두통을 호소함. 추가적으로 환자는 최근 몇 주간 약물 복용을 자주 잊어버린다고 보고함.\n\n오늘 아침 혈압 측정 결과는 155/95 mmHg였으며, 오전 중에 일시적으로 145/90 mmHg로 감소하는 경향을 보임. 환자는 불규칙적인 혈압 변화에 대해 우려를 표현함. 환자의 혈압 변동과 관련하여 생활 습관, 식이, 약물 복용 패턴 등을 면밀히 평가함.\n\n간호 중재로는 혈압 모니터링을 매 4시간마다 실시하고, 의사 지시에 따라 혈압 조절 약물을 투여함. 환자에게 약물 복용 일정을 관리할 수 있는 방법을 안내하고, 저염식 식단과 규칙적인 운동의 중요성에 대해 교육함.\n\n오후 혈압 측정 결과는 150/92 mmHg로 다소 안정된 경향을 보임. 환자는 제공된 교육 내용에 대해 긍정적인 반응을 보이며, 약물 복용과 생활 습관 개선에 대한 의지를 표현함. 향후 계획으로는 지속적인 혈압 모니터링, 약물 관리 및 환자 교육을 이어갈 예정임."}
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_instrucion),
        few_shot_prompt,
        ("human", "{input}"),
        ("ai", "{output}"),

    ]
)

"""### 간호기록 생성 클래스"""

# 간호기록 생성 클래스

class Gen_ENR():
  def __init__(self):

    pass

  # 챗봇 모델 정의
  def chat():
      generation_config = GenerationConfig.from_pretrained(model_name)
      pipe = pipeline(
          "text-generation",
          model=azure,
          max_length=2500,
          temperature=0.1,
          top_p=0.0,
          repetition_penalty=0.0,
          generation_config=generation_config,
      )

  # model
  def chat_model(self, text):
      # system role & human message template
      system_prompt = SystemMessagePromptTemplate.from_template(system_instrucion)
      template = f"{text}."
      human_prompt = HumanMessagePromptTemplate.from_template(template)

      # chat prompt template
      chat_prompt = ChatPromptTemplate.from_messages([system_prompt, few_shot_prompt, human_prompt])

      # result output
      result = azure(chat_prompt.format_prompt(text=text).to_messages())

      print(f"Question  : {text}")
      print(f"Answer : {result}\n")

      return result

 # process

  def process_query(self, text, i):
      # 결과를 저장할 리스트
      candidates = []

      # 각 사용자 질문에 대해 generate_ENR 함수 호출 및 결과 리스트에 추가
      for i, question in enumerate(text):
          result = self.chat_model(text, i)
          candidates.append(result)

      # candidates 리스트를 반환하거나 다른 작업 수행
      return candidates

# 사용자 쿼리 list
user_questions = "분만통증 간호기록을 SOAPIE 양식으로 작성해줘."

# 사용자 질문 처리 함수 호출
my_instance = Gen_ENR()

result_model = my_instance.chat_model(user_questions)
reult_process = my_instance.process_query(user_questions)
