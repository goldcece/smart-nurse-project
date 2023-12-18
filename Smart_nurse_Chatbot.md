

# NANDA 2.0 :** 실습용 전자간호기록시스템 LLM 적용 / 전자간호기록 문구 작동 작성 시스템 

🍀팀장 : 황재연

☘️팀원 : 안지은, 홍세헌, 박아영

🗄️Repository : [https://github.com/SmartNurse/Year_Dream](https://github.com/SmartNurse/Year_Dream)

---

- **목차**
    1. [프로젝트 개요]
    2. [데이터 EDA]
        1. 데이터 현황
        2. 1차 전처리
        3. 2차 전처리
        4. LLM 학습 데이터셋 제작
    3. [Model : Fine-tuning & LLM 학습]
        1. GPT API : Microsoft Azure Open AI
        2. GPT Fine-tuning
        3. LLM 학습
    4. [서비스 구현 및 배포]
        1. Chatbot : Langchain 
        2. Webapp deployment : Streamlit 



    

# 1. 프로젝트 **개요**

✨프로젝트 필요성 (문제 제시)

![발표자료.001.png](Smart_nurse_Chatbot/prologue.png)

![간호대학생 및 신규간호사를 위한 DK Medi Info 의 Smart Nurse ENR 시스템](Smart_nurse_Chatbot/smartnursesystem.png)

간호대학생 및 신규간호사를 위한 DK Medi Info 의 Smart Nurse ENR 시스템

### **✨프로젝트 목표**

- 전자 간호기록 데이터 기반 실습용 전자 간호기록 문구 자동 생성 챗봇 시스템 구현

### ✨Work Flow

- **데이터 전처리 → LLM 모델 Fine-tuning  → Prompt fine-tuning → 챗봇 서비스 구현 → 웹앱 배포**
    
    ![work flow](Smart_nurse_Chatbot/work_flow.001.png)
    
    work flow
    

### ✨사용한 툴

- GPT-3.5 turbo (Azure Open AI) : Pretrained LLM Model
- (하이퍼클로바X) : Pretrained LLM Model
- Langchain : Chatbot implementation
- Streamlit : Webapp deployment

### ✨ 용어 설명

- 스마트널스에서 제공하는 주요 간호기록 양식
    - NANDA : 북미 간호협회 표준 진단이 포함된 기록 체계
        - [domain, class, diagnosis, 자료, 간호 목표, 간호 계획] 등 포함
    - SOAPIE : 문제중심 기록체계
        - [주관적 자료, 객관적 자료, 간호사정, 간호계획, 중재, 평가] 내용 포함
    - Focus DAR : 핵심 중심 기록체계
        - [핵심, 자료(주관적/객관적), 간호활동, 환자반응] 내용 포함
    - 서술기록(Narrative Note) : 시간 경과에 따른 전통적인 기록방법
    
    ![스마트널스 시스템 간호기록 양식 예시](Smart_nurse_Chatbot/record.001.png)
    
    스마트널스 시스템 간호기록 양식 예시
    

# 2. 데이터 EDA

### a. 데이터 현황

- DK Medi Info 의 Smart Nurse ENR 시스템에 저장된 자연어 간호기록 데이터
- 데이터의 수 : *18,400+ (NCP + AWS DB)*
    - NANDA : *1,900+*
        
        ![스크린샷 2023-11-30 210912.png](Smart_nurse_Chatbot/rawdata.png)
        
    - SOAPIE : *900+*
        
        ![스크린샷 2023-11-30 211048.png](Smart_nurse_Chatbot/raw_soapie.png)
        
    - Focus_DAR : *2,200+*
        
        ![스크린샷 2023-11-30 211151.png](Smart_nurse_Chatbot/raw_focusdar.png)
        
    - Narrative Note : *13,000+*
        
        ![스크린샷 2023-11-30 211333.png](Smart_nurse_Chatbot/raw_narrnote.png)
        
    - Nursing Data: *400+*
        
        ![스크린샷 2023-11-30 211421.png](Smart_nurse_Chatbot/rawraw.png)
        

![data.png](Smart_nurse_Chatbot/data.png)

<aside>
💡 NaN, 특수문자, 숫자, 한글 자음, 알파벳, 노래가사, 어색한 문장 등 잘못 작성된 raw 데이터 확인

</aside>



---

### b. 1차 전처리

- raw data 직접 검수
- 정규표현식 코드 작성
    - 코드
        
        ```python
        # 필터링 조건에 맞지 않는 행을 확인하는 함수
        def is_valid_row(text):
            # 숫자만 / 알파벳만 / 한글자모음만 / 특수 문자만 있는 경우
            if re.fullmatch(r'\d+', text) or re.fullmatch(r'[a-zA-Z]+', text) or re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣ]+', text) or re.fullmatch(r'[^\w\s]+', text):
                return False
            return True
        
        # 필터링 조건에 맞지 않는 단어를 제거하는 함수
        def remove_invalid_words(text)
            words = str(text).split()
            valid_words = [word for word in words if is_valid_row(word)]
            return ' '.join(valid_words) if valid_words else pd.NA  # 빈 문자열 대신 pd.NA 반환
        
        # 길이가 n 미만인 데이터가 있는 행 제거
        def remove_short_words(text):
            # 문자열로 변환하고 단어로 분리
            words = str(text).split()
            # 길이가 3 초과인 단어만 유효하다고 판단
            valid_words = [word for word in words if len(word) > 3]
            # 유효한 단어들을 다시 문자열로 합치기
            return ' '.join(valid_words) if valid_words else pd.NA
        
        # 동음이의어 제거
        def unify_terms(sentence):
            S_patterns_to_unify = [
                r'<주관적(?![\]자료])',
                r'\[주관적\]',
                r'주관적:',
                r'-주관적:',
                r'주관적자료:',
                r'\[주관적자료\]',
                r'주관적자료(?!\])',
                r'\<주관적자료\>',
                r'\[주관적',
                r'\*주관적',
                r'\(주관적\)',
                r'\(주관적',
                r'주관적-보호자',
                r'\(\[주관적자료\]\)',
                r'-\[주관적자료\]',
                r'\*\[주관적자료\]',
            ]
        
            O_patterns_to_unify = [
                r'<객관적(?![\]자료])',
                r'\[객관적\]',
                r'객관적:',
                r'-객관적:',
                r'객관적자료:',
                r'\[객관적자료\]',
                r'객관적자료(?!\])',
                r'\<객관적자료\>',
                r'\[객관적',
                r'\*객관적',
                r'\(객관적\)',
                r'\(객관적',
                r'객관적-보호자',
                r'\(\[객관적자료\]\)',
                r'-\[객관적자료\]',
                r'\*\[객관적자료\]',
            ]
            ST_patterns_to_unitfy = [
                r'단기목표:',
                r'\<단기목표\>',
                r'단기목표',
                r'-단기목표',
                r'단기',
                r'단기:',
                r'\[단기\]',
                r'\<단기\>',
                r'\*단기목표:',
                r'\*\[단기목표\]',
                r'\*단기:',
                r'\*\<단기목표\>',
                r'\*단기목표',
                r'\[단기목표\]\)',
                r'\(\[단기목표\]\)'
            ]
            LT_patterns_to_unitfy = [
                r'장기목표:',
                r'\<장기목표\>',
                r'장기목표',
                r'-장기목표',
                r'장기',
                r'장기:',
                r'\[장기\]',
                r'\<장기\>',
                r'\*장기목표:',
                r'\*\[장기목표\]',
                r'\*장기:',
                r'\*\<장기목표\>',
                r'\*장기목표',
                r'\[장기목표\]\)',
                r'\(\[장기목표\]\)'
        
            ]
        
            for pattern in S_patterns_to_unify:
              sentence = re.sub(pattern, '[주관적자료]', sentence)
        
            for pattern in O_patterns_to_unify:
              sentence = re.sub(pattern, '[객관적자료]', sentence)
        
            for pattern in ST_patterns_to_unitfy:
              sentence = re.sub(pattern, '[단기목표]', sentence)
        
            for pattern in LT_patterns_to_unitfy:
              sentence = re.sub(pattern, '[장기목표]', sentence)
        
            # '[주관적자료]자료]'와 같은 패턴을 '[주관적자료]'로 치환
            sentence = re.sub(r'\[주관적자료\](자료\])+', '[주관적자료]', sentence)
            sentence = re.sub(r'\[객관적자료\](자료\])+', '[객관적자료]', sentence)
            sentence = re.sub(r'\[\[\[단기목표\]목표\]\]', '[단기목표]', sentence)
            sentence = re.sub(r'\[\[단기목표\]목표\]', '[단기목표]', sentence)
            sentence = re.sub(r'\[단기목표\]:', '[단기목표]', sentence)
            sentence = re.sub(r'\[\[\[장기목표\]목표\]\]', '[장기목표]', sentence)
            sentence = re.sub(r'\[\[장기목표\]목표\]', '[장기목표]', sentence)
            sentence = re.sub(r'\[장기목표\]:', '[장기목표]', sentence)
            return sentence
        
        # 중복 행 제거
        def check_row_for_duplicates(row):
            row_values = row.tolist()
            return len(row_values) != len(set(row_values))
        
        # 위 전처리 함수들을 적용하는 함수
        def function_apply(df, filter_column) :
            # 모든 열에 적용
            for column in df.columns :
                df[column] = df[column].apply(remove_invalid_words)
        
            # filter된 열에만 적용
            for column in filter_column :
                df[column] = df[column].apply(remove_short_words)
        
            # 행별로 적용
            df = df[~df.apply(check_row_for_duplicates, axis = 1)]
        
            # '<NA>'가 포함된 행 제거
            df = df[~df.apply(lambda x: x.astype(str).str.contains('<NA>')).any(axis=1)]
            df = df.reset_index(drop=True)
        
            # 모든 행 동음이의어 처리
            for column in df.columns:
              df[column] = df[column].apply(unify_terms)
        
            return df
        ```
        
    - 전과 후 비교
        
        ![**전처리(전)**](Smart_nurse_Chatbot/predata.png)
        
        **전처리(전)**
        
        ![스크린샷 2023-11-30 213046.png](Smart_nurse_Chatbot/pro_data.png)
        
        전처리(후)
        
    

**이슈 및 한계점**

- 불용어 처리 코드로 인해 의미있는 단어도 삭제되는 경우 발생
- 도메인에 특화된 자연어 데이터 처리의 어려움 → 유의미한 정보와 무의미한 정보의 구분 및 취사선택
- 데이터 전처리를 꼼꼼하게 할수록 사용할 수 있는 데이터의 수가 줄어듬 → 학습이 잘 되지 않을 가능성
- 실습용(교육용) 데이터의 한계 → 의료업계는 정확한 데이터가 중요한데 그렇지 못한 경우가 많았음
- 데이터 전처리에 많은 시간 소요 → 그러나 **데이터전처리는 한글 자연어처리 과제의 핵심**

### c. 2차 전처리

- 학습 데이터셋에서 총 17개의 빈출 병명 추출
    - `비효과적 호흡 양상 / 비효과적 기도청결 / 가스교환 장애 / 낙상의 위험 / 불안정한 혈압의 위험 / 근육의 긴장 / 충수염으로 인한 복통 / 외상성 지주막하 출혈 / 당뇨병 / 무릎관절증 / 퇴행성 관절염 / 통증 / 욕창 / 자발적 환기장애 / 급성통증 / 고체온 / 분만 통증`
        
        ![스크린샷 2023-11-30 224302.png](Smart_nurse_Chatbot/frequent.png)
        
    - 추출한 병명 키워드 별로 데이터를 분류하여 구체적으로 전처리
        - Langchain GPT prompting 을 활용하여 양식별 데이터 전처리(맞춤법 등)
        - 코드
            
            ```python
            import pandas as pd
            from langchain.chat_models import AzureChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            
            # Azure OpenAI를 사용하는 객체 생성
            chat = AzureChatOpenAI(
                deployment_name='모델',
                model_name='gpt-35-turbo',
            )
            
            # 한국어 프롬프트 템플릿 생성
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        
                        """ 당신은 간호기록 작성지를 작성해주는 전문가입니다. 다음의 의료기록을 분석하고 중간에 끊긴 문장이 있거나 오탈자가 있으면 알맞게 수정하고 NANDA 형식으로 출력하세요.
                        한글로 작성하세요.
                        """,
                    ),
                    ("human", "{medical_record}"),
                ]
            )
            
            # 프롬프트와 채팅 모델 결합
            chain = prompt | chat  
            
            # CSV 파일에서 데이터 불러오기
            pd.read_csv("각 전처리한 파일 경로")
            
            # 새로운 데이터셋
            new_dataset = []
            
            # 각 행 처리
            for index, row in df.iterrows():
                response = chain.invoke({"medical_record": row['input']})
                new_dataset.append({'input': row['input'], 'output': response})
                print(response)
                
            
            # 새로운 데이터셋을 DataFrame으로 변환
            new_df = pd.DataFrame(new_dataset)
            
            # 새로운 데이터셋을 CSV 파일로 저장
            new_df.to_csv("저장하고 싶은 경로\\이름.csv", index=False, encoding='utf-8')
            ```
            
        - 예시
            
            ![preprocess.png](Smart_nurse_Chatbot/preprocess.png)
            

### d. LLM 학습 데이터셋 제작

- 전처리한 데이터셋을 LLM 학습에 필요한 jsonl 형태로 변환
    - 간호기록 데이터 컬럼에서 병명(disease name) 추출
        - 병명을 추출 할 수 없는 데이터는 제거
    - 코드
        
        ```python
        import pandas as pd
        from langchain.chat_models import ChatOpenAI
        from langchain.chat_models import AzureChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Azure OpenAI를 사용하는 객체 생성
        chat = AzureChatOpenAI(
            deployment_name='모델',
            model_name='gpt-35-turbo',
        )
        
        # 병명 추출 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    
                    """ 당신은 간호기록 작성지를 작성해주는 전문가입니다. 다음의 의료기록을 분석하고 예시를 참고하여 대표적인 병명 하나만 기입하세요.
                    예시 : 비효과적 호흡 양상, 비효과적 기도청결, 가스교환 장애, 낙상의 위험, 불안정한 혈압의 위험, 근육의 긴장,  충수염으로 인한 복통,  외상성 지주막하, 출혈,  당뇨병,  무릎관절증 퇴행성 관절염,  통증,  욕창,  자발적 환기장애,  급성통증,  고체온,  분만 통증 
                    """,
                ),
                ("human", "{medical_record}"),
            ]
        )
        
        # 프롬프트와 채팅 모델 결합
        chain = prompt | chat  
        # CSV 파일에서 데이터 불러오기
        pd.read_csv("문장 수정한 파일 경로", encoding="utf-8")
        # 새로운 데이터셋
        new_dataset = []
        
        # 각 행 처리
        for index, row in df.iterrows():
            response = chain.invoke({"medical_record": row['input']})
            new_dataset.append({'input': row['input'], 'disease name': response})
            print(response)
            
        
        # 새로운 데이터셋을 DataFrame으로 변환
        new_df = pd.DataFrame(new_dataset)
        
        # 정제 함수정의
        def cleaning(text):
            replacements = {
                r'content=': '',
                r'.*병명': '병명',
                r'병명 : ' : '병명: ',
                r'A:' : '병명:',
                r'주진단명': '병명',
                r'진단명' : '병명',
        
            }
        
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
        
            return text
        
        def clean2(text):
          text = re.sub(r'병명: ', '', text)
          return text
        
        # 데이터프레임에 함수 적용
        new_df['disease name'] = new_df['disease name'].apply(cleaning)
        new_df
        
        # 새로운 데이터셋을 CSV 파일로 저장
        new_df.to_csv("저장하고 싶은 경로\\이름.csv", index=False, encoding='utf-8')
        ```
        
    - 결과
        
        ![스크린샷 2023-12-14 오후 2.16.34.png](Smart_nurse_Chatbot/sampledata1.png)
        
        ![스크린샷 2023-12-14 오후 2.31.28.png](Smart_nurse_Chatbot/sampledata2.png)
        
    - jsonl 형태로 변환
    - 예시
        
        ```json
        
        {"role": "user", "content": "**호흡곤란**에 대한 간호기록을 FOCUS DAR형식으로 작성해줘"}, {"role": "assistant", "content": "FOCUS:  비강캐뉼라를 통한 산소요법 D: [주관적 자료] \"숨쉬기가 답답해요.\"    [객관적 자료] 호흡곤란과 전신쇠약 증상으로 인해 순환기내과 입원치료를 요청함. 안절부절 말초산소포화도 측정 결과 Sp02. A: - 주치의가 cannula로 3L/min 말초산소포화도 측정 결과를 계속해서 모니터링하며, 호흡곤란으로 Semi-Fowler\\'s position을 적용하여 대상자의 전반적인 상태를 확인함.    - 대상자의 활력징후 및 호흡수와 산소 사용량으로 화재 발생 위험성이 있어 비강캐뉼라를 통한 산소요법을 시행하고, 피부 손상의 위험성에 대해 유의함. R: 혈압 100/70mmHg, 맥박 90회/분, 호흡수 20회/분, 체온 37.0˚C, Sp02 측정 결과 호흡곤란이 완화되었으며, 대상자가 교육 내용을 이해하고 편안하다고 함."}]}
        {"role": "user", "content": "**위장관 잔류증**에 대한 간호기록을 NarrativeNote형식으로 작성해줘"}, {"role": "assistant", "content": "NarrativeNote: 대상자의 위관영양을 확인한 결과 23mL의 잔류량이 있었습니다. 이에 대해 1200cc의 액체를 L-Tube 방법을 통해 주입하였으며, gtt/min을 설정하였습니다. 대상자는 복부팽만 증상이 없었고, 반좌위를 취하도록 교육하였습니다."}]}
        {"role": "user", "content": "**상세불명의 바이러스에 의한 비효과적 호흡 양상**에 대한 간호기록을 NANDA DAR형식으로 작성해줘"}, {"role": "assistant", "content": "Domain: 활동/휴식  [Class] 심혈관/호흡기계 반응  [Diagnosis] 비효과적 호흡 양상  [Collecting Data] 환자는 \"숨쉬기가 불편해요\" 라고 언급하였으며, 진단명은 상세불명의 바이러스이고, x-ray 결과는 right이며 110/70 RR25 Spo2 쌕색거리는 \"sutum\"이 발견되었다.  [Goal] - 단기목표: 대상자는 정상적인 호흡양상을 가지게 된다. - 장기목표: 대상자는 호흡하기 편안하게 된다.  [Plan] - 호흡수와 리듬을 측정하고, 필요시 조정한다. - 효율적인 기침을 하도록 대상자를 격려한다. - 대상자를 쿠션으로 지지하여 호흡기계 사용이 필요하다면 사용한다.  [Interventions] - 호흡수와 리듬을 측정하였고, 11/15 09:00에 변경하였다. - 환자분께서 호흡기 관리를 위해 변경해주었다. - 대상자가 효율적으로 기침할 수 있도록 격려하였다. - 대상자를 효율적으로 격려하여 배출이 원활하도록 도왔다. - 대상자를 쿠션으로 지지하여 호흡기계 사용이 필요하다면 사용하였다.  [Evaluation] - 단기목표: 대상자는 정상적인 호흡양상을 (미달성) 11/16 09:00 환자분께서 전보다는 나아졌지만 불편함을 호소하였다. - 장기목표: 대상자는 호흡하기 편안하다고 판단할 수 없으며, 추후 관찰이 필요하다. (2022.11.15)"}]}
        {"role": "user", "content": "**지주막하 출혈**에 대한 간호기록을 SOAPIE형식으로 작성해줘"}, {"role": "assistant", "content": "S: \"환자가 먹을 때마다 걸려요\" O: 지주막하 출혈 진단으로 입원하여 치료 중이며 기침이 관찰됩니다. A: -연하곤란으로 구강섭취 어려움이 있음. P: semi-fowler의 자세로 취침 시키기, 필요시 경장 영양액 제공 및 흡인 시 투여하기. I: -경장 영양액 수행함. -경장 영양 투여를 위해 흡인하여 확인함. -L-tube 삽입하여 200ml 주입함. 30ml씩 주입함. 측정함. -앉아있을 때의 교육을 시행함. E: -L-tube 삽입 후 semi-fowler의 자세를 유지함."}]}
        ```
        
- 최종적으로 학습에 사용한 데이터
    
    ![train_data.png](Smart_nurse_Chatbot/train_data.png)
    

# 3. Model : LLM(GPT-3.5 Turbo) Fine Tuning

### a. GPT API : Microsoft Azure Open AI

- GPT API 불러오기 코드
    
    ```python
    !pip install openai
    from openai import OpenAI
    
    #api 불러오기
    client = OpenAI(api_key=" ")
    prompt = [{"role": "system", "content": "system"}]
    
    def chat(text):
      user_input = {"role": "user", "content": text}
      prompt.append(user_input)
      answer = client.chat.completions.create(
          model="gpt-3.5-turbo",
          prompt=prompt,
          temperature=1,
          top_p=0.9) # 다른 parameter 추가
    
      answ = answer.to_dict_recursive()
      bot_answ = answer['choices'][0]['meassage']['content']
      bot_resp = {"role" : "assistant", "content" : bot_answ}
      prompt.append(bot_resp)
      return bot_answ
    
    print(chat("낙상 고위험군 간호기록 써줘."))
    ```
    
    → 기업에서 제공한 Azure를 통해서 OpenAI API를 사용할 예정
    

### b. GPT Fine-tuning

- 빈출 병명별 각 양식의 이상적인 샘플 데이터셋 제작
    - ChatGPT 활용
        
        ![generation.png](Smart_nurse_Chatbot/generation.png)
        
        - ChatGPT에 대한 오류 체크
        (e.g. 통증환자의 간호기록을 작성하라는 instruct에 ‘하품 통증 환자’ 예시를 생성)
    - 예tl
- GPT Fine-tuning
    - Instruction tuning
        
        ```python
        system_instruction =
        """
        너는 대학병원 간호사야. 다음의 지침대로 비효과적 호흡 양상 환자의 간호기록을 구체적으로 작성해줘.
        
        1. NANDA, SOAPIE, Focus DAR 등 간호기록 양식에 맞게 적어주세요.
        2. 주요 증상 및 특성, 단기 및 장기 목표, 간호 중재, 간호 평가 항목을 필수 기재하세요.
        3. 최대한 구체적으로 작성하세요. 예를 들어 '적절한 호흡 기법 교육'이란 답변 보단 '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육'이 좋습니다.
        4. 정확한 수치를 기록하세요. 예를 들어 '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소'
        5. 정확하고 간결한 말투를 사용하세요. ~입니다. ~에요 같은 문장식 서술은 피해주세요.
        
        """
        ```
        
    - Prompt tuning
        
        ```python
        prompt_format = """"
        {"role": "system", "content": instruction"}, {"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}
        """
        prompt = [
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "비효과적 호흡 양상 환자의 간호기록 NANDA 양식으로 작성해줘."}, {"role": "assistant", "content": "환자 정보:\n\n이름: 이해준\n나이: 34세\n성별: 남성\n\n진단: 비효과적 호흡 양상 (NANDA-I #00032)\n\n주요 증상 및 특성:\n\n*호흡수 증가 (분당 24회)\n*호흡곤란 증상 (얕고 빠른 호흡)\n*산소포화도 감소 (92%)\n*환자의 주관적 호흡 불편 표현\n\n단기 목표:\n\n*환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소\n*산소포화도가 95% 이상으로 유지\n\n장기 목표:\n\n*환자가 호흡 곤란을 호소하지 않고, 산소포화도가 지속적으로 95% 이상 유지\n*환자가 호흡에 관련된 자가 관리 방법을 이해하고 실행\n\n간호 계획 및 중재:\n\n*호흡 상태 모니터링: 2시간마다 호흡수, 산소포화도 체크\n*산소 요법: 의사의 지시에 따라 산소 공급 (필요 시)\n*호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육\n*안위 제공: 편안한 자세 유지, 스트레스 감소 방법 안내\n*교육 및 상담: 호흡곤란의 원인과 관리 방법 설명\n\n간호 평가:\n*호흡 상태: 호흡수 및 산소포화도 변화 모니터링\n*환자의 자가 관리 능력: 호흡 운동 실시 여부 및 효과 평가\n*교육 효과: 환자의 이해도 및 반응 관찰"},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "고체온 환자의 간호기록을 SOAPIE 양식으로 작성해줘."}, {"role": "assistant", "content":"환자 정보: \n 이름: 정수현\n 나이: 32세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 체온이 계속해서 높아지며 "'계속 땀이 나고 몸이 쑤셔'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n체온이 38.8°C로 상승\n피부는 뜨거워 보이고 땀이 많이 나고 있음. \n식욕 감소 및 체중 감소도 관찰됨.\n\nA: 평가 (Assessment): \n환자는 고체온 상태에 있음\n고체온이 식욕 감소와 체중 감소를 유발함.\n\nP: 계획 (Plan):  \n1. 적절한 수분 공급과 냉각을 위한 간호 중재   \n2. 체온 조절을 위한 의사와의 협력   \n3. 영양 상태 평가 및 영양제 투여 검토.\n\nI: 간호 중재 (Intervention): \n1. 적절한 수분 공급 및 냉각을 위해 쿨링 테크닉 적용   \n2. 의사의 지시에 따라 체온 조절을 위한 치료 시행   \n3. 영양사와 협력하여 환자의 영양 상태를 평가하고 필요 시 영양제 투여.\n\nE: 평가 (Evaluation): \n1. 쿨링 테크닉과 수분 공급으로 체온이 안정되고 땀이 감소함  \n 2. 의사의 치료로 체온이 정상 범위로 회복됨   \n3. 영양 상태가 향상되고 체중이 증가함."},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "분만통증 환자의 간호기록을 Focus DAR 양식으로 작성해줘."}, {"role": "assistant", "content":"환자 정보:\n이름: 김지은  \n나이: 28세  \n성별: 여성  \n\nFocus: 분만 통증 관리  \n\nData:\n 환자는 현재 38주에 입성한 임신부로, 자연분만을 통한 분만을 계획하고 있음.\n 환자는 1시간 간격으로 자발적으로 시작된 자궁 수축이 진행 중이며, 수축 간격이 점차 감소하고 있음.\n 분만 도중에 통증이 강하게 느껴지며, 자극을 받을 때와 수축 중에 가장 강한 통증을 느끼고 있음.\n 통증의 강도는 Visual Analog Scale (VAS)로 8점으로 평가되며, 통증과 함께 호흡 곤란 및 안절부절 못하는 느낌을 호소함.\n 산모는 자연분만을 원하며, 진통제에 대한 개입을 최소화하고자 함.\n\nAction:\n1. 간호진은 산모의 통증 경험을 정확히 기록하고, 자연분만을 위한 통증 관리 계획 수립.\n2. 통증의 강도 및 느낌을 정확히 기록하고, 수축과 통증 간의 관계를 분석하여 최적의 관리 전략을 찾음.\n3. 호흡 및 안정적인 자세 유지에 대한 교육을 실시하여 산모가 효과적인 통증 관리를 위해 적절한 방법을 활용하도록 지원.\n4. 산모의 통증에 대한 이해도를 높이고, 산모와 동반자에게 협력적인 통증 관리에 대한 지침을 제공.\n5. 필요한 경우 의사와 협력하여 추가적인 통증 관리 옵션을 검토하고, 분만 진행 상황에 따라 관리 계획을 조절.\n\nResponse:\n 산모는 호흡 및 안정적인 자세 유지를 통해 통증을 일부 완화하고, 적절한 방법을 활용하여 통증에 대처함.\n 통증의 강도와 느낌을 정확히 기록하여 간호진은 효과적인 통증 관리 계획을 수립하고 조절함.\n 산모는 진통제에 대한 개입을 최소화하고, 자연분만의 경험을 중요시하여 통증을 더욱 관리 가능한 수준으로 유지함.\n 간호진의 교육과 지원을 통해 산모와 동반자는 통증 관리에 대한 이해도가 향상되어 협력적으로 참여함.\n 분만 도중에는 산모의 통증 관리를 계속 모니터링하고, 필요에 따라 의사와 협력하여 추가적인 조치를 취함."},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "가스 교환 장애 환자의 간호기록 Narrative Note 양식으로 작성해줘."}, {"role": "assistant", "content": "환자 정보: \n 이름: 이철수\n 나이: 65세\n 성별: 남성\n\n환자 이철수 씨는 불안정한 혈압으로 입원하였음. 입원 당시 혈압은 160/100 mmHg로 측정되었으며, 과거력으로 고혈압 및 당뇨병이 있음. 환자는 현기증과 두통을 호소함. 추가적으로 환자는 최근 몇 주간 약물 복용을 자주 잊어버린다고 보고함.\n\n오늘 아침 혈압 측정 결과는 155/95 mmHg였으며, 오전 중에 일시적으로 145/90 mmHg로 감소하는 경향을 보임. 환자는 불규칙적인 혈압 변화에 대해 우려를 표현함. 환자의 혈압 변동과 관련하여 생활 습관, 식이, 약물 복용 패턴 등을 면밀히 평가함.\n\n간호 중재로는 혈압 모니터링을 매 4시간마다 실시하고, 의사 지시에 따라 혈압 조절 약물을 투여함. 환자에게 약물 복용 일정을 관리할 수 있는 방법을 안내하고, 저염식 식단과 규칙적인 운동의 중요성에 대해 교육함.\n\n오후 혈압 측정 결과는 150/92 mmHg로 다소 안정된 경향을 보임. 환자는 제공된 교육 내용에 대해 긍정적인 반응을 보이며, 약물 복용과 생활 습관 개선에 대한 의지를 표현함. 향후 계획으로는 지속적인 혈압 모니터링, 약물 관리 및 환자 교육을 이어갈 예정임."}
            ]
        ```
        
    - 코드
        
        ```python
        import os
        from openai import AzureOpenAI
        import pandas as pd
        # API 불러오기
        client = AzureOpenAI(
            api_key = 'key',
            api_version = 'version',
            azure_endpoint = 'url'
        
        # prompt tuning model
        
        system_instruction = "You are an assistant who needs to help nurses who are overwhelmed with nursing records(ENR).\
         You are a nurse in a tertiary care hospital or university hospital. Please complete a detailed nursing record for a patient with ineffective respiration using the following instructions.\
         1. NANDA 기록양식, SOAPIE( S: subject data, O: object data, A: assessment, P: plan, I: intervention, E: evalutaion), Focus DAR(Focus, Data, Action, Response) 등의 기록 양식에 맞춰서 작성 하세요.\
         2. 최대한 구체적으로 작성하세요. 예를 들어 ‘적절한 호흡 기법 교육‘이란 답변 보단 ‘호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육‘이 좋습니다.\
         3. 가능한 정확한 수치 데이터를 사용하세요. 예를 들어 ‘환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소’\
         4. 정확하고 간결한 말투를 사용하세요. -음, -함, -임 과 같은 어미를 사용하세요.\
         5. 환자 정보(이름, 나이, 성별) 을 포함하세요.\
         6. 하위구조는 숫자 대신 bullet 형식을 사용하세요.\
         7. user 가 제시하는 데이터가 있을 때는 그 데이터를 사용하여 작성하세요. "
        
        prompt = [
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "비효과적 호흡 양상 환자의 간호기록 NANDA 양식으로 작성해줘."}, {"role": "assistant", "content": "환자 정보:\n\n이름: 이해준\n나이: 34세\n성별: 남성\n\n진단: 비효과적 호흡 양상 (NANDA-I #00032)\n\n주요 증상 및 특성:\n\n*호흡수 증가 (분당 24회)\n*호흡곤란 증상 (얕고 빠른 호흡)\n*산소포화도 감소 (92%)\n*환자의 주관적 호흡 불편 표현\n\n단기 목표:\n\n*환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소\n*산소포화도가 95% 이상으로 유지\n\n장기 목표:\n\n*환자가 호흡 곤란을 호소하지 않고, 산소포화도가 지속적으로 95% 이상 유지\n*환자가 호흡에 관련된 자가 관리 방법을 이해하고 실행\n\n간호 계획 및 중재:\n\n*호흡 상태 모니터링: 2시간마다 호흡수, 산소포화도 체크\n*산소 요법: 의사의 지시에 따라 산소 공급 (필요 시)\n*호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육\n*안위 제공: 편안한 자세 유지, 스트레스 감소 방법 안내\n*교육 및 상담: 호흡곤란의 원인과 관리 방법 설명\n\n간호 평가:\n*호흡 상태: 호흡수 및 산소포화도 변화 모니터링\n*환자의 자가 관리 능력: 호흡 운동 실시 여부 및 효과 평가\n*교육 효과: 환자의 이해도 및 반응 관찰"},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "고체온 환자의 간호기록을 SOAPIE 양식으로 작성해줘."}, {"role": "assistant", "content":"환자 정보: \n 이름: 정수현\n 나이: 32세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 체온이 계속해서 높아지며 "'계속 땀이 나고 몸이 쑤셔'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n체온이 38.8°C로 상승\n피부는 뜨거워 보이고 땀이 많이 나고 있음. \n식욕 감소 및 체중 감소도 관찰됨.\n\nA: 평가 (Assessment): \n환자는 고체온 상태에 있음\n고체온이 식욕 감소와 체중 감소를 유발함.\n\nP: 계획 (Plan):  \n1. 적절한 수분 공급과 냉각을 위한 간호 중재   \n2. 체온 조절을 위한 의사와의 협력   \n3. 영양 상태 평가 및 영양제 투여 검토.\n\nI: 간호 중재 (Intervention): \n1. 적절한 수분 공급 및 냉각을 위해 쿨링 테크닉 적용   \n2. 의사의 지시에 따라 체온 조절을 위한 치료 시행   \n3. 영양사와 협력하여 환자의 영양 상태를 평가하고 필요 시 영양제 투여.\n\nE: 평가 (Evaluation): \n1. 쿨링 테크닉과 수분 공급으로 체온이 안정되고 땀이 감소함  \n 2. 의사의 치료로 체온이 정상 범위로 회복됨   \n3. 영양 상태가 향상되고 체중이 증가함."},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "분만통증 환자의 간호기록을 Focus DAR 양식으로 작성해줘."}, {"role": "assistant", "content":"환자 정보:\n이름: 김지은  \n나이: 28세  \n성별: 여성  \n\nFocus: 분만 통증 관리  \n\nData:\n 환자는 현재 38주에 입성한 임신부로, 자연분만을 통한 분만을 계획하고 있음.\n 환자는 1시간 간격으로 자발적으로 시작된 자궁 수축이 진행 중이며, 수축 간격이 점차 감소하고 있음.\n 분만 도중에 통증이 강하게 느껴지며, 자극을 받을 때와 수축 중에 가장 강한 통증을 느끼고 있음.\n 통증의 강도는 Visual Analog Scale (VAS)로 8점으로 평가되며, 통증과 함께 호흡 곤란 및 안절부절 못하는 느낌을 호소함.\n 산모는 자연분만을 원하며, 진통제에 대한 개입을 최소화하고자 함.\n\nAction:\n1. 간호진은 산모의 통증 경험을 정확히 기록하고, 자연분만을 위한 통증 관리 계획 수립.\n2. 통증의 강도 및 느낌을 정확히 기록하고, 수축과 통증 간의 관계를 분석하여 최적의 관리 전략을 찾음.\n3. 호흡 및 안정적인 자세 유지에 대한 교육을 실시하여 산모가 효과적인 통증 관리를 위해 적절한 방법을 활용하도록 지원.\n4. 산모의 통증에 대한 이해도를 높이고, 산모와 동반자에게 협력적인 통증 관리에 대한 지침을 제공.\n5. 필요한 경우 의사와 협력하여 추가적인 통증 관리 옵션을 검토하고, 분만 진행 상황에 따라 관리 계획을 조절.\n\nResponse:\n 산모는 호흡 및 안정적인 자세 유지를 통해 통증을 일부 완화하고, 적절한 방법을 활용하여 통증에 대처함.\n 통증의 강도와 느낌을 정확히 기록하여 간호진은 효과적인 통증 관리 계획을 수립하고 조절함.\n 산모는 진통제에 대한 개입을 최소화하고, 자연분만의 경험을 중요시하여 통증을 더욱 관리 가능한 수준으로 유지함.\n 간호진의 교육과 지원을 통해 산모와 동반자는 통증 관리에 대한 이해도가 향상되어 협력적으로 참여함.\n 분만 도중에는 산모의 통증 관리를 계속 모니터링하고, 필요에 따라 의사와 협력하여 추가적인 조치를 취함."},
            {"role": "system", "content": system_instruction}, {"role": "user", "content": "가스 교환 장애 환자의 간호기록 Narrative Note 양식으로 작성해줘."}, {"role": "assistant", "content": "환자 정보: \n 이름: 이철수\n 나이: 65세\n 성별: 남성\n\n환자 이철수 씨는 불안정한 혈압으로 입원하였음. 입원 당시 혈압은 160/100 mmHg로 측정되었으며, 과거력으로 고혈압 및 당뇨병이 있음. 환자는 현기증과 두통을 호소함. 추가적으로 환자는 최근 몇 주간 약물 복용을 자주 잊어버린다고 보고함.\n\n오늘 아침 혈압 측정 결과는 155/95 mmHg였으며, 오전 중에 일시적으로 145/90 mmHg로 감소하는 경향을 보임. 환자는 불규칙적인 혈압 변화에 대해 우려를 표현함. 환자의 혈압 변동과 관련하여 생활 습관, 식이, 약물 복용 패턴 등을 면밀히 평가함.\n\n간호 중재로는 혈압 모니터링을 매 4시간마다 실시하고, 의사 지시에 따라 혈압 조절 약물을 투여함. 환자에게 약물 복용 일정을 관리할 수 있는 방법을 안내하고, 저염식 식단과 규칙적인 운동의 중요성에 대해 교육함.\n\n오후 혈압 측정 결과는 150/92 mmHg로 다소 안정된 경향을 보임. 환자는 제공된 교육 내용에 대해 긍정적인 반응을 보이며, 약물 복용과 생활 습관 개선에 대한 의지를 표현함. 향후 계획으로는 지속적인 혈압 모니터링, 약물 관리 및 환자 교육을 이어갈 예정임."}
            ]
        
        def chat(text):
          user_input =  {"role":"user","content": text}
          prompt.append(user_input)
          answer = client.chat.completions.create(
            model="NursingAI",
            messages = prompt,
            temperature=0.2,
            max_tokens=1200,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        
          bot_answ = answer.choices[0].message.content
          bot_resp = {"role":"assistant","content": bot_answ}
          prompt.append(bot_resp)
          return bot_answ
        
        completion = client.chat.completions.create(
          model="NursingAI",
          messages = prompt,
          temperature=0.7,
          max_tokens=800,
          top_p=0.95,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )
        
        user_questions = [
            "낙상 환자 간호기록 써줘.",
            "비효과적 호흡 양사 환자의 Focus DAR 간호기록 작성해줘.",
            "급성 통증 환자 간호기록 예시를 SOAPIE 기록양식으로 작성해줘."
        ]
        
        candidates = [] # candidates ["First sentence", "Second sentence"]
        for question in user_questions:
          response = chat(question)
          candidates.append(response)
        
        # 후보군 리스트 출력
        for i, candidate in enumerate(candidates):
          print(f"Question {i+1} : {user_questions[i]}")
          print(f"Answer: {candidate}\n")
        ```
        
    - 결과 예시
        
        Narrative Note(양식 비지정)
        
        ![스크린샷 2023-12-14 오후 3.12.15.png](Smart_nurse_Chatbot/narrnote.png)
        
        Focus DAR
        
        ![스크린샷 2023-12-14 오후 3.12.38.png](Smart_nurse_Chatbot/focusdar.png)
        
        SOAPIE
        
        ![스크린샷 2023-12-14 오후 3.12.49.png](Smart_nurse_Chatbot/soapie.png)
        

### c. LLM 학습

- GPT LLM 학습
    - 모델 성능
        
        > epoch = 3, 일괄처리크기 = 4, 학습 속도 승수 = 1
        > 
        
        Loss
        
        ![그림1.png](Smart_nurse_Chatbot/model.png)
        

**이슈 & 한계점**

- 우리가 가진 간호기록 데이터로 학습시킨 LLM 모델을 deploy 해도 prompting 이 의도한 대로 반영되지 않음
    - 통일되지 않은 항목명
    - 목차 형식 혼재(번호순, bullet point 등)
    - 비효율적인 ‘-입니다’ 등의 만연체
    - 전문성이 느껴지지 않는 포괄적인 답변
    - 간호 관련 질문이 아닌 경우에도 임의의 간호기록 제공
    - **해결한 방법**
        - prompt instruction 을 더욱 정교화
            
            ```python
            instruction_NANDA = """
            Persona:
            - You are a nursing assistant with the knowledge equivalent to a nurse with 10 years of experience.
            - When the user mentions a diagnosis, write a nursing record for that specific condition.
            - If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.
            
            Instructions:
            - All responses should be in Korean.
            - Exclude any theoretical basis or guideline in '간호수행/중재/이론적 근거 Interventions' section.
            - Write '단기목표' and '장기목표' as substructures under the '간호목표 단기/장기 Goal' section.
            - The content of each section should be specific and include precise numerical information. Instead of a general response like '적절한 호흡 기법 교육,' provide specific details such as '호흡 운동 교육: 깊고 천천히 숨쉬기, 복식호흡법 교육.' Include exact figures, for example, '환자의 호흡수가 24시간 이내에 분당 12~20회로 정상 범위 내로 감소.'
            - End all responses with a noun or a nominal ending. Change endings such as '습니다', '합니다', '했다', '한다', '입니다', '있다', '됩니다', '된다' to '음', '함', '함', '함', '임', '있음', '됨', '됨' respectively.
            - Organize patient information and related sections in a list format, with line breaks. Each item should start with a bullet point and each section should contain only one substructure.
            - Write nursing records following the specified format below.
            ```
            
        - 프롬프트, 인스트럭션 fine-tuning 방향: 간호 실무 현장을 고려해 전문적이면서도 효율적이고 간편한 답변 제공
            - instruction 을 영어로 작성
            - ai에 페르소나 부여(10년차 간호사)
            - 기록양식별 작성 형식 통일(SmartNurse)
            - 지시 내용 구체화 (양식 별 반드시 포함해야 하는 내용 명시 등)
            - 간결하고 전문성이 느껴지는 말투(명사형 종결어미 사용)
            - 최대한 구체적으로 기록(수치, 정확한 조치명 사용 등)
            - 간호 관련 질문이 아닌 경우 → 간호기록에 대한 질문을 할 수 있도록 유도
            - LLM 기반 데이터가 아닌 정보 무단 제공(Hallucination 우려)
        - 학습 데이터셋의 질 향상 : prompt tuning 에 사용한 이상적인 답변 데이터셋을 추가하여 LLM fine-tuning
- 모델 성능평가가 어려움 : LLM 모델에 대한 성능평가 방법에 현실적인 한계가 존재하며,  우리 task와 같은 양식에 맞게 생성한 답변 결과를 평가할 정량지표가 부재.
    - 전문가에 의한 생성 답변 정확성 평가
    - 또는 여러개의 LLM 모델에 같은 prompt instruction 을 주었을 때 생성된 결과를 서로 비교하는 방법 등을 사용할 수 있음.

---

# 4. 서비스 구현 및 배포

### a. Chatbot : Langchain

- 랭체인을 활용한 챗봇 코드 구현
    - fine-tuning LLM 모델에 기록 양식별 few shot prompt 학습 코드 구현
    - 예시 : NANDA
        
        ```python
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
        - If the user asks questions not related to nursing records, respond in a way that guides them to ask questions about nursing-related topics.
        
        Instructions:
        - All responses should be in Korean.
        - Exclude any theoretical basis or guideline in '간호수행/중재/이론적 근거 Interventions' section.
        - Write '단기목표' and '장기목표' as substructures under the '간호목표 단기/장기 Goal' section.
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
        
        자료 수집 주관적 / 객관적
        - 하위구조
        
        간호목표 단기/장기 Goal
        - 하위구조
        
        간호계획 Plan
        - 하위구조
        
        간호수행/중재/이론적 근거 Interventions
        - 하위구조
        
        간호평가 Evaluation
        - 하위구조
        ###
        """
        final_prompt_NANDA = ChatPromptTemplate.from_messages(
            [
                ("system", instruction_NANDA),
                few_shot_prompt_NANDA,
                ("human", "{input}"),
            ]
        )
        ```
        

### b. Webapp deployment : Streamlit

- 웹앱 배포를 위한 Streamlit 코드 구현
    - chat-input
    - 코드
        
        ```python
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
        
        ###랭체인 프롬프트 코드 삽입### 
        
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
        
        ```
        
    - chatbot UI
    - 코드
        
        ```python
        # 웹 탭 꾸미기
        st.set_page_config(
            page_title="SmartNurse ChatBot",
            page_icon="👩‍⚕️",
        
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
        ```
        
- 데모
    
    ![챗봇 메인](Smart_nurse_Chatbot/chatbot_page.png)
    
    챗봇 메인
    
    ![사용자 입력 쿼리 → 예시 생성](Smart_nurse_Chatbot/chatbot_ex.png)
    
    사용자 입력 쿼리 → 예시 생성
    

[웹앱 실행 데모 영상](Smart_nurse_Chatbot/final_(1).mp4)

웹앱 실행 데모 영상

**이슈 & 한계점**

수많은 에러와의 싸움………디버깅디버깅디버깅

###
