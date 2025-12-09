import streamlit as st
import pandas as pd
import requests 
import google.generativeai as genai
import json
import re
import time
from urllib.parse import urlparse, parse_qs

# =============================================================================
# 🔐 보안 설정 (Security Setup)
# =============================================================================
# API 키는 이제 코드에 직접 적지 않고, Streamlit의 'Secrets' 기능을 통해 불러옵니다.
# 로컬 실행 시: .streamlit/secrets.toml 파일 필요
# 클라우드 실행 시: Streamlit Cloud 대시보드의 Secrets 메뉴 설정 필요

try:
    NOTION_API_KEY = st.secrets["NOTION_API_KEY"]
    NOTION_DATABASE_ID = st.secrets["NOTION_DATABASE_ID"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("⚠️ API 키가 설정되지 않았습니다.")
    st.info("""
    **[설정 방법]**
    1. **로컬 실행 시**: 프로젝트 폴더 안에 `.streamlit` 폴더를 만들고 `secrets.toml` 파일을 생성하여 키를 입력하세요.
    2. **웹 배포 시**: Streamlit Cloud 설정 페이지의 **Secrets** 란에 키를 복사해 넣으세요.
    """)
    st.stop()

# -----------------------------------------------------------------------------
# 노션 공개 주소
PUBLIC_NOTION_DOMAIN = "greation83.notion.site"
PUBLIC_NOTION_URL = f"https://{PUBLIC_NOTION_DOMAIN}/2c1576d96adb80bab598f4232e364f3f?v=2c1576d96adb80bba8dc000cee9827e8"

# =============================================================================
# 초기화
# =============================================================================

st.set_page_config(layout="wide", page_title="설교 비서 Pro (Cloud)")

# 유료 키 설정
if GEMINI_API_KEY.startswith("AIza"):
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ API 키 형식이 올바르지 않습니다.")

# =============================================================================
# AI 프롬프트 (심층 분석 및 GBS)
# =============================================================================

ANALYSIS_PROMPT = """
당신은 20년 차 설교학 교수이자 예화 전문가입니다. 
다음 설교 초안을 **깊이 있게 분석**하여 이 설교에 필요한 예화의 조건을 도출해주세요.

## 설교 초안:
{draft}

## 분석 요청 항목:
1. **핵심주제**: 설교를 관통하는 핵심 키워드 5개 (명사형)
2. **감정선**: 이 설교의 주된 정서 (예: 위로, 도전, 회개, 감사, 경고, 유머)
3. **연관성경**: 설교와 연관된 성경 권(Book) 이름 (예: 창세기, 마태복음)
4. **설교요약**: 설교의 핵심 메시지를 2문장으로 요약

## 출력 형식 (JSON):
{{
    "핵심주제": ["주제1", "주제2", ...],
    "감정선": ["감정1", "감정2"],
    "연관성경": ["창세기"],
    "설교요약": "요약문..."
}}
"""

FEEDBACK_PROMPT = """
당신은 20년 차 설교학 교수이자, 청중의 삶을 변화시키는 **'실천적 적용(Application)'의 대가**입니다.
제자가 작성한 설교 초안을 검토하고, 특히 **'실천적 적용'이 약하거나 추상적인 부분을 아주 구체적으로 보완**해주세요.

## 설교 초안:
{draft}

## 피드백 요청 항목:
1. **논리적 점검**: 
   - 논리적 비약이 있는 부분이나 성경 해석의 무리수 점검
2. **구체적 행동 제안 (Action Plan)**: 
   - **(가장 중요)** 설교자의 적용이 추상적("사랑하자", "나아가자", "섬기자")이라면, 이를 **청중이 오늘 당장 실행할 수 있는 구체적인 행동**으로 바꿔 제안하세요.
   - 뜬구름 잡는 이야기는 하지 마세요. 대상, 장소, 금액, 행동을 명시하세요.
   - **나쁜 예**: "소외된 이웃을 돌봅시다."
   - **좋은 예**: "우리 아파트 경비원분께 따뜻한 음료수 한 병을 건네며 '감사합니다'라고 인사합시다.", "오늘 하루 커피 한 잔 값(5,000원)을 아껴 미혼모 시설이나 구호 단체에 기부합시다."
3. **강점 칭찬**:
   - 설교에서 가장 훌륭한 통찰이나 표현

## 출력 형식 (JSON):
{{
    "논리점검": ["지적사항1", "지적사항2"],
    "보완제안": ["구체적 행동 제안1", "구체적 행동 제안2", "구체적 행동 제안3"],
    "강점": "이 설교의 훌륭한 점..."
}}
"""

RECOMMENDATION_PROMPT = """
당신은 설교 작성자를 돕는 조수입니다.
설교 초안의 내용과 흐름을 고려할 때, 아래 후보 예화들 중 가장 적절한 것을 추천해주세요.

## 설교 요약:
{sermon_summary}

## 설교 감정선/주제:
{sermon_tags}

## 후보 예화 목록:
{candidates}

## 요청:
가장 잘 어울리는 예화 **10개에서 15개**를 선정하여 이유와 함께 알려주세요.
단순히 키워드가 같아서가 아니라, **설교의 맥락(Context)을 살려줄 수 있는 것**을 고르세요.
예를 들어, 설교가 '고난 중의 인내'를 다룬다면, '가벼운 유머'보다는 '깊이 있는 간증'이나 '역사적 사례'를 추천하세요.

## 중요: 
반드시 후보 예화 목록에 있는 **'번호(ID)'**를 함께 출력해주세요.

## 출력 형식 (JSON):
{{
    "추천목록": [
        {{
            "번호": 1, 
            "제목": "예화 제목",
            "추천이유": "이 예화는 설교의 [어떤 부분]에서 [어떤 효과]를 줄 수 있어 추천합니다. (구체적으로)",
            "활용팁": "서론 예화로 사용하거나, 결론부 적용 질문으로 던지기 좋습니다."
        }},
        ...
    ]
}}
"""

GBS_PROMPT_TEMPLATE = """
## 역할 정의 (Role)
당신은 20년 경력의 베테랑 성경 교재 집필가이자 **{target_dept} 전문가**입니다. 
당신의 임무는 사용자가 입력한 [설교 초안]을 바탕으로, **{target_dept}** 구성원들이 소그룹에서 깊이 있게 나눌 수 있는 **'맞춤형 GBS(Group Bible Study) 교재'**를 제작하는 것입니다.

## 타겟 오디언스 (Target Audience)
* 대상: **{target_dept}** ({age_range})
* 특징: {dept_characteristics}

## 작업 목표 (Objective)
설교의 핵심 메시지를 훼손하지 않으면서, {target_dept}가 지루해하지 않고 삶에 구체적으로 적용할 수 있는 **단 하나의 깔끔한 교재**를 생성하십시오.
**복잡한 서식(표, 박스 등)을 피하고, 아이콘(이모지)과 텍스트로만 구성하여 복사+붙여넣기가 쉽도록 작성하십시오.**

## 출력 형식 (Output Format)
반드시 아래 목차에 따라 작성하며, **중복된 내용을 절대 출력하지 마십시오.** (별도의 요약본이나 노션 폼용 버전을 만들지 마세요.)

---
### **[제목: (설교 제목을 재치 있게 변형)]**

### **1. 🧊 아이스브레이크 (Ice Break)**
* 설교 주제와 연결된, 마음을 여는 가벼운 질문 2개

### **2. 📖 말씀 속으로 (Observation)**
* 본문 관찰 및 해석 질문 3개 (깊이 있는 통찰 유도)

### **3. 🏃‍♂️ 삶으로 (Application)**
* **Apply (진단):** 나의 상태를 점검하는 질문 2~3개
* **Break (깨달음):** 본문이 주는 교훈 한 문단 (4~5줄)
* **Build (실천):** 이번 주 바로 할 수 있는 구체적 실천 3~5개 (관계, 재정, 시간 등 구체적 영역 언급)
* **Pray (기도):** 함께 읽을 수 있는 짧은 기도문

### **4. 🧠 팝 퀴즈 (Pop Quiz)**
* 객관식 3문제 + OX 2문제
* **(정답은 퀴즈 바로 아래에 작게 표기)**

### **5. 🌊 플로잉 위크 (한 주간의 미션)**
* **월(See):** (미션 내용)
* **화(Speak):** (미션 내용)
* **수(Cost):** (미션 내용)
* **목(Listen):** (미션 내용)
* **금(Act):** (미션 내용)
* **토(Pray):** (미션 내용)
---

## 톤앤매너
* {tone_manner}
* 설명조보다는 대화체로 자연스럽게.

## 설교 초안:
{draft}
"""

# =============================================================================
# 함수 정의
# =============================================================================

def get_gemini_response(prompt, model_name='gemini-2.5-flash'):
    """Gemini API 호출 (일반 텍스트 반환)"""
    try:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return None

def get_gemini_json(prompt):
    """Gemini API 호출 (JSON 반환)"""
    text = get_gemini_response(prompt)
    if text:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except: 
                pass
    return None

def convert_to_public_url(page_id):
    if not page_id: return PUBLIC_NOTION_URL
    clean_id = page_id.replace("-", "")
    return f"https://{PUBLIC_NOTION_DOMAIN}/{clean_id}"

def extract_start_time(url):
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if 't' in qs: return int(qs['t'][0])
    except: pass
    return 0

# 🔥 함수 이름 변경으로 강제 캐시 갱신 (v6 - DB 변경)
@st.cache_data(ttl=3600) 
def fetch_all_illustrations_v6():
    """Notion에서 모든 예화 데이터 가져오기"""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {NOTION_API_KEY}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    
    results = []
    has_more = True
    next_cursor = None

    with st.spinner("📚 서재(Notion)에서 최신 데이터를 가져오는 중..."):
        while has_more:
            try:
                payload = {"filter": {"property": "종류", "select": {"equals": "예화"}}, "page_size": 100}
                if next_cursor: payload["start_cursor"] = next_cursor
                
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code != 200: break
                
                data = response.json()
                results.extend(data.get('results', []))
                has_more = data.get('has_more', False)
                next_cursor = data.get('next_cursor', None)
            except: break
    
    processed_data = []
    for page in results:
        props = page.get('properties', {})
        title_prop = props.get('title', {}).get('title', [])
        title = title_prop[0].get('plain_text', "") if title_prop else ""
        
        subjects = []
        subject_prop = props.get('주제', {})
        prop_type = subject_prop.get('type')

        if prop_type == 'multi_select':
            subjects = [item.get('name', "") for item in subject_prop.get('multi_select', [])]
        elif prop_type == 'rich_text':
            text_list = subject_prop.get('rich_text', [])
            if text_list:
                raw_text = text_list[0].get('plain_text', "")
                subjects = [s.strip() for s in raw_text.split(',') if s.strip()]

        emotions = [item.get('name', "") for item in props.get('감정톤', {}).get('multi_select', [])]
        
        summary_prop = props.get('예화요약', {}).get('rich_text', [])
        summary = summary_prop[0].get('plain_text', "") if summary_prop else ""
        
        source_url = props.get('URL', {}).get('url', "") 
            
        processed_data.append({
            "id": page['id'],
            "title": title,
            "subjects": subjects,
            "emotions": emotions,
            "summary": summary,
            "url": page.get('url', ""),
            "source_url": source_url
        })
    return processed_data

# 🔥 함수 이름 변경으로 강제 캐시 갱신 (v6 - DB 변경)
@st.cache_data(ttl=3600)
def fetch_page_content_v6(page_id):
    """특정 페이지의 본문(Block) 내용을 가져와 텍스트로 변환"""
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {"Authorization": f"Bearer {NOTION_API_KEY}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    
    content_text = ""
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            blocks = response.json().get('results', [])
            for block in blocks:
                b_type = block.get('type')
                text_content = ""
                
                if b_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3', 'bulleted_list_item', 'numbered_list_item', 'callout', 'quote']:
                    rich_text = block.get(b_type, {}).get('rich_text', [])
                    for rt in rich_text:
                        text_content += rt.get('plain_text', "")
                    
                    if text_content:
                        if b_type == 'heading_1': content_text += f"\n# {text_content}\n"
                        elif b_type == 'heading_2': content_text += f"\n## {text_content}\n"
                        elif b_type == 'heading_3': content_text += f"\n### {text_content}\n"
                        elif b_type == 'bulleted_list_item': content_text += f"• {text_content}\n"
                        elif b_type == 'numbered_list_item': content_text += f"1. {text_content}\n"
                        elif b_type == 'quote': content_text += f"\n> {text_content}\n"
                        else: content_text += f"{text_content}\n\n"
        else: return "(본문을 가져오는데 실패했습니다.)"
    except: return "(본문 로드 중 오류 발생)"
    return content_text if content_text else "(본문 내용이 없습니다.)"

def calculate_relevance_score(illustration, sermon_analysis):
    score = 0
    if not illustration['subjects'] or not sermon_analysis.get('핵심주제'): pass 
    else:
        matches = set(illustration['subjects']) & set(sermon_analysis['핵심주제'])
        score += len(matches) * 5
    if illustration['emotions'] and sermon_analysis.get('감정선'):
        emo_matches = set(illustration['emotions']) & set(sermon_analysis['감정선'])
        score += len(emo_matches) * 3
    full_text = (illustration['title'] + " " + illustration['summary']).replace(" ", "")
    for keyword in sermon_analysis.get('핵심주제', []):
        if keyword in full_text: score += 1
    return score

# =============================================================================
# 메인 UI
# =============================================================================

def main():
    with st.sidebar:
        st.markdown("### 🕊️ Sermon Assistant Pro")
        st.info("유료 API 모드 (Cloud)")
        st.markdown("---")
        st.link_button("📚 전체 예화 도서관(Notion) 가기", PUBLIC_NOTION_URL)
        if st.button("🔄 데이터 새로고침 (캐시 삭제)"):
            st.cache_data.clear()
            st.rerun()

    st.title("🕊️ 설교 비서: 예화 & GBS 메이커")
    st.markdown("설교 초안을 넣으면 **예화 추천, 설교 클리닉, 그리고 소그룹 교재**까지 한 번에 제작합니다.")

    with st.expander("ℹ️ 사용 가이드"):
        st.markdown("""
        1. **설교 입력**: 설교 원고를 붙여넣으세요.
        2. **부서 선택**: 교재를 만들 대상을 선택하세요 (청년부 등).
        3. **분석 시작**: 버튼을 누르면 모든 작업이 자동으로 진행됩니다.
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 설교 초안 및 설정")
        target_dept = st.selectbox("교재 제작 대상 (부서)", ["청년부", "장년부", "중고등부", "유초등부"])
        sermon_draft = st.text_area("설교 본문 붙여넣기", height=600, placeholder="본문을 입력하면 예화도 찾고 교재도 만들어드립니다.")
        
        analyze_btn = st.button("🚀 분석 및 교재 생성 시작", type="primary")

    if analyze_btn and sermon_draft:
        # v6 함수 사용 (캐시 갱신용)
        illustrations = fetch_all_illustrations_v6()
        
        with col2:
            st.subheader("📊 분석 결과")
            
            # 1. 설교 분석
            with st.status("🔍 설교를 분석하고 주제를 추출합니다...") as status:
                analysis_result = get_gemini_json(ANALYSIS_PROMPT.format(draft=sermon_draft))
                if not analysis_result:
                    st.error("분석 실패: API 키를 확인해주세요.")
                    return
                status.update(label="✅ 설교 분석 완료!", state="complete")
            
            # 2. 예화 추천
            with st.status("📚 서재에서 가장 적절한 예화를 찾습니다...") as status:
                top_candidates = []
                if illustrations:
                    scored_candidates = []
                    for illust in illustrations:
                        score = calculate_relevance_score(illust, analysis_result)
                        # score > 0 조건 삭제: 점수가 0이어도 AI에게 판단을 맡기기 위함
                        illust['score'] = score
                        scored_candidates.append(illust)
                    
                    # 점수순 정렬 후 상위 30개 (점수가 모두 0이라도 상위 30개를 가져감)
                    top_candidates = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)[:30]

                recommendation_result = None
                if top_candidates:
                    candidates_text = ""
                    for idx, cand in enumerate(top_candidates):
                        candidates_text += f"{idx+1}. 제목: {cand['title']} | 요약: {cand['summary']} | 태그: {cand['subjects']}\n"
                    
                    curation_prompt = RECOMMENDATION_PROMPT.format(
                        sermon_summary=analysis_result['설교요약'],
                        sermon_tags=f"주제: {analysis_result['핵심주제']}, 감정: {analysis_result['감정선']}",
                        candidates=candidates_text
                    )
                    recommendation_result = get_gemini_json(curation_prompt)
                status.update(label="✅ 예화 추천 완료!", state="complete")

            # 3. 피드백 & GBS 생성
            with st.status(f"✍️ {target_dept} 맞춤형 교재와 피드백을 작성 중입니다...") as status:
                feedback_result = get_gemini_json(FEEDBACK_PROMPT.format(draft=sermon_draft))
                
                # 부서별 프롬프트 설정
                if target_dept == "청년부":
                    age_range = "20~30대 대학생/직장인"
                    dept_characteristics = "권위적인 가르침보다 진정성 있는 나눔 선호, 구체적 삶의 적용 원함"
                    tone_manner = "친근하고 위트 있으면서도 핵심을 찌르는 어조 (MZ/Alpha 감성)"
                    quiz_difficulty = "중학생~청년 초신자도 풀 수 있는 수준"
                elif target_dept == "장년부":
                    age_range = "40~60대 성인"
                    dept_characteristics = "삶의 연륜이 있으며 가정과 직장의 무게를 견디는 세대, 깊은 위로와 통찰 필요"
                    tone_manner = "정중하고 깊이 있으며 목회적 돌봄이 느껴지는 어조"
                    quiz_difficulty = "성경 지식을 확인할 수 있는 수준"
                elif target_dept == "중고등부":
                    age_range = "10대 청소년"
                    dept_characteristics = "학업 스트레스와 정체성 고민, 짧고 임팩트 있는 메시지 선호"
                    tone_manner = "에너지 넘치고 짧고 간결한 어조"
                    quiz_difficulty = "쉽고 재미있는 수준"
                else: # 유초등부
                    age_range = "초등학생"
                    dept_characteristics = "활동적이고 쉬운 언어 필요, 스토리텔링 중요"
                    tone_manner = "다정하고 쉬운 선생님 말투 (존댓말 사용)"
                    quiz_difficulty = "아주 쉬움 (OX 퀴즈 위주)"

                gbs_prompt = GBS_PROMPT_TEMPLATE.format(
                    target_dept=target_dept,
                    age_range=age_range,
                    dept_characteristics=dept_characteristics,
                    tone_manner=tone_manner,
                    quiz_difficulty=quiz_difficulty,
                    draft=sermon_draft
                )
                
                gbs_content = get_gemini_response(gbs_prompt)
                status.update(label="✅ 모든 작업 완료!", state="complete")

            # === 결과 탭 ===
            tab1, tab2, tab3 = st.tabs(["🤖 AI 추천 예화", "✍️ GBS 교재 (복사용)", "👨‍🏫 설교 클리닉"])
            
            with tab1:
                st.info(f"**💡 설교 요약:** {analysis_result.get('설교요약', '')}")
                if recommendation_result and recommendation_result.get('추천목록'):
                    for idx, rec in enumerate(recommendation_result['추천목록']):
                        original_data = None
                        if '번호' in rec and isinstance(rec['번호'], int):
                            try:
                                candidate_index = rec['번호'] - 1
                                if 0 <= candidate_index < len(top_candidates):
                                    original_data = top_candidates[candidate_index]
                            except: pass
                        if not original_data:
                            original_data = next((item for item in top_candidates if item["title"] == rec["제목"]), None)

                        with st.container():
                            st.markdown(f"#### 📌 {rec['제목']}")
                            st.write(f"**🗣️ 추천 이유:** {rec['추천이유']}")
                            st.caption(f"**💡 활용 팁:** {rec['활용팁']}")
                            if original_data:
                                with st.expander("📖 예화 본문 & 영상 보기", expanded=(idx == 0)):
                                    with st.spinner("본문을 불러오는 중..."):
                                        if original_data.get('source_url'):
                                            start_time = extract_start_time(original_data['source_url'])
                                            st.markdown(f"**📺 관련 설교 영상 (시작 시간: {start_time}초)**")
                                            st.video(original_data['source_url'], start_time=start_time)
                                            
                                            # [추가] 영상 시간 불일치 시 해결 팁
                                            st.info("""
                                            💡 **예화와 영상 시간이 맞지 않는 경우에는**
                                            1. 영상 링크를 누른 후 
                                            2. 조회 수가 적혀 있는 박스 아래 쪽 **"더보기"** 클릭
                                            3. 박스 밑에 파란 글씨로 **"스크립트 표시"**를 클릭
                                            4. `ctrl+F` 를 눌러 예화 내용의 단어들을 검색하면 예화를 들을 수 있습니다. 
                                            """)
                                            
                                        st.divider()
                                        content_text = fetch_page_content_v6(original_data['id'])
                                        st.markdown(content_text)
                                        st.divider()
                                        public_url = convert_to_public_url(original_data['id'])
                                        st.link_button("🔗 노션 페이지 열기", public_url)
                            st.divider()
                else:
                    st.write("추천된 예화가 없습니다.")
            
            with tab2:
                st.markdown(f"### 📖 {target_dept} 맞춤형 소그룹 교재")
                st.info("우측 상단의 'Copy' 아이콘을 누르면 전체 내용을 한 번에 복사할 수 있습니다.")
                st.code(gbs_content, language='markdown') 
                st.markdown("---")
                st.markdown(gbs_content) 

            with tab3:
                if feedback_result:
                    st.markdown("### 📢 설교 논리 & 전달력 클리닉")
                    st.success(f"**👍 강점:** {feedback_result.get('강점', '훌륭한 설교입니다.')}")
                    st.markdown("#### ⚠️ 논리적 점검")
                    for point in feedback_result.get('논리점검', []): st.markdown(f"- {point}")
                    st.markdown("#### 🏃‍♂️ 구체적 행동 제안 (Action Plan)")
                    for point in feedback_result.get('보완제안', []): st.markdown(f"- {point}")

if __name__ == "__main__":
    main()