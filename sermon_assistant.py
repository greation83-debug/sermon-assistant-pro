import streamlit as st
import pandas as pd
import requests 
import google.generativeai as genai
import json
import re
import time
import numpy as np
from urllib.parse import urlparse, parse_qs

# =============================================================================
# 🔐 보안 설정 (Security Setup)
# =============================================================================

try:
    NOTION_API_KEY = st.secrets["NOTION_API_KEY"]
    NOTION_DATABASE_ID = st.secrets["NOTION_DATABASE_ID"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except (FileNotFoundError, KeyError) as e:
    st.error(f"⚠️ API 키가 설정되지 않았습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 노션 공개 주소
PUBLIC_NOTION_DOMAIN = "greation83.notion.site"
PUBLIC_NOTION_URL = f"https://{PUBLIC_NOTION_DOMAIN}/2c1576d96adb80bab598f4232e364f3f?v=2c1576d96adb80bba8dc000cee9827e8"

# 임베딩 설정
EMBEDDING_MODEL = "models/text-embedding-004"

# =============================================================================
# 초기화
# =============================================================================

st.set_page_config(layout="wide", page_title="설교 비서 Pro")

if GEMINI_API_KEY.startswith("AIza"):
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ API 키 형식이 올바르지 않습니다.")

# =============================================================================
# Supabase 함수
# =============================================================================

def get_supabase_headers():
    """Supabase API 헤더"""
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }


def get_illustration_count():
    """Supabase에 저장된 예화 개수 조회"""
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/illustrations?select=id",
            headers={**get_supabase_headers(), "Prefer": "count=exact"},
        )
        count = response.headers.get('content-range', '').split('/')[-1]
        return int(count) if count else 0
    except:
        return 0


def get_random_illustrations(count=30):
    """랜덤 예화 가져오기 (fallback용)"""
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/illustrations?select=id,title,summary,subjects,emotions,source_url,preacher&limit={count}",
            headers=get_supabase_headers(),
        )
        if response.status_code == 200:
            results = response.json()
            # similarity 필드 추가 (fallback이므로 0.5로 설정)
            for item in results:
                item['similarity'] = 0.5
            return results
        return []
    except:
        return []


def semantic_search_supabase(query_embedding, top_k=30):
    """Supabase 벡터 검색"""
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_illustrations",
            headers=get_supabase_headers(),
            json={
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        )
        
        # 디버깅 로그
        st.sidebar.write(f"🔍 검색 응답: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            st.sidebar.write(f"🔍 검색 결과 수: {len(results) if results else 0}")
            
            if results:
                return results
            else:
                # 결과가 비어있으면 fallback
                st.warning("⚠️ 벡터 검색 결과 없음, 기본 예화를 가져옵니다.")
                return get_random_illustrations(top_k)
        else:
            st.warning(f"⚠️ 검색 오류 ({response.status_code}): {response.text[:200]}")
            return get_random_illustrations(top_k)
    except Exception as e:
        st.warning(f"⚠️ 검색 실패: {e}, 기본 예화를 가져옵니다.")
        return get_random_illustrations(top_k)


def get_query_embedding(text):
    """검색 쿼리용 임베딩 생성"""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        st.warning(f"⚠️ 쿼리 임베딩 생성 실패: {e}")
        return None


# =============================================================================
# AI 프롬프트
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
- 반드시 후보 예화 목록에 있는 **'번호(ID)'**를 함께 출력해주세요.
- **동일한 설교자의 예화는 최대 3개**까지만 추천하세요. 다양한 출처의 예화를 선정해주세요.

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
# 기타 함수
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


def get_gemini_json(prompt, max_retries=3):
    """Gemini API 호출 (JSON 반환) - 딜레이 + 재시도 포함"""
    for attempt in range(max_retries):
        # 딜레이 추가 (첫 시도 제외)
        if attempt > 0:
            time.sleep(2)
        
        text = get_gemini_response(prompt)
        if text:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        
        # 재시도 전 대기
        if attempt < max_retries - 1:
            time.sleep(1)
    
    return None


def convert_to_public_url(page_id):
    if not page_id:
        return PUBLIC_NOTION_URL
    clean_id = page_id.replace("-", "")
    return f"https://{PUBLIC_NOTION_DOMAIN}/{clean_id}"


def extract_start_time(url):
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if 't' in qs:
            return int(qs['t'][0])
    except:
        pass
    return 0


@st.cache_data(ttl=3600)
def fetch_page_content(page_id):
    """특정 페이지의 본문(Block) 내용을 가져와 텍스트로 변환"""
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    content_text = ""
    in_related_section = False  # 관련예화 섹션인지 추적
    
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
                        plain_text = rt.get('plain_text', "")
                        
                        # 링크가 있는 경우 마크다운 링크로 변환
                        href = rt.get('href', '')
                        if href and 'notion' in href:
                            # 노션 링크를 공개 URL로 변환
                            # 노션 링크에서 페이지 ID 추출
                            link_page_id = href.split('/')[-1].split('?')[0].split('-')[-1]
                            if len(link_page_id) == 32:
                                public_link = f"https://{PUBLIC_NOTION_DOMAIN}/{link_page_id}"
                                text_content += f"[{plain_text}]({public_link})"
                            else:
                                text_content += plain_text
                        elif href:
                            text_content += f"[{plain_text}]({href})"
                        else:
                            text_content += plain_text
                    
                    if text_content:
                        # "관련예화" 섹션 감지
                        if "관련예화" in text_content or "관련 예화" in text_content:
                            in_related_section = True
                            content_text += f"\n---\n### 🔗 {text_content}\n"
                        # "핵심내용" 감지
                        elif "핵심내용" in text_content or "핵심 내용" in text_content:
                            in_related_section = False
                            content_text += f"\n### 📌 {text_content}\n"
                        # 관련예화 섹션이 끝나고 본문 시작 (보통 긴 텍스트)
                        elif in_related_section and len(text_content) > 100:
                            in_related_section = False
                            content_text += f"\n---\n### 📖 예화 본문\n\n{text_content}\n\n"
                        elif b_type == 'heading_1':
                            content_text += f"\n# {text_content}\n"
                        elif b_type == 'heading_2':
                            content_text += f"\n## {text_content}\n"
                        elif b_type == 'heading_3':
                            content_text += f"\n### {text_content}\n"
                        elif b_type == 'bulleted_list_item':
                            content_text += f"• {text_content}\n"
                        elif b_type == 'numbered_list_item':
                            content_text += f"1. {text_content}\n"
                        elif b_type == 'quote':
                            content_text += f"\n> {text_content}\n"
                        else:
                            content_text += f"{text_content}\n\n"
        else:
            return "(본문을 가져오는데 실패했습니다.)"
    except:
        return "(본문 로드 중 오류 발생)"
    return content_text if content_text else "(본문 내용이 없습니다.)"


# =============================================================================
# 메인 UI
# =============================================================================

def main():
    # Supabase에서 예화 개수 확인
    illustration_count = get_illustration_count()
    
    with st.sidebar:
        st.markdown("### 🕊️ Sermon Assistant Pro")
        st.info("Supabase 벡터 검색 v3.0")
        st.markdown("---")
        st.caption(f"📊 예화 DB: {illustration_count:,}개")
        
        if st.button("🔄 캐시 새로고침"):
            st.cache_data.clear()
            st.rerun()

    st.title("🕊️ 설교 비서: 예화 & GBS 메이커")
    st.markdown("설교 초안을 넣으면 **예화 추천, 설교 클리닉, 그리고 소그룹 교재**까지 한 번에 제작합니다.")

    # DB 체크
    if illustration_count == 0:
        st.error("⚠️ Supabase에 예화 데이터가 없습니다.")
        return

    # 정상 UI
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
        with col2:
            st.subheader("📊 분석 결과")
            
            # 1. 설교 분석
            with st.status("🔍 설교를 분석하고 주제를 추출합니다...") as status:
                analysis_result = get_gemini_json(ANALYSIS_PROMPT.format(draft=sermon_draft))
                if not analysis_result:
                    st.error("분석 실패: API 키를 확인해주세요.")
                    return
                status.update(label="✅ 설교 분석 완료!", state="complete")
            
            time.sleep(1)  # API Rate Limit 방지
            
            # 2. 예화 추천 (Supabase 벡터 검색)
            with st.status("📚 의미 기반으로 가장 적절한 예화를 찾습니다...") as status:
                search_query = analysis_result.get('설교요약', '')
                if analysis_result.get('핵심주제'):
                    search_query += " " + " ".join(analysis_result['핵심주제'])
                if analysis_result.get('감정선'):
                    search_query += " " + " ".join(analysis_result['감정선'])
                
                # 쿼리 임베딩 생성
                query_embedding = get_query_embedding(search_query)
                st.sidebar.write(f"🔍 임베딩 생성: {'✅' if query_embedding else '❌'}")
                
                top_candidates = []
                if query_embedding:
                    top_candidates = semantic_search_supabase(query_embedding, top_k=30)
                else:
                    # 임베딩 실패시 fallback
                    st.warning("⚠️ 임베딩 생성 실패, 기본 예화를 가져옵니다.")
                    top_candidates = get_random_illustrations(30)
                
                st.sidebar.write(f"🔍 후보 예화 수: {len(top_candidates)}")

                recommendation_result = None
                if top_candidates:
                    candidates_text = ""
                    for idx, cand in enumerate(top_candidates):
                        preacher_info = f" | 설교자: {cand.get('preacher', '미상')}" if cand.get('preacher') else ""
                        candidates_text += f"{idx+1}. 제목: {cand['title']} | 요약: {cand.get('summary', '')} | 태그: {cand.get('subjects', [])}{preacher_info}\n"
                    
                    curation_prompt = RECOMMENDATION_PROMPT.format(
                        sermon_summary=analysis_result['설교요약'],
                        sermon_tags=f"주제: {analysis_result['핵심주제']}, 감정: {analysis_result['감정선']}",
                        candidates=candidates_text
                    )
                    recommendation_result = get_gemini_json(curation_prompt)
                    st.sidebar.write(f"🔍 AI 큐레이션: {'✅' if recommendation_result else '❌'}")
                status.update(label="✅ 예화 추천 완료!", state="complete")
            
            time.sleep(1)  # API Rate Limit 방지

            # 3. 피드백 & GBS 생성
            with st.status(f"✍️ {target_dept} 맞춤형 교재와 피드백을 작성 중입니다...") as status:
                feedback_result = get_gemini_json(FEEDBACK_PROMPT.format(draft=sermon_draft))
                
                time.sleep(1)  # API Rate Limit 방지
                
                if target_dept == "청년부":
                    age_range = "20~30대 대학생/직장인"
                    dept_characteristics = "권위적인 가르침보다 진정성 있는 나눔 선호, 구체적 삶의 적용 원함"
                    tone_manner = "친근하고 위트 있으면서도 핵심을 찌르는 어조 (MZ/Alpha 감성)"
                elif target_dept == "장년부":
                    age_range = "40~60대 성인"
                    dept_characteristics = "삶의 연륜이 있으며 가정과 직장의 무게를 견디는 세대, 깊은 위로와 통찰 필요"
                    tone_manner = "정중하고 깊이 있으며 목회적 돌봄이 느껴지는 어조"
                elif target_dept == "중고등부":
                    age_range = "10대 청소년"
                    dept_characteristics = "학업 스트레스와 정체성 고민, 짧고 임팩트 있는 메시지 선호"
                    tone_manner = "에너지 넘치고 짧고 간결한 어조"
                else:
                    age_range = "초등학생"
                    dept_characteristics = "활동적이고 쉬운 언어 필요, 스토리텔링 중요"
                    tone_manner = "다정하고 쉬운 선생님 말투 (존댓말 사용)"

                gbs_prompt = GBS_PROMPT_TEMPLATE.format(
                    target_dept=target_dept,
                    age_range=age_range,
                    dept_characteristics=dept_characteristics,
                    tone_manner=tone_manner,
                    draft=sermon_draft
                )
                
                gbs_content = get_gemini_response(gbs_prompt)
                status.update(label="✅ 모든 작업 완료!", state="complete")

            # === 결과 탭 ===
            tab1, tab2, tab3 = st.tabs(["🤖 AI 추천 예화", "✍️ GBS 교재 (복사용)", "👨‍🏫 설교 클리닉"])
            
            with tab1:
                st.info(f"**💡 설교 요약:** {analysis_result.get('설교요약', '')}")
                st.caption("💎 **Tip:** 노션 페이지에 들어가면 이 예화와 비슷한 예화 5개를 추천해 줍니다.")
                
                # AI 추천 결과가 있으면 사용, 없으면 top_candidates 직접 표시
                if recommendation_result and recommendation_result.get('추천목록'):
                    display_list = recommendation_result['추천목록']
                    use_ai_recommendation = True
                elif top_candidates:
                    # AI 추천 실패시 fallback: top_candidates 직접 표시
                    st.warning("⚠️ AI 큐레이션 실패, 유사도 기반 예화를 표시합니다.")
                    display_list = [{"번호": i+1, "제목": c['title'], "추천이유": f"설교 내용과 유사도 {c.get('similarity', 0):.1%}", "활용팁": "본문을 확인하고 적절히 활용하세요."} for i, c in enumerate(top_candidates[:15])]
                    use_ai_recommendation = False
                else:
                    display_list = []
                    use_ai_recommendation = False
                
                if display_list:
                    for idx, rec in enumerate(display_list):
                        original_data = None
                        if '번호' in rec and isinstance(rec['번호'], int):
                            try:
                                candidate_index = rec['번호'] - 1
                                if 0 <= candidate_index < len(top_candidates):
                                    original_data = top_candidates[candidate_index]
                            except:
                                pass
                        if not original_data:
                            original_data = next((item for item in top_candidates if item["title"] == rec["제목"]), None)

                        with st.container():
                            preacher_badge = f" `{original_data.get('preacher', '')}`" if original_data and original_data.get('preacher') else ""
                            st.markdown(f"#### 📌 {rec['제목']}{preacher_badge}")
                            st.write(f"**🗣️ 추천 이유:** {rec['추천이유']}")
                            st.caption(f"**💡 활용 팁:** {rec['활용팁']}")
                            if original_data:
                                if 'similarity' in original_data:
                                    st.caption(f"📊 의미 유사도: {original_data['similarity']:.2%}")
                                
                                with st.expander("📖 예화 본문 & 영상 보기", expanded=(idx == 0)):
                                    with st.spinner("본문을 불러오는 중..."):
                                        if original_data.get('source_url'):
                                            start_time = extract_start_time(original_data['source_url'])
                                            st.markdown(f"**📺 관련 설교 영상 (시작 시간: {start_time}초)**")
                                            st.video(original_data['source_url'], start_time=start_time)
                                            
                                            st.info("""
                                            💡 **예화와 영상 시간이 맞지 않는 경우에는**
                                            1. 영상 링크를 누른 후 
                                            2. 조회 수가 적혀 있는 박스 아래 쪽 **"더보기"** 클릭
                                            3. 박스 밑에 파란 글씨로 **"스크립트 표시"**를 클릭
                                            4. `ctrl+F` 를 눌러 예화 내용의 단어들을 검색하면 예화를 들을 수 있습니다. 
                                            """)
                                            
                                        st.divider()
                                        content_text = fetch_page_content(original_data['id'])
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
                    for point in feedback_result.get('논리점검', []):
                        st.markdown(f"- {point}")
                    st.markdown("#### 🏃‍♂️ 구체적 행동 제안 (Action Plan)")
                    for point in feedback_result.get('보완제안', []):
                        st.markdown(f"- {point}")


if __name__ == "__main__":
    main()
