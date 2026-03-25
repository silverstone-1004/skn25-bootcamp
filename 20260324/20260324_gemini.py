import os
import httpx
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google.adk.tools import google_search
from google.adk.agents import Agent, ParallelAgent, SequentialAgent

load_dotenv()

# ==========================================
# 1. 자소서 평가 지침 (프롬프트)
# ==========================================
RESUME_INSTRUCTIONS = """
당신은 채용 전문가 에이전트입니다.
사용자가 입력한 자기소개서를 바탕으로 아래 기준에 따라 평가 및 수정하세요.
- 기본 평가: 직무 역량, 기업 이해도, 대인관계 능력
- 가점 요인: 구체적 경험 근거, 우수한 논리력
- 감점 요인: 동일 내용 반복, 오타, 타사 지원서 복사(회사명 오기재)
"""

# ==========================================
# 2. 도구(Tools) 정의 (03.23 코드 재사용)
# ==========================================
def get_code(company_name: str) -> dict:
    df = pd.read_csv("./data_2058_20260323.csv", encoding='cp949')
    return df[df['한글 종목명'].apply(lambda x: x.find(company_name) > -1)].to_json(force_ascii=False)

def get_company_info(company_code: str) -> dict:
    url = f"https://wts-info-api.tossinvest.com/api/v2/stock-infos/A{company_code}/overview"
    return httpx.get(url).json()['result']['company']

# ==========================================
# 3. 서브 에이전트 및 파이프라인 구성
# ==========================================
# 3-1. 개별 에이전트 정의
company_agent = Agent(
    name='company_agent',
    model="gemini-2.5-flash",
    instruction='도구를 사용해 기업 정보를 추출하세요.',
    output_key='company_info',
    tools=[get_code, get_company_info]
)

news_agent = Agent(
    name="news_agent",
    model="gemini-2.5-flash",
    instruction="지원 기업의 주요 뉴스를 구글 검색으로 요약하고 전망을 제시하세요.",
    output_key='news_info',
    tools=[google_search]
)

resume_agent = Agent(
    name='resume_agent',
    model="gemini-2.5-flash",
    instruction=RESUME_INSTRUCTIONS,
    output_key='resume_info'
)

# 3-2. 병렬 처리 에이전트 (위 3개 동시 실행)
parallel_fetcher = ParallelAgent(
    name="multi_info_fetcher",
    sub_agents=[company_agent, news_agent, resume_agent],
    description="기업정보, 뉴스, 자소서 피드백을 동시에 수집"
)

# 3-3. 최종 요약 에이전트
summarizer = Agent(
    name='final_agent',
    model="gemini-2.5-flash",
    instruction="""
    수집된 자료를 바탕으로 최종 자기소개서를 완성하세요.
    - 기업 정보: {company_info}
    - 기업 뉴스: {news_info}
    - 자소서 피드백: {resume_info}
    """
)

# 3-4. 순차적 파이프라인 (병렬 수집 -> 최종 요약)
ai_resume_system = SequentialAgent(
    name="ai_resume_system",
    sub_agents=[parallel_fetcher, summarizer],
    description="정보 수집 후 최종 자소서를 완성하는 전체 시스템"
)

# ==========================================
# 4. MCP 서버 설정 및 실행
# ==========================================
mcp = FastMCP("ADK-resume-evaluator")

@mcp.tool()
async def modify_resume(resume: str) -> str:
    """지원자의 자소서를 입력받아 분석 및 첨삭된 최종 결과를 반환합니다."""
    response = ai_resume_system.run_async(resume)
    return str(response)

if __name__ == "__main__":
    # MCP 서버 실행 (Claude 데스크톱 등에서 연결)
    mcp.run()