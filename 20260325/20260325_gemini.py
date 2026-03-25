import os
import io
import json
import requests
import asyncio
import pandas as pd
from typing import TypedDict
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, START, END

# Selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

# MCP
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ==========================================
# 1. 도구(Tools) 정의
# ==========================================
@tool
def finance_report(company_code: str) -> str:
    """종목 코드(예: 005930)를 받아 연간/분기 재무제표를 스크래핑하여 반환"""
    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    
    driver.get(f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={company_code}")
    report = BeautifulSoup(driver.page_source, "html.parser").find_all('table', class_='gHead01 all-width')
    
    # HTML 테이블을 Pandas로 읽어 JSON 형태로 변환
    report_text = "\n".join([pd.read_html(io.StringIO(str(x)))[0].to_json(force_ascii=False) for x in report])
    driver.quit()
    return report_text

@tool
def get_news(company_name: str) -> str:
    """네이버 API를 통해 기업 뉴스를 검색하고 본문을 크롤링하여 반환"""
    # 주의: 실제 운영 시 이 키들은 .env 파일로 빼서 os.getenv()로 관리하세요.
    client_id = os.getenv("NAVER_CLIENT_ID", "당신의_아이디") 
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "당신의_시크릿")
    
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {'query': company_name, 'display': 10, 'start': 1, "sort": "date"}
    
    response = requests.get(url, headers=headers, params=params)
    links = [item['link'] for item in response.json().get('items', []) if 'naver.com' in item['link']]
    
    if not links:
        return "관련 네이버 뉴스를 찾을 수 없습니다."

    bs4_kwargs = {'parse_only': SoupStrainer("div", id="newsct_article")}
    news_content = [WebBaseLoader(link, bs_kwargs=bs4_kwargs).load()[0].page_content.strip() for link in links[:3]]
    return " ".join(news_content)

@tool
def get_data(company_code: str, sdate: str, edate: str) -> str:
    """시작일~종료일 사이의 주가 데이터(시가, 고가, 저가, 종가 등) 반환"""
    stock_url = f"https://m.stock.naver.com/front-api/external/chart/domestic/info?symbol={company_code}&requestType=1&startTime={sdate}&endTime={edate}&timeframe=day"
    data = eval(requests.get(stock_url).text.strip())
    return json.dumps(data, ensure_ascii=False)

@tool
def get_code(company_name: str) -> str:
    """회사 이름으로 종목 코드를 찾아 반환 (간략화된 버전)"""
    df = pd.read_csv("./data_2058_20260323.csv", encoding='cp949')
    matched = df[df['한글 종목명'].str.contains(company_name, na=False)]
    return matched.to_json(force_ascii=False)

# ==========================================
# 2. LangGraph 노드 및 상태 정의
# ==========================================
llm = ChatOpenAI(model='gpt-4o', temperature=0.2)

class CompanyState(TypedDict):
    question: str
    company_finance: str
    company_news: str
    company_stock: str
    final_report: str

async def finance_node(state: CompanyState):
    """재무제표 분석 노드"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 금융 감독위원회의 재무 분석가입니다. 연간/분기별 재무 건전성, 수익성을 분석하세요."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, [finance_report, get_code], prompt)
    executor = AgentExecutor(agent=agent, tools=[finance_report, get_code])
    result = await executor.ainvoke({"input": state['question']})
    return {'company_finance': result['output']}

async def news_node(state: CompanyState):
    """뉴스 기반 미래 전망 분석 노드"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 기업 분석가입니다. 뉴스를 바탕으로 기업의 현재 상태와 미래 전망을 분석하세요."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, [get_news], prompt)
    executor = AgentExecutor(agent=agent, tools=[get_news])
    result = await executor.ainvoke({"input": state['question']})
    return {'company_news': result['output']}

async def stock_node(state: CompanyState):
    """주가 차트 분석 노드"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 20년 경력의 전업 투자자입니다. 제공된 주가 데이터를 통해 기술적 분석을 수행하세요."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, [get_data, get_code], prompt)
    executor = AgentExecutor(agent=agent, tools=[get_data, get_code])
    result = await executor.ainvoke({"input": state['question']})
    return {'company_stock': result['output']}

async def summarize_node(state: CompanyState):
    """최종 투자 의견 종합 노드"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        아래 자료를 종합하여 최종 매수/매도 의견과 요약 보고서를 작성하세요.
        - 재무 분석: {company_finance}
        - 뉴스 전망: {company_news}
        - 주가 차트: {company_stock}
        """),
        ("human", "{question}")
    ])
    chain = prompt | llm
    result = await chain.ainvoke({
        'company_finance': state.get('company_finance', ""),
        'company_news': state.get('company_news', ""),
        'company_stock': state.get('company_stock', ""),
        'question': state['question']
    })
    return {'final_report': result.content}

# ==========================================
# 3. LangGraph 빌드
# ==========================================
workflow = StateGraph(CompanyState)
workflow.add_node('finance_node', finance_node)
workflow.add_node('news_node', news_node)
workflow.add_node('stock_node', stock_node)
workflow.add_node('summarize_node', summarize_node)

# 병렬 실행 후 종합 노드로 모임
workflow.add_edge(START, 'finance_node')
workflow.add_edge(START, 'news_node')
workflow.add_edge(START, 'stock_node')
workflow.add_edge('finance_node', 'summarize_node')
workflow.add_edge('news_node', 'summarize_node')
workflow.add_edge('stock_node', 'summarize_node')
workflow.add_edge('summarize_node', END)

app = workflow.compile()

# ==========================================
# 4. MCP 서버 설정 및 실행
# ==========================================
mcp = FastMCP("Stock_Analysis_Graph")

@mcp.tool()
async def stock_analysis(question: str) -> str:
    """
    사용자가 주식 종목을 질문하면 랭그래프를 태워 종합 분석을 반환합니다.
    예: "2026년 3월 25일 기준으로 코리아써키트에 대해 분석해줘"
    """
    try:
        result = await app.ainvoke({'question': question})
        return result['final_report']
    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}"

if __name__ == "__main__":
    mcp.run()