import os
import httpx
import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from google.adk.agents import Agent

# 환경 변수 로드 (API 키 등)
load_dotenv()

# ==========================================
# 1. 도구(Tools) 정의
# ==========================================
def greet_user() -> str:
    """사용자에게 가볍게 인사하는 기본 도구"""
    return "안녕! 나는 너의 AI 어시스턴트야."

def get_weather(city_name: str) -> dict:
    """도시 이름을 받아 위도/경도를 찾고 현재 날씨를 반환"""
    geo = Nominatim(user_agent='weather_app')
    location = geo.geocode(city_name)
    
    if not location:
        raise ValueError(f'{city_name} 위치의 데이터를 찾을 수 없습니다.')
        
    lat, lon = location.latitude, location.longitude
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()

def get_code(company_name: str) -> dict:
    """로컬 CSV 파일을 읽어 회사 이름에 해당하는 종목 코드를 반환"""
    # 주의: 본인의 실제 CSV 경로로 수정하세요
    csv_path = "./data_2058_20260323.csv" 
    df = pd.read_csv(csv_path, encoding='cp949')
    matched_df = df[df['한글 종목명'].apply(lambda x: x.find(company_name) > -1)]
    return matched_df.to_json(force_ascii=False)

def get_company_info(company_code: str) -> dict:
    """토스 증권 API를 활용해 종목 코드의 기업 개요 정보를 반환"""
    url = f"https://wts-info-api.tossinvest.com/api/v2/stock-infos/A{company_code}/overview"
    response = httpx.get(url).json()
    return response['result']['company']

# ==========================================
# 2. 에이전트(Agent) 생성
# ==========================================
root_agent = Agent(
    name='skn_agent',
    model="gemini-2.5-flash",
    description='회사 정보 수집 및 날씨 확인이 가능한 만능 에이전트',
    instruction='사용자 요청을 바탕으로 회사의 정보나 날씨를 친절하게 알려주세요.',
    tools=[greet_user, get_weather, get_code, get_company_info]
)

# 실행 예시 (직접 실행해볼 때 주석 해제)
# if __name__ == "__main__":
#     response = root_agent.run("삼성전자의 종목 코드와 기업 정보를 알려주고, 분당 날씨도 알려줘.")
#     print(response)