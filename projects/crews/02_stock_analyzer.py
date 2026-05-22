"""
02_stock_analyzer.py - a 2 agent crew that researches a
   company's (e.g. Apple, Reliance) and creates a report.

@author: Manish Bhobe
My experiments with Python, AI, Gen AI and Agentic AI
Code shared for learning purposes only. Use at your own risk.
"""

from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown

from crewai import Agent, Task, Crew
from crewai.llm import LLM

load_dotenv(find_dotenv(), override=True)
console = Console()

llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    # model="openai/gpt-4.1-mini",
    temperature=0.1,
)
# print(llm.call("Who won the English Premier League in 2022?"))

researcher = Agent(
    role="Stock Researcher",
    goal="Research a {company}'s stock performance over the past {num_years} years from {today}.",
    backstory="An expert in financial analysis",
    llm=llm,
    verbose=True,
)

research_task = Task(
    description="Find out {company}’s stock trends for the past {num_years} years from {today}.",
    expected_output="bullet-point list of key financial movements",
    agent=researcher,
)

summarizer = Agent(
    role="Summary Writer",
    goal="Summarize stock trends and deliver a report",
    backstory="An AI trained in report generation",
)

summarization_task = Task(
    description="Write a clear summary of the findings from the researcher.",
    expected_output="A short report in markdown format with recommendations. Add a disclaimer at the end.",
    agent=summarizer,
)


crew = Crew(
    agents=[researcher, summarizer],
    tasks=[research_task, summarization_task],
    verbose=True,
)


# and run the crew
response = crew.kickoff(
    {
        "company": "Bharat Forge",
        "num_years": 2,
        "today": datetime.now().strftime("%d-%b-%Y"),
    }
)
console.print(f"[blue]Final Report:[/blue]\n\n")
console.print(Markdown(response))
