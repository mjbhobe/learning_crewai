"""
01_email_age.py - a CrewAI powered email agent that converts a rough email to a polished version

@author: Manish Bhobe
My experiments with Python, AI, Gen AI and Agentic AI
Code shared for learning purposes only. Use at your own risk.
"""

from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM

load_dotenv(find_dotenv(), override=True)

llm = LLM(
    # model="gemini/gemini-3.1-flash-lite",
    model="openai/gpt-4.1-mini",
    temperature=0.1,
)
# print(llm.call("Who won the English Premier League in 2022?"))

agent = Agent(
    role="Email Assistant",
    goal="Improve emails and make them sound professional and clear.",
    backstory="A highly expert communication specialist skilled in professional email writing",
    llm=llm,
    verbose=True,
)

task = Task(
    description="""Take the following email and rewrite it into a professional and polished version.
    Expand abbreviations, remove all expletives and profanities.

    Email:
    {original_email}
    """,
    agent=agent,
    expected_output="A professional email with proper formatting and content",
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

# here is the raw email
raw_email = """
Hey team, just wanted to inform you that the demo is kind of ready, bu there is still stuff left.
Maybe we can show the client what we have and say rest is WIP.
Let me know what u think. Thanks.
"""

# really nasty complaint email from a really frustrated customer. Contains expletives/profanity and street lingo, which should all be polished by our agent.
complaint_email = """
Subject: This fkin CleenoBot is straight garbage, run me my money RN 🤬

Yo, I am absolutely losing my st right now. Whoever designed this piece of crap CleenoBot needs to get smacked upside the head because yall are out here straight up robbing people. I paid mad money for this supposed "high-tech" vacuum, and it is easily the biggest fkin L of the century.

First off, your "smart navigation" is total blshit. This brain-dead hockey puck has zero IQ, istg. It just moves mad aggressive around my crib, slamming into the walls like it’s blind, chipping my godfkin paint, and getting stuck on literally nothing. I gotta babysit this stupid a machine 24/7. It don't do st but do donuts in the kitchen until the battery dies. RN my floors look way crustier than before I turned it on.

But today? Today was the final fkin straw. My dog had a little accident by the rug. Tell me why this useless piece of st didn't detect it at all?? Your trash-a machine ran straight through the turd and dragged it across the whole living room. No cap, it painted the entire apartment in dog st. The vibes are ruined, the smell is fkin egregious, and I spent my whole night scrubbing the floor on my hands and knees while this robotic bch just sat on its dock charging. I am literally sick to my stomach.

Yall got me messed up if you think I’m keeping this fkin brick. I need my godfkin refund ASAP, no cap, no delayed responses, no automated bot replies. Check the serial number attached and run me my money immediately. I’m dead serious, fix this st rn or I’m putting yall on blast all over TikTok, exposing this scam, and calling the BBB.

Unfkinbelievable smh,

[Your Name]

Order #847291

Txt me at: [Phone Number]
"""

# and run the crew
response = crew.kickoff({"original_email": complaint_email})
print(f"Final response:\n{response}")
