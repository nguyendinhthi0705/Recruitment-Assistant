import os
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.agents import load_tools,initialize_agent, Tool, load_tools
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from dotenv import load_dotenv
load_dotenv()

def init_llm():
        
    model_parameter = {"temperature": 0, "top_p": 0.5, "max_tokens_to_sample": 2000}
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
        model_id="anthropic.claude-v2", 
        model_kwargs=model_parameter,
        ) 
    
    return llm

def get_llm(streaming_callback):
        
    model_parameter = {"temperature": 0, "top_p": 0.5, "max_tokens_to_sample": 2000}
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
        model_id="anthropic.claude-v2", 
        model_kwargs=model_parameter,
        streaming=True,
        callbacks=[streaming_callback]
        ) 
    
    return llm

def get_rag_chat_response(input_text, streaming_callback): 
    llm = get_llm(streaming_callback)
    return llm.invoke(input=input_text)


def summary_resume_stream(input_text, streaming_callback): 
    llm = get_llm(streaming_callback)
    prompt = """You are the best recruitment consultant expert, you will scan the resume and output concide content for the following informantion: 
        Contact, Experience, Skills, and suggested jobs based on the resume in a bulleted list.
        Output all content same as input or in Vietnamese with the following format\n\n
        <format>
         Contact:
         Experience:
         Skills:
         Suggested jobs:
        </format>
        \n\nHuman: here is the resume content
        <text>""" + str(input_text) + """</text>
    \n\nAssistant: """

    return llm.invoke(prompt)


def suggested_jobs(input_text): 
    llm = init_llm()
    prompt = """You are the best recruitment consultant expert, you will scan the resume and output concide content for the following informantion: 
            suggested jobs based on the resume.
        Output all content same as input or in Vietnamese with the following format\n\n
        <format>
         Suggested jobs:
        </format>
        \n\nHuman: here is the resume content
        <text>""" + input_text + """</text>
    \n\nAssistant: """
    print(input_text)
    return llm.invoke(prompt)


def get_jobs(jobs):
    tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())  
    res =  tool.run(jobs)
    print(res)
    return res


tools=[
    Tool(
        name="get suggested jobs",
        func=suggested_jobs,
        description="use this tool get suggested jobs for resume"
    ),
    Tool(
        name="search jobs",
        func=get_jobs,
        description="Use this to search jobs"
    ),
]

def initializeAgent():
    agent=initialize_agent(
    llm=init_llm(),
    agent="zero-shot-react-description",
    tools=tools,
    verbose=True,
    max_iteration=2,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    output_key="output",
)
    prompt="""\n\nHuman: You are a jobs advisor. Give jobs recommendations for given resume based on following instructions. 
<instructions>
Answer the following questions as best you can. You have access to the following tools:
get suggested jobs: use this tool get suggested jobs for a input resume.
search jobs: Use this to search jobs.
</instructions>

<steps>
    Note- if you fail in satisfying any of the step below, Just move to next one
    1) Use "get suggested jobs" tool get suggested jobs for a input resume. Output- suggested jobs
    2) Use "search jobs" tool to search jobs.. Output- List of Jobs
</steps>

Use the following format:
Question: the input resume you must scan to suggest jobs
Thought: you should always think about what to do, Also try to follow All steps mentioned above
Action: the action to take, should be one of [get suggested jobs, search jobs]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question
 {input}

\n\nAssistant:
{agent_scratchpad}

"""
    agent.agent.llm_chain.prompt.template=prompt
    return agent
    