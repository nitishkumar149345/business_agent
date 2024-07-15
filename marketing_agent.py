from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('.env')
from langchain_core.output_parsers import JsonOutputParser
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

class Queries(BaseModel):
    questions : list[str] = Field(description='A list of detailed and relevant questions that can help gather comprehensive information about a business model')

class BusinessModel(BaseModel):
    business_model : str = Field(description='Best business model on the provided domain')
    description : str = Field(description='description about the business model')


class MarketingAgent():
    def __init__(self, domain) :
        self.domain = domain
        self.observations = 'Marketing Report: '

    def find_business_model(self):
        parser = JsonOutputParser(pydantic_object=BusinessModel)
        prompt = ChatPromptTemplate.from_messages([
            ('system','''You are an expert in business strategies and models.
                Your task is to suggest the best business model for a given domain and provide a detailed description of this model.
                \nformat_instructions:{format_instructions}'''),
            ('user','{domain}')
        ])

    
        chain = prompt | llm | parser
        response = chain.invoke({'domain':self.domain, 'format_instructions':parser.get_format_instructions()})
        self.business_model = response['business_model']
        self.description = response['description']
        

    def generate_questions(self):

        parser = JsonOutputParser(pydantic_object=Queries)

        prompt = ChatPromptTemplate.from_messages([
            ('system','''
            
                Generate a list of detailed and relevant questions that can help gather comprehensive information about a business model.
                The questions should cover various aspects such as the Current Market Trends, Competitor Analysis,Revenue Streams,Target Customer Segments,
                Value Propositions, Potential Risks and Opportunities and other business model related.
                You will recieve the business model, then you have to generate the questions as mentioned.
                \nformat_instructions: {format_instructions}
                '''),
            ('user','{domain}')
        ])

        chain = prompt | llm | parser
        result = chain.invoke({'domain':self.business_model, 'format_instructions':parser.get_format_instructions()})
        self.questions = result['questions']

    def agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system',''' Answer the questions using your knowlodge and provided tools: The tool you have is TavilySearchResults for online searching.
                    first try to answer with your knowledge, if you dit't get, then try the provided tool.
                    Please ensure that the response is long, descriptive, comprehensive and informative, rather than just listing points or being overly concise.
                    Ensure that the answers you provide are thorough, well-informed, and cover all relevant aspects.
                    The responses should be detailed, insightful, and easy to understand, offering a complete and clear explanation on the topic'''),
                    
            ('user','{question}'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        tools = [TavilySearchResults()]
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        self.observations = 'business mdoel---'+self.business_model
        self.observations= self.observations +'\n'+ f'description: {self.description}'

        for question in self.questions:
            response = agent_executor.invoke({'question':question})
            self.observations+=response['output']

        
    
    def __call__(self):
        self.find_business_model()
        self.generate_questions()
        self.agent()
        print ('---------------formatting report-----------------')
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                    - Your task is to transform the provided input into a detailed and well-structured Marketing Report.
                    - Your role is only to format the output properly. Do not add, remove, or compress any information from the input.
                    - The report should be organized with clear titles and sections, covering all relevant aspects of the input.
                    - Ensure that the report is comprehensive, informative, and professionally presented.
                '''),
            ('user','{input}')
        ])

        chain = prompt | llm
        response = chain.invoke({'input':self.observations})
        return {'business_model':self.business_model, 'marketing_report':response.content}


        

# agent = MarketingAgent('large language models with langchain and langgraph')
# re = agent()
# print (re)