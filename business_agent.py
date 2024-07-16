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



class BusineesAgent():
    def __init__(self, business_model) :
        self.business_model = business_model
        self.observations = 'Business Report: '


    def generate_questions(self):

        parser = JsonOutputParser(pydantic_object=Queries)

        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                Generate a list of detailed and relevant questions that can help gather comprehensive information about a business model.
                The questions should cover the following aspects to ensure a complete business plan:
                    - Financial Plan: Budget, funding requirements, revenue forecasts
                    - Marketing Strategy: Target audience, marketing channels, promotional tactics
                    - Operational Plan: Key activities, resources, and timelines
                You will receive the business model, and based on that, you need to generate the questions as mentioned above.
                \nformat_instructions: {format_instructions}
                '''),
            ('user','{model}')
        ])

        chain = prompt | llm | parser
        result = chain.invoke({'model':self.business_model, 'format_instructions':parser.get_format_instructions()})
        self.questions = result['questions']

    def agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                Role :- Business Researcher
                Goal :- Generate informative and comprehensive answers for the user questions based on your research
                Task :- Do the research to acquire the related information to answer the user question.
                        The answer should be descriptive, comprehensive and informative. The answers should be detailed.
                        Avoid answering with straight and point answers. 
                Tools:- You have access to this tool 'DuckDuckGoSearchRun' to do online searching.
                '''),
                    
            ('user','{question}'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        tools = [TavilySearchResults()]
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
  

        for question in self.questions:
            response = agent_executor.invoke({'question':question})
            self.observations+=response['output']

        
    
    def __call__(self):
        self.generate_questions()
        self.agent()
        print ('---------------formatting report-----------------')
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                Role :- Content Writer
                Goal :- Format the given input into a nice business report
                Task :- You need to format the given input into a marketing report.
                        The report has to be well formted, titled and organized
            
                warning :- Do not add or remove anything from input. Just format it.'''),
            ('user','{input}')
        ])

        chain = prompt | llm
        response = chain.invoke({'input':self.observations})
        return response.content


        

# agent = BusineesAgent('Subscription-Based Model for Large Language Models')
# re = agent()
# print (re)