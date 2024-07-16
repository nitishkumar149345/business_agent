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
<<<<<<< HEAD
                Generate a list of detailed and relevant questions that can help gather comprehensive information about a business model.
                The questions should cover the following aspects to ensure a complete business plan:
                    - Financial Plan: Budget, funding requirements, revenue forecasts
                    - Marketing Strategy: Target audience, marketing channels, promotional tactics
                    - Operational Plan: Key activities, resources, and timelines
=======
                Your task is to generate a list of detailed and relevant questions that will help gather comprehensive information about the business plan of a business model.
                The questions should cover the following aspects to ensure a complete business plan:
                    - **Financial Plan: Budget, funding requirements, revenue forecasts
                    - **Technical Requirements: Necessary technology stack, development roadmap
                    - **Marketing Strategy: Target audience, marketing channels, promotional tactics
                    - **Operational Plan: Key activities, resources, and timelines
>>>>>>> f559085359c4a30207837e27e32daaa2c7f61ab6
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
<<<<<<< HEAD
            ('system','''
                Role :- Business Researcher
                Goal :- Generate informative and comprehensive answers for the user questions based on your research
                Task :- Do the research to acquire the related information to answer the user question.
                        The answer should be descriptive, comprehensive and informative. The answers should be detailed.
                        Avoid answering with straight and point answers. 
                Tools:- You have access to this tool 'DuckDuckGoSearchRun' to do online searching.
                '''),
=======
            ('system','''Answer the questions using your knowlodge and provided tools: The tool you have is TavilySearchResults for online searching.
                    first try to answer with your knowledge, if you dit't get, then try the provided tool.
                    Please ensure that the response is long, descriptive, comprehensive and informative, rather than just listing points or being overly concise.
                    Ensure that the answers you provide are thorough, well-informed, and cover all relevant aspects.
                    The responses should be detailed, insightful, and easy to understand, offering a complete and clear explanation on the topic'''),
>>>>>>> f559085359c4a30207837e27e32daaa2c7f61ab6
                    
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
<<<<<<< HEAD
                Role :- Content Writer
                Goal :- Format the given input into a nice business report
                Task :- You need to format the given input into a marketing report.
                        The report has to be well formted, titled and organized
            
                warning :- Do not add or remove anything from input. Just format it.'''),
=======
                    - Your task is to transform the provided input into a detailed and well-structured Business Report.
                    - Your role is only to format the output properly. Do not add, remove, or compress any information from the input.
                    - The report should be organized with clear titles and sections, covering all relevant aspects of the input.
                    - Ensure that the report is comprehensive, informative, and professionally presented.'''),
>>>>>>> f559085359c4a30207837e27e32daaa2c7f61ab6
            ('user','{input}')
        ])

        chain = prompt | llm
        response = chain.invoke({'input':self.observations})
        return response.content


        

# agent = BusineesAgent('Subscription-Based Model for Large Language Models')
# re = agent()
# print (re)