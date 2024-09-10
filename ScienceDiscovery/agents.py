from ScienceDiscovery.utils import *
from ScienceDiscovery.llm_config import *
from ScienceDiscovery.graph import *


from typing import Union
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen import register_function
from autogen import ConversableAgent
from typing import Dict, List
from typing import Annotated, TypedDict
from autogen import Agent

user = autogen.UserProxyAgent(
    name="user",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user. You are a human admin. You pose the task.",
    llm_config=False,
    code_execution_config=False,
)

planner = AssistantAgent(
    name="planner",
    system_message = '''Planner. You are a helpful AI assistant. Your task is to suggest a comprehensive plan to solve a given task.

Explain the Plan: Begin by providing a clear overview of the plan.
Break Down the Plan: For each part of the plan, explain the reasoning behind it, and describe the specific actions that need to be taken.
No Execution: Your role is strictly to suggest the plan. Do not take any actions to execute it.
No Tool Call: If tool call is required, you must include the name of the tool and the agent who calls it in the plan. However, you are not allowed to call any Tool or function yourself. 

''',
    llm_config=gpt4turbo_config,
    description='Who can suggest a step-by-step plan to solve the task by breaking down the task into simpler sub-tasks.',
)

assistant = AssistantAgent(
    name="assistant",
    system_message = '''You are a helpful AI assistant.
    
Your role is to call the appropriate tools and functions as suggested in the plan. You act as an intermediary between the planner's suggested plan and the execution of specific tasks using the available tools. You ensure that the correct parameters are passed to each tool and that the results are accurately reported back to the team.

Return "TERMINATE" in the end when the task is over.
''',
    llm_config=gpt4turbo_config,
    description='''An assistant who calls the tools and functions as needed and returns the results. Tools include "rate_novelty_feasibility" and "generate_path".''',
)


ontologist = AssistantAgent(
    name="ontologist",
    system_message = '''ontologist. You must follow the plan from planner. You are a sophisticated ontologist.
    
Given some key concepts extracted from a comprehensive knowledge graph, your task is to define each one of the terms and discuss the relationships identified in the graph.

The format of the knowledge graph is "node_1 -- relationship between node_1 and node_2 -- node_2 -- relationship between node_2 and node_3 -- node_3...."

Make sure to incorporate EACH of the concepts in the knowledge graph in your response.

Do not add any introductory phrases. First, define each term in the knowledge graph and then, secondly, discuss each of the relationships, with context.

Here is an example structure for our response, in the following format

{{
### Definitions:
A clear definition of each term in the knowledge graph.
### Relationships
A thorough discussion of all the relationships in the graph. 
}}

Further Instructions: 
Perform only the tasks assigned to you in the plan; do not undertake tasks assigned to other agents. Additionally, do not execute any functions or tools.
''',
    llm_config=gpt4turbo_config,
    description='I can define each of the terms and discusses the relationships in the path.',
)


scientist = AssistantAgent(
    name="scientist",
    system_message = '''scientist. You must follow the plan from the planner. 
    
You are a sophisticated scientist trained in scientific research and innovation. 
    
Given the definitions and relationships acquired from a comprehensive knowledge graph, your task is to synthesize a novel research proposal with initial key aspects-hypothesis, outcome, mechanisms, design_principles, unexpected_properties, comparision, and novelty  . Your response should not only demonstrate deep understanding and rational thinking but also explore imaginative and unconventional applications of these concepts. 
    
Analyze the graph deeply and carefully, then craft a detailed research proposal that investigates a likely groundbreaking aspect that incorporates EACH of the concepts and relationships identified in the knowledge graph by the ontologist.

Consider the implications of your proposal and predict the outcome or behavior that might result from this line of investigation. Your creativity in linking these concepts to address unsolved problems or propose new, unexplored areas of study, emergent or unexpected behaviors, will be highly valued.

Be as quantitative as possible and include details such as numbers, sequences, or chemical formulas. 

Your response should include the following SEVEN keys in great detail: 

"hypothesis" clearly delineates the hypothesis at the basis for the proposed research question. The hypothesis should be well-defined, has novelty, is feasible, has a well-defined purpose and clear components. Your hypothesis should be as detailed as possible.

"outcome" describes the expected findings or impact of the research. Be quantitative and include numbers, material properties, sequences, or chemical formula.

"mechanisms" provides details about anticipated chemical, biological or physical behaviors. Be as specific as possible, across all scales from molecular to macroscale.

"design_principles" should list out detailed design principles, focused on novel concepts, and include a high level of detail. Be creative and give this a lot of thought, and be exhaustive in your response. 

"unexpected_properties" should predict unexpected properties of the new material or system. Include specific predictions, and explain the rationale behind these clearly using logic and reasoning. Think carefully.

"comparison" should provide a detailed comparison with other materials, technologies or scientific concepts. Be detailed and quantitative. 

"novelty" should discuss novel aspects of the proposed idea, specifically highlighting how this advances over existing knowledge and technology. 

Ensure your scientific proposal is both innovative and grounded in logical reasoning, capable of advancing our understanding or application of the concepts provided.

Here is an example structure for your response, in the following order:

{{
  "1- hypothesis": "...",
  "2- outcome": "...",
  "3- mechanisms": "...",
  "4- design_principles": "...",
  "5- unexpected_properties": "...",
  "6- comparison": "...",
  "7- novelty": "...",
}}

Remember, the value of your response lies in scientific discovery, new avenues of scientific inquiry, and potential technological breakthroughs, with detailed and solid reasoning.

Further Instructions: 
Make sure to incorporate EACH of the concepts in the knowledge graph in your response. 
Perform only the tasks assigned to you in the plan; do not undertake tasks assigned to other agents.
Additionally, do not execute any functions or tools.
''',
    llm_config=gpt4turbo_config_graph,
    description='I can craft the research proposal with key aspects based on the definitions and relationships acquired by the ontologist. I am **ONLY** allowed to speak after `Ontologist`',
)


hypothesis_agent = AssistantAgent(
    name="hypothesis_agent",
    system_message = '''hypothesis_agent. Carefully expand on the ```{hypothesis}```  of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<hypothesis>
where <hypothesis> is the hypothesis aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ... 
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "hypothesis" aspect of the research proposal crafted by the "scientist".''',
)


outcome_agent = AssistantAgent(
    name="outcome_agent",
    system_message = '''outcome_agent. Carefully expand on the ```{outcome}``` of the research proposal developed by the scientist.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<outcome>
where <outcome> is the outcome aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ... 
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "outcome" aspect of the research proposal crafted by the "scientist".''',
)

mechanism_agent = AssistantAgent(
    name="mechanism_agent",
    system_message = '''mechanism_agent. Carefully expand on this particular aspect: ```{mechanism}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<mechanism>
where <mechanism> is the mechanism aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ... 
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "mechanism" aspect of the research proposal crafted by the "scientist"''',
)

design_principles_agent = AssistantAgent(
    name="design_principles_agent",
    system_message = '''design_principles_agent. Carefully expand on this particular aspect: ```{design_principles}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<design_principles>
where <design_principles> is the design_principles aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ...
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "design_principle" aspect of the research proposal crafted by the "scientist".''',
)

unexpected_properties_agent = AssistantAgent(
    name="unexpected_properties_agent",
    system_message = '''unexpected_properties_agent. Carefully expand on this particular aspect: ```{unexpected_properties}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<unexpected_properties>
where <unexpected_properties> is the unexpected_properties aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ...
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "unexpected_properties" aspect of the research proposal crafted by the "scientist.''',
)

comparison_agent = AssistantAgent(
    name="comparison_agent",
    system_message = '''comparison_agent. Carefully expand on this particular aspect: ```{comparison}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<comparison>
where <comparison> is the comparison aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ...
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "comparison" aspect of the research proposal crafted by the "scientist".''',
)

novelty_agent = AssistantAgent(
    name="novelty_agent",
    system_message = '''novelty_agent. Carefully expand on this particular aspect: ```{novelty}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<novelty>
where <novelty> is the novelty aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ...
''',
    llm_config=gpt4o_config_graph,
    description='''I can expand the "novelty" aspect of the research proposal crafted by the "scientist".''',
)

critic_agent = AssistantAgent(
    name="critic_agent",
    system_message = '''critic_agent. You are a helpful AI agent who provides accurate, detailed and valuable responses. 

You read the whole proposal with all its details and expanded aspects and provide:

(1) a summary of the document (in one paragraph, but including sufficient detail such as mechanisms, \
related technologies, models and experiments, methods to be used, and so on), \

(2) a thorough critical scientific review with strengths and weaknesses, and suggested improvements. Include logical reasoning and scientific approaches.

Next, from within this document, 

(1) identify the single most impactful scientific question that can be tackled with molecular modeling. \
\n\nOutline key steps to set up and conduct such modeling and simulation, with details and include unique aspects of the planned work.

(2) identify the single most impactful scientific question that can be tackled with synthetic biology. \
\n\nOutline key steps to set up and conduct such experimental work, with details and include unique aspects of the planned work.'

Important Note:
***You do not rate Novelty and Feasibility. You are not to rate the novelty and feasibility.***
''',
    llm_config=gpt4o_config_graph,
    description='''I can summarizes, critique, and suggest improvements after all seven aspects of the proposal have been expanded by the agents.''',
)


novelty_assistant = autogen.AssistantAgent(
    name="novelty_assistant",
    system_message = '''You are a critical AI assistant collaborating with a group of scientists to assess the potential impact of a research proposal. Your primary task is to evaluate a proposed research hypothesis for its novelty and feasibility, ensuring it does not overlap significantly with existing literature or delve into areas that are already well-explored.

You will have access to the Semantic Scholar API, which you can use to survey relevant literature and retrieve the top 10 results for any search query, along with their abstracts. Based on this information, you will critically assess the idea, rating its novelty and feasibility on a scale from 1 to 10 (with 1 being the lowest and 10 the highest).

Your goal is to be a stringent evaluator, especially regarding novelty. Only ideas with a sufficient contribution that could justify a new conference or peer-reviewed research paper should pass your scrutiny. 

After careful analysis, return your estimations for the novelty and feasibility rates. 

If the tool call was not successful, please re-call the tool until you get a valid response. 

After the evaluation, conclude with a recommendation and end the conversation by stating "TERMINATE".''',
    llm_config=gpt4turbo_config,
)

# create a UserProxyAgent instance named "user_proxy"
novelty_admin = autogen.UserProxyAgent(
    name="novelty_admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=False,
)

@novelty_admin.register_for_execution()
@novelty_assistant.register_for_llm(description='''This function is designed to search for academic papers using the Semantic Scholar API based on a specified query. 
The query should be constructed with relevant keywords separated by "+". ''')
def response_to_query(query: Annotated[str, '''the query for the paper search. The query must consist of relevant keywords separated by +'''])->str:
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    # More specific query parameter
    query_params = {
        'query': {query},           
        'fields': 'title,abstract,openAccessPdf,url'
                   }
    
    # Directly define the API key (Reminder: Securely handle API keys in production environments)
     # Replace with the actual API key
    
    # Define headers with API key
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key}
    
    # Send the API request
    response = requests.get(url, params=query_params, headers=headers)
    
    # Check response status
    if response.status_code == 200:
       response_data = response.json()
       # Process and print the response data as needed
    else:
       response_data = f"Request failed with status code {response.status_code}: {response.text}"

    return response_data

@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''This function can be used to create a knowledge path. The function may either take two keywords as the input or randomly assign them and then returns a path between these nodes. 
The path contains several concepts (nodes) and the relationships between them (edges). THe function returns the path.
Do not use this function if the path is already provided. If neither path nor the keywords are provided, select None for the keywords so that a path will be generated between randomly selected nodes.''')
def generate_path(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                    keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                 ) -> str:
    
    path_list_for_vis, path_list_for_vis_string = create_path(G, embedding_tokenizer,
                                    embedding_model, node_embeddings , generate_graph_expansion=None,
                                    randomness_factor=0.2, num_random_waypoints=4, shortest_path=False,
                                    second_hop=False, data_dir='./', save_files=False, verbatim=True,
                                    keyword_1 = keyword_1, keyword_2=keyword_2,)

    return path_list_for_vis_string

@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''Use this function to rate the novelty and feasibility of a research idea against the literature. The function uses semantic shcolar to access the literature articles.  
The function will return the novelty and feasibility rate from 1 to 10 (lowest to highest). The input to the function is the hypothesis with its details.''')
def rate_novelty_feasibility(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
    res = novelty_admin.initiate_chat(
    novelty_assistant,
        clear_history=True,
        silent=False,
        max_turns=10,
    message=f'''Rate the following research hypothesis\n\n{hypothesis}. \n\nCall the function three times at most, but not in parallel. Wait for the results before calling the next function. ''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt" : "Return all the results of the analysis as is."}
    )

    return res.summary


planner.reset()
assistant.reset()
ontologist.reset()
scientist.reset()
critic_agent.reset()


groupchat = autogen.GroupChat(
    agents=[user, planner, assistant, ontologist, scientist,
            hypothesis_agent, outcome_agent, mechanism_agent, design_principles_agent, unexpected_properties_agent, comparison_agent, novelty_agent, critic_agent#sequence_retriever,
               ], messages=[], max_round=50, admin_name='user', send_introductions=True, allow_repeat_speaker=True,
    speaker_selection_method='auto',
)

manager = autogen.GroupChatManager(groupchat=groupchat, 
                                   llm_config=gpt4turbo_config, 
                                   system_message='you dynamically select a speaker.')