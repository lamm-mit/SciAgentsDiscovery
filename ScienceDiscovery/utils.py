import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Markdown and PDF Handling
import markdown
import markdown2
from weasyprint import HTML
import pdfkit

# Utility and System Libraries
import random
import re
import uuid
import time
import glob
from datetime import datetime
from copy import deepcopy
from pathlib import Path

# Progress Bars
from tqdm.notebook import tqdm
try:
    get_ipython
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting
sns.set_palette("hls")

# PyVis for Graph Visualization
from pyvis.network import Network

# IPython Display
from IPython.display import display, Markdown

# Data Processing Libraries
import pandas as pd
import numpy as np

# Machine Learning and AI
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Graph Libraries
import networkx as nx

# LangChain Document Loaders and Splitters
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader, 
    PyPDFDirectoryLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Custom Graph Reasoning Module
from GraphReasoning import *

# JSON Handling
import json

from functools import partial



def markdown_to_pdf(markdown_text, output_pdf_path):
    """
    Convert a Markdown string to a PDF file using markdown2 and pdfkit.

    Args:
    markdown_text (str): The Markdown text to convert.
    output_pdf_path (str): The path where the output PDF should be saved.
    """
    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_text)
    
    # Define CSS for smaller font size
    css = """
    <style>
    body {
        font-size: 10px;  /* Adjust the font size as needed */
    }
    </style>
    """
    
    # Combine CSS and HTML content
    full_html = f"{css}{html_content}"

    # Convert HTML to PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_md_path = f"{output_pdf_path}_{timestamp}.md"
    output_pdf_path = f"{output_pdf_path}_{timestamp}.pdf"

    # Save the Markdown text to a .md file
    with open(output_md_path, 'w') as md_file:
        md_file.write(markdown_text)   

    pdfkit.from_string(full_html, output_pdf_path)

    return output_pdf_path


def convert_response_to_JSON (text_with_json):
    match = re.search(r"\{.*\}", text_with_json, re.DOTALL)
    if match:
        json_str = match.group(0)  # This is the extracted JSON string
    
        # Step 2: Parse the JSON string into a dictionary (also performs a cleanup)
        json_obj = json.loads(json_str)
    
        # Step 3: Convert the dictionary back into a JSON-formatted string
        cleaned_json_str = json.dumps(json_obj, ensure_ascii=False)
          
        #print("JSONL file created with the extracted JSON.")
    else:
        print("No JSON content found.")
        cleaned_json_str=''
    return cleaned_json_str


def json_to_formatted_text(json_data):
    formatted_text = ""

    formatted_text += f"### Hypothesis\n{json_data['hypothesis']}\n\n"
    formatted_text += f"### Outcome\n{json_data['outcome']}\n\n"
    formatted_text += f"### Mechanisms\n{json_data['mechanisms']}\n\n"

    formatted_text += "### Design Principles\n"

    design_principles_list=json_data['design_principles']

    if isinstance(design_principles_list, list):
        for principle in design_principles_list:
            formatted_text += f"- {principle}\n"
    else:
        formatted_text += f"- {design_principles_list}\n"

    formatted_text += "\n"

    formatted_text += f"### Unexpected Properties\n{json_data['unexpected_properties']}\n\n"
    formatted_text += f"### Comparison\n{json_data['comparison']}\n\n"
    formatted_text += f"### Novelty\n{json_data['novelty']}\n"

    return formatted_text




def create_path(G, embedding_tokenizer, embedding_model, node_embeddings,
                          generate_graph_expansion=None,
                          second_hop=False, data_dir='./', save_files=False, verbatim=False,
                          keyword_1 = None, keyword_2 = None, 
                          shortest_path=True, #if set to False, do NOT use shortest path but sample a random path 
                          top_k=5, #for random walk, if shortest_path=False
                          randomness_factor=0,
                          num_random_waypoints=0,
                         ):

    if keyword_1==None or keyword_2==None:
        # Randomly pick two distinct nodes
        random_nodes = random.sample(list(G.nodes()), 2)
        
        if keyword_1==None:
             keyword_1 = random_nodes[0]
        if keyword_2==None:
             keyword_2 = random_nodes[1]
        
        if verbatim:
             print("Randomly selected nodes:", keyword_1, "and", keyword_2)
    
    print(">>> Selected nodes:", keyword_1, "and", keyword_2)
    '''
    try:
        keyword_1=keyword_1[0]
    except:
        keyword_1=keyword_1
    try:
        keyword_2  =keyword_2[0]
    except:
        keyword_2  =keyword_2
    '''
    if shortest_path:
        (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML=find_path( G, node_embeddings,
                                         embedding_tokenizer, embedding_model , second_hop=False, 
                                         data_dir=data_dir, save_files=False,
                                         keyword_1 = keyword_1, keyword_2 = keyword_2, )

    else: #random path
        print ("Random walk to get path:", keyword_1, "and", keyword_2)
        
        if randomness_factor>0 or num_random_waypoints>0:
            path, path_graph, shortest_path_length, _, _= heuristic_path_with_embeddings_with_randomization_waypoints(
                G, 
                embedding_tokenizer, 
                embedding_model, 
                keyword_1, 
                keyword_2, 
                node_embeddings, 
                top_k=5, 
                #perturbation_factor=0.1, 
                second_hop=False, 
                data_dir=data_dir, 
                verbatim=True, 
                save_files=False,
                randomness_factor=randomness_factor,
                num_random_waypoints=num_random_waypoints,
            )

        else:
            path, path_graph, shortest_path_length, _, _ = heuristic_path_with_embeddings(G, embedding_tokenizer, embedding_model, 
                                                                                      keyword_1, keyword_2, 
                                                                                      node_embeddings, top_k=top_k, 
                                                                                      second_hop=False,data_dir=data_dir, 
                                                                                      verbatim=verbatim,
                                                                                      save_files=save_files)

        
        print ("Done random walk to get path")

    print("Path:", path)

    path_list_for_vis, path_list_for_vis_string=path_list=print_path_with_edges_as_list(G, path, keywords_separator=' -- ') 
    print  ( path_list_for_vis_string )

    return path_list_for_vis, path_list_for_vis_string





def develop_qa_over_path (G, embedding_tokenizer, embedding_model,node_embeddings,
                          generate, generate_graph_expansion=None,
                          second_hop=False, data_dir='./', save_files=False, verbatim=False,
                          keyword_1 = None, keyword_2 = None, 
                          shortest_path=True, #if set to False, do NOT use shortest path but sample a random path 
                          top_k=5, #for random walk, if shortest_path=False
                          randomness_factor=0,
                          num_random_waypoints=0,
                         ):

    if generate_graph_expansion==None:
        generate_graph_expansion=generate
    if keyword_1==None or keyword_2==None:
        # Randomly pick two distinct nodes
        random_nodes = random.sample(list(G.nodes()), 2)
        
        if keyword_1==None:
             keyword_1 = random_nodes[0]
        if keyword_2==None:
             keyword_2 = random_nodes[1]
        
        if verbatim:
             print("Randomly selected nodes:", keyword_1, "and", keyword_2)
    
    print(">>> Selected nodes:", keyword_1, "and", keyword_2)
    '''
    try:
        keyword_1=keyword_1[0]
    except:
        keyword_1=keyword_1
    try:
        keyword_2  =keyword_2[0]
    except:
        keyword_2  =keyword_2
    '''
    if shortest_path:
        (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML=find_path(G, node_embeddings,
                                         embedding_tokenizer, embedding_model , second_hop=False, 
                                         data_dir=data_dir, save_files=False,
                                         keyword_1 = keyword_1, keyword_2 = keyword_2, )

    else: #random path
        print ("Random walk to get path:", keyword_1, "and", keyword_2)
        
        if randomness_factor>0 or num_random_waypoints>0:
            path, path_graph, shortest_path_length, _, _= heuristic_path_with_embeddings_with_randomization_waypoints(
                G, 
                embedding_tokenizer, 
                embedding_model, 
                keyword_1, 
                keyword_2, 
                node_embeddings, 
                top_k=5, 
                #perturbation_factor=0.1, 
                second_hop=False, 
                data_dir=data_dir, 
                verbatim=True, 
                save_files=False,
                randomness_factor=randomness_factor,
                num_random_waypoints=num_random_waypoints,
            )

        else:
            path, path_graph, shortest_path_length, _, _ = heuristic_path_with_embeddings(G, embedding_tokenizer, embedding_model, 
                                                                                      keyword_1, keyword_2, 
                                                                                      node_embeddings, top_k=top_k, 
                                                                                      second_hop=False,data_dir=data_dir, 
                                                                                      verbatim=verbatim,
                                                                                      save_files=save_files)

        
        print ("Done random walk to get path")

    print("Path:", path)

    path_list_for_vis, path_list_for_vis_string=path_list=print_path_with_edges_as_list(G, path, keywords_separator=' -- ') 
    print  ( path_list_for_vis_string )
    
    print ("---------------------------------------------")

    prompt=f"""You are a sophisticated ontologist trained in scientific research, engineering, and innovation. 
    
Given the following key concepts extracted from a comprehensive knowledge graph, your task is to define each one of the terms and discuss the relationships identified in the graph.

Consider this list of nodes and relationships from a knowledge graph between "{keyword_1}" and "{keyword_2}". 

The format of the knowledge graph is "node_1 -- relationship between node_1 and node_2 -- node_2 -- relationship between node_2 and node_3 -- node_3...."

Here is the graph:

{path_list_for_vis_string}

Make sure to incorporate EACH of the concepts in the knowledge graph in your response. 

Do not add any introductory phrases. First, define each term in the knowledge graph and then, secondly, discuss each of the relationships, with context. """

    expanded=''
    expanded=generate_graph_expansion( system_prompt='You are a creative scientist who provides accurate, detailed and valuable responses.',  
                            prompt=prompt, max_tokens=1024, temperature=.1,  )

    print ("EXPANDED: ", expanded, "\n\n")

    if expanded != "":
        expanded = f"Here is an analysis of the concepts and relationships in the graph:\n\n{expanded}\n\n"
    
    prompt=f"""You are a sophisticated scientist trained in scientific research and innovation. 
    
Given the following key concepts extracted from a comprehensive knowledge graph, your task is to synthesize a novel research hypothesis. Your response should not only demonstrate deep understanding and rational thinking but also explore imaginative and unconventional applications of these concepts. 
    
Consider this list of nodes and relationships from a knowledge graph between "{keyword_1}" and "{keyword_2}". \
The format of the graph is "node_1 -- relationship between node_1 and node_2 -- node_2 -- relationship between node_2 and node_3 -- node_3...."

Here is the graph:

{path_list_for_vis_string}

{expanded}Analyze the graph deeply and carefully, then craft a detailed research hypothesis that investigates a likely groundbreaking aspect that incorporates EACH of these concepts. Consider the implications of your hypothesis and predict the outcome or behavior that might result from this line of investigation. Your creativity in linking these concepts to address unsolved problems or propose new, unexplored areas of study, emergent or unexpected behaviors, will be highly valued.

Be as quantitative as possible and include details such as numbers, sequences, or chemical formulas. Please structure your response in JSON format, with SEVEN keys: 

"hypothesis" clearly delineates the hypothesis at the basis for the proposed research question.

"outcome" describes the expected findings or impact of the research. Be quantitative and include numbers, material properties, sequences, or chemical formula.

"mechanisms" provides details about anticipated chemical, biological or physical behaviors. Be as specific as possible, across all scales from molecular to macroscale.

"design_principles" should list out detailed design principles, focused on novel concepts and include a high level of detail. Be creative and give this a lot of thought, and be exhaustive in your response. 

"unexpected_properties" should predict unexpected properties of the new material or system. Include specific predictions, and explain the rationale behind these clearly using logic and reasoning. Think carefully.

"comparison" should provide a detailed comparison with other materials, technologies or scientific concepts. Be detailed and quantitative. 

"novelty" should discuss novel aspects of the proposed idea, specifically highlighting how this advances over existing knowledge and technology. 

Ensure your scientific hypothesis is both innovative and grounded in logical reasoning, capable of advancing our understanding or application of the concepts provided.

Here is an example structure for your response, in JSON format:

{{
  "hypothesis": "...",
  "outcome": "...",
  "mechanisms": "...",
  "design_principles": "...",
  "unexpected_properties": "...",
  "comparison": "...",
  "novelty": "...",
}}

Remember, the value of your response is as scientific discovery, new avenues of scientific inquiry and potential technological breakthroughs, with details and solid reasoning.

Make sure to incorporate EACH of the concepts in the knowledge graph in your response. 
"""
    if verbatim:
        print ("##############################################")
        print (prompt)
        print ("##############################################")

    res=generate( system_prompt='You are a creative scientist who provides accurate, detailed and valuable responses, in JSON format.',  
                            prompt=prompt, max_tokens=2048, temperature=.2,  )

    res=convert_response_to_JSON(res)

    if verbatim:
        display (Markdown(res) )

    res_dict=None
    try:    
        res_dict = json.loads(res)
        res_dict['path_string'] = path_list_for_vis_string
        res_dict['expanded'] = expanded

    except:
        print ("Dict generation failed...")
    
    return res, res_dict, path_list_for_vis_string, json_to_formatted_text(res_dict), (keyword_1, keyword_2)

def research_generation(G, embedding_tokenizer,
                        embedding_model, node_embeddings, 
                        generate, 
                        generate_graph_expansion,
                        randomness_factor, num_random_waypoints,shortest_path,
                        second_hop, data_dir, save_files, verbatim,
                        keyword_1 = None, keyword_2=None,
                       ):

    df_total = pd.DataFrame()

    res, res_data, path_string, formatted_text, (start_node, end_node) = develop_qa_over_path (G=G, 
                          embedding_tokenizer=embedding_tokenizer,
                          embedding_model=embedding_model,
                          node_embeddings=node_embeddings,
                          generate=generate,
                          generate_graph_expansion=generate_graph_expansion,
                          randomness_factor=randomness_factor, 
                          num_random_waypoints=num_random_waypoints,
                          shortest_path=shortest_path,
                          second_hop=second_hop, data_dir=data_dir, save_files=save_files, verbatim=verbatim,
                          keyword_1 = keyword_1, keyword_2=keyword_2,
                         )
 

    print (start_node, "---->", end_node)

#generate=generate_Anthropic

    expanded_text=''
    res_data_expanded={}
    #for i, field in tqdm(enumerate (res_data.keys())):
    for i, field in tqdm(enumerate (list (res_data.keys())[:7])):    
        prompt=f'''You are given a new resaerch idea:
        
{formatted_text}

This research idea was developed based on a knowledge graph that describes relationships between two concepts, {start_node} and {end_node}:
    
{path_string}
    
Now, carefully expand on this particular aspect: ```{field}```.
    
Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information, if possible, such as chemical formulas, numbers, protein sequences, processing conditions, microstructures, etc. \
Include a clear rationale and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques and codes, experimental methods, or particular analyses. 
    
Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science:
    
{res_data[field]}
    
Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ... 
'''
        res=generate( system_prompt='You are a creative scientist who provides accurate, detailed and valuable responses.',  
                                    prompt=prompt, max_tokens=2048, temperature=.2,  )
        
        display (Markdown(res [:256]))
    
        res_data_expanded[field]=res
    #    expanded_text = expanded_text+f'\n\n## Expanded field {i+1}: {field}\n\n'+res
        expanded_text = expanded_text+f'\n\n'+res
    print ('---------------------------------------------')
    
    complete=f"# Research concept between '{start_node}' and '{end_node}'\n\n### KNOWLEDGE GRAPH:\n\n{res_data['path_string']}\n\n"+f"### EXPANDED GRAPH:\n\n{res_data['expanded']}"+f"### PROPOSED RESEARCH/MATERIAL:\n\n{formatted_text}"+f'\n\n### EXPANDED DESCRIPTIONS:\n\n'+expanded_text

#display (complete)
#generate=generate_Anthropic

    prompt=f'Read this document:\n\n{complete}\n\nProvide (1) a summary of the document (in one paragraph, but including sufficient detail such as mechanisms, \
related technologies, models and experiments, methods to be used, and so on), \
and (2) a thorough critical scientific review with strengths and weaknesses, and suggested improvements. Include logical reasoning and scientific approaches.'
    critiques=generate( system_prompt='You are a critical scientist who provides accurate, detailed and valuable responses.',  
                                    prompt=prompt, max_tokens=2048, temperature=.1,  )
        
    res_data['critiques'] = critiques
    res_data['res_data_expanded'] = res_data_expanded
    
    #display(Markdown(critiques))
    complete_doc=complete+ f'\n\n## SUMMARY, CRITICAL REVIEW AND IMPROVEMENTS:\n\n'+critiques
    
    
    #generate=generate_Anthropic
    prompt=f'Read this document:\n\n{complete_doc}\n\nFrom within this document, identify the single most impactful scientific question that can be tackled with molecular modeling. \
\n\nOutline key steps to set up and conduct such modeling and simulation, with details and include unique aspects of the planned work.'
    modeling_priority=generate( system_prompt='You are a scientist who provides accurate, detailed and valuable responses.',  
                                    prompt=prompt, max_tokens=2048, temperature=.1,  )
    prompt=f'Read this document:\n\n{complete_doc}\n\nFrom within this document, identify the single most impactful scientific question that can be tackled with synthetic biology. \
\n\nOutline key steps to set up and conduct such experimental work, with details and include unique aspects of the planned work.'
    synbio_priority=generate( system_prompt='You are a  scientist who provides accurate, detailed and valuable responses.',  
                                    prompt=prompt, max_tokens=2048, temperature=.1,  )
    display (Markdown(modeling_priority))
    display (Markdown(synbio_priority))
    
    complete_doc=complete_doc+ f'\n\n## MODELING AND SIMULATION PRIORITIES:\n\n'+modeling_priority
    complete_doc=complete_doc+ f'\n\n## SYNTHETIC BIOLOGY EXPERIMENTAL PRIORITIES:\n\n'+synbio_priority
    
    res_data['modeling_priority'] = modeling_priority
    res_data['synbio_priority'] = synbio_priority
    
    output_pdf_path = f"{data_dir}/output_"
    fname=markdown_to_pdf(complete_doc, output_pdf_path)
    
    df = pd.DataFrame([res_data])
    df_total = pd.concat([df_total, df], ignore_index=True)
    #df_total.to_csv(fname)
    df_total.to_csv(fname[:-4]+'.csv') 

    return None