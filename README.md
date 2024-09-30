# SciAgents
## Automating scientific discovery through multi-agent intelligent graph reasoning
A. Ghafarollahi, M.J. Buehler*

Massachusetts Institute of Technology

*mbuehler@MIT.EDU

## Summary
A key challenge in artificial intelligence is the creation of systems capable of autonomously advancing scientific understanding by exploring novel domains, identifying complex patterns, and uncovering previously unseen connections in vast scientific data. In this work, we present SciAgents, an approach that leverages three core concepts: (1) the use of large-scale ontological knowledge graphs to organize and interconnect diverse scientific concepts, (2) a suite of large language models (LLMs) and data retrieval tools, and (3) multi-agent systems with in-situ learning capabilities. Applied to biologically inspired materials, SciAgents reveals hidden interdisciplinary relationships that were previously considered unrelated, achieving a scale, precision, and exploratory power that surpasses traditional human-driven research methods. The framework autonomously generates and refines research hypotheses, elucidating underlying mechanisms, design principles, and unexpected material properties. By integrating these capabilities in a modular fashion, the intelligent system yields material discoveries, critique and improve existing hypotheses, retrieve up-to-date data about existing research, and highlights their strengths and limitations. Our case studies demonstrate scalable capabilities to combine generative AI, ontological representations, and multi-agent modeling, harnessing a `swarm of intelligence' similar to biological systems. This provides new avenues for materials discovery and accelerates the development of advanced materials by unlocking Nature’s design principles. 

![Fig_1](https://github.com/user-attachments/assets/3cae1052-427a-407c-8c9d-629111a3c070)

Figure 1. **Overview of the multi-agent graph-reasoning system developed here**  
**Panel a**: Overview of graph construction, as reported in [M.J. Buehler et al., 2024](https://iopscience.iop.org/article/10.1088/2632-2153/ad7228/meta). The visual shows the progression from scientific papers as a data source to graph construction, with the image on the right showing a zoomed-in view of the graph.  
**Panels b and c**: Two distinct approaches are presented. In **b**, a multi-agent system based on a pre-programmed sequence of interactions between agents ensures consistency and reliability. In **c**, a fully automated, flexible multi-agent framework adapts dynamically to the evolving research context. Both systems leverage a sampled path within a global knowledge graph as context to guide the research idea generation process. Each agent plays a specialized role: **Ontologist** defines key concepts and relationships, **Scientist 1** crafts a detailed research proposal, **Scientist 2** expands and refines the proposal, **Critic agent** conducts a thorough review and suggests improvements.  In the second approach, **Planner** develops a detailed plan, and the **Assistant** checks the novelty of the generated research hypotheses.
This collaborative framework enables the generation of innovative and well-rounded scientific hypotheses that extend beyond conventional human-driven methods.

![silk_energy_results](https://github.com/user-attachments/assets/19c5e9d9-d6d1-4d9b-9a66-8bda742c7579)

Figure 2: Results from our multi-agent model, illustrating a novel research hypothesis based on a knowledge
graph connecting the keywords “silk” and “energy-intensive”, as an example. This visual overview shows that the
system produces detailed, well-organized documentation of research development with multiple pages and detailed text
(the example shown here includes 8,100 words).

### Codes
This repository contains code for generating novel research ideas in the field of bio-inspired materials.

The notebook files ```SciAgents_ScienceDiscovery_GraphReasoning_non-automated.ipynb``` and ```SciAgents_ScienceDiscovery_GraphReasoning_automated.ipynb``` in the Notebooks directory correspond to the non-automated and automated multi-agent frameworks, respectively, as explained in the accompanying paper.

The automated multi-agent model is implemented in [AutoGen](https://github.com/microsoft/autogen), an open-source ecosystem for agent-based AI modeling. 

### Audio file generation (podcast style, lecture, summary and others)

Please see: [lamm-mit/PDF2Audio](https://github.com/lamm-mit/PDF2Audio) or use the version at 🤗 Hugging Face Spaces [lamm-mit/PDF2Audio](https://huggingface.co/spaces/lamm-mit/PDF2Audio).

### Example
https://github.com/user-attachments/assets/d5a972f8-5308-4e42-b7dc-d68ba84e2140


### Requirements

You need to install the GraphReasoning package, as describe below. Further, (a) OpenAI and (b) Semantic Scholar APIs are required to run the codes. 

#### Graph Reasoning installation 

Install directly from GitHub:
```
pip install git+https://github.com/lamm-mit/GraphReasoning
```
Or, editable:
```
pip install -e git+https://github.com/lamm-mit/GraphReasoning.git#egg=GraphReasoning
```
You may need wkhtmltopdf:
```
sudo apt-get install wkhtmltopdf
```
#### Graph file:
```
from huggingface_hub import hf_hub_download   
graph_name='large_graph_simple_giant.graphml'
filename = f"{graph_name}"
file_path = hf_hub_download(repo_id='lamm-mit/bio-graph-1K', filename=filename,  local_dir='./graph_giant_component')
```

#### Embeddings:
```
from huggingface_hub import hf_hub_download
embedding_name='embeddings_simple_giant_ge-large-en-v1.5.pkl'
filename = f"{embedding_name}"
file_path = hf_hub_download(repo_id='lamm-mit/bio-graph-1K', filename=filename,  local_dir='./graph_giant_component')
```

### Additional background

![Fig_2](https://github.com/user-attachments/assets/88f6a9f3-77b5-4b9c-ad7a-73e4b0841f0b)

Figure 3. Overview of the entire process from initial keyword selection to the final document, following a hierarchical expansion strategy where answers are successively refined and improved, enriched with retrieved data, critiqued and amended by identification or critical modeling, simulation and experimental tasks. The process begins with initial keyword identification or random exploration within a graph, followed by path sampling to create a subgraph of relevant concepts and relationships. This subgraph forms the basis for generating structured output in JSON, including the hypothesis, outcome, mechanisms, design principles, unexpected properties, comparison, and novelty. Each component is subsequently expanded on with individual prompting, to yield significant amount of additional detail, forming a comprehensive draft. This draft then undergoes a critical review process, including amendments for modeling and simulation priorities (e.g., molecular dynamics) and experimental priorities (e.g., synthetic biology). The final integrated draft, along with critical analyses, results in a document that guides further scientific inquiry.

![Fig_3](https://github.com/user-attachments/assets/c356a6da-7218-42d0-b0f2-966193436f4c)


Figure 4. SciAgents presents a framework for generative materials informatics, showcasing the iterative process of ideation and reasoning driven by input data, questions, and context.} The cycle of ideation and reasoning leads to predictive outcomes, offering insights into new material designs and properties. The visual elements on the edges represent various data modalities such as images, documents, scientific data, DNA sequences, video content, and microscopy, illustrating the diverse sources of information feeding into this process.

![image](https://github.com/user-attachments/assets/c11b7448-2c7b-43ae-89f2-f0e8ecac6849)

Figure 5. Visualization of the ontological knowledge graph (left: whole graph, right: sub-graph) that organizes information. 

### Original papers

Please cite this work as:
```
@article{ghafarollahi2024sciagentsautomatingscientificdiscovery,
      title={SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning}, 
      author={Alireza Ghafarollahi and Markus J. Buehler},
      year={2024},
      eprint={2409.05556},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.05556}, 
}

@article{buehler2024graphreasoning,
	author={Markus J. Buehler},
	title={Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning},
	journal={Machine Learning: Science and Technology},
	year={2024},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ad7228},
}
```
