# SciAgents
## Automating scientific discovery through multi-agent intelligent graph reasoning
A. Ghafarollahi, M.J. Buehler*

MIT

*mbuehler@MIT.EDU

## Summary
A key challenge in artificial intelligence is the creation of systems capable of autonomously advancing scientific understanding by exploring novel domains, identifying complex patterns, and uncovering previously unseen connections in vast scientific data. In this work, we present SciAgents, an approach that leverages three core concepts: (1) the use of large-scale ontological knowledge graphs to organize and interconnect diverse scientific concepts, (2) a suite of large language models (LLMs) and data retrieval tools, and (3) multi-agent systems with in-situ learning capabilities. Applied to biologically inspired materials, SciAgents reveals hidden interdisciplinary relationships that were previously considered unrelated, achieving a scale, precision, and exploratory power that surpasses traditional human-driven research methods. The framework autonomously generates and refines research hypotheses, elucidating underlying mechanisms, design principles, and unexpected material properties. By integrating these capabilities in a modular fashion, the intelligent system yields material discoveries, critique and improve existing hypotheses, retrieve up-to-date data about existing research, and highlights their strengths and limitations. Our case studies demonstrate scalable capabilities to combine generative AI, ontological representations, and multi-agent modeling, harnessing a `swarm of intelligence' similar to biological systems. This provides new avenues for materials discovery and accelerates the development of advanced materials by unlocking Nature’s design principles. 

![Fig_1](https://github.com/user-attachments/assets/3cae1052-427a-407c-8c9d-629111a3c070)

Figure 1. **Overview of the multi-agent graph-reasoning system developed here**  
**Panel a**: Overview of graph construction, as reported in [Buehler et al., 2024](https://iopscience.iop.org/article/10.1088/2632-2153/ad7228/meta). The visual shows the progression from scientific papers as a data source to graph construction, with the image on the right showing a zoomed-in view of the graph.  
**Panels b and c**: Two distinct approaches are presented. In **b**, a multi-agent system based on a pre-programmed sequence of interactions between agents ensures consistency and reliability. In **c**, a fully automated, flexible multi-agent framework adapts dynamically to the evolving research context. Both systems leverage a sampled path within a global knowledge graph as context to guide the research idea generation process. Each agent plays a specialized role:  
- **The Ontologist** defines key concepts and relationships.  
- **Scientist 1** crafts a detailed research proposal.  
- **Scientist 2** expands and refines the proposal.  
- **The Critic agent** conducts a thorough review and suggests improvements.  
- In the second approach, **The Planner** develops a detailed plan, and the **Assistant** checks the novelty of the generated research hypotheses.
This collaborative framework enables the generation of innovative and well-rounded scientific hypotheses that extend beyond conventional human-driven methods.

### Codes
This repository contains code for generating novel research ideas in the field of bio-inspired materials.

The Jupyter notebook files ```approach_1.ipynb``` and ```approach_2.ipynb``` in the main directory correspond to the non-automated and automated multi-agent frameworks, respectively, as explained in the accompanying paper.

The automated multi-agent model is implemented in [AutoGen](https://github.com/microsoft/autogen), an open-source ecosystem for agent-based AI modeling. 

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

### Additional background

![Fig_2](https://github.com/user-attachments/assets/88f6a9f3-77b5-4b9c-ad7a-73e4b0841f0b)

Figure 2. Overview of the entire process from initial keyword selection to the final document, following a hierarchical expansion strategy where answers are successively refined and improved, enriched with retrieved data, critiqued and amended by identification or critical modeling, simulation and experimental tasks. The process begins with initial keyword identification or random exploration within a graph, followed by path sampling to create a subgraph of relevant concepts and relationships . This subgraph forms the basis for generating structured output in JSON, including the hypothesis, outcome, mechanisms, design principles, unexpected properties, comparison, and novelty. Each component is subsequently expanded on with individual prompting, to yield significant amount of additional detail, forming a comprehensive draft. This draft then undergoes a critical review process, including amendments for modeling and simulation priorities (e.g., molecular dynamics) and experimental priorities (e.g., synthetic biology). The final integrated draft, along with critical analyses, results in a document that guides further scientific inquiry.

![Fig_3](https://github.com/user-attachments/assets/c356a6da-7218-42d0-b0f2-966193436f4c)


Figure 3. SciAgents presents a framework for generative materials informatics, showcasing the iterative process of ideation and reasoning driven by input data, questions, and context.} The cycle of ideation and reasoning leads to predictive outcomes, offering insights into new material designs and properties. The visual elements on the edges represent various data modalties such as images, documents, scientific data, DNA sequences, video content, and microscopy, illustrating the diverse sources of information feeding into this process.

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
