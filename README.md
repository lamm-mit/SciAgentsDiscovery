# SciAgents
## Automating scientific discovery through multi-agent intelligent graph reasoning
A. Ghafarollahi, M.J. Buehler*

MIT

*mbuehler@MIT.EDU

## Summary
A key challenge in artificial intelligence is the creation of systems capable of autonomously advancing
scientific understanding by exploring novel domains, identifying complex patterns, and uncovering
previously unseen connections in vast scientific data. In this work, we present SciAgents, an approach
that leverages three core concepts: (1) the use of large-scale ontological knowledge graphs to
organize and interconnect diverse scientific concepts, (2) a suite of large language models (LLMs)
and data retrieval tools, and (3) multi-agent systems with in-situ learning capabilities. Applied to
biologically inspired materials, SciAgents reveals hidden interdisciplinary relationships that were
previously considered unrelated, achieving a scale, precision, and exploratory power that surpasses
traditional human-driven research methods. The framework autonomously generates and refines
research hypotheses, elucidating underlying mechanisms, design principles, and unexpected material
properties. By integrating these capabilities in a modular fashion, the intelligent system yields material
discoveries, critique and improve existing hypotheses, retrieve up-to-date data about existing research,
and highlights their strengths and limitations. Our case studies demonstrate scalable capabilities to
combine generative AI, ontological representations, and multi-agent modeling, harnessing a ‘swarm
of intelligence’ similar to biological systems. This provides new avenues for materials discovery and
accelerates the development of advanced materials by unlocking Nature’s design principles.

![Fig_overview](https://github.com/user-attachments/assets/3cae1052-427a-407c-8c9d-629111a3c070)

**Overview of the multi-agent graph-reasoning system developed here**  
**Panel a**: Overview of graph construction, as reported in [Buehler et al., 2024](https://iopscience.iop.org/article/10.1088/2632-2153/ad7228/meta). The visual shows the progression from scientific papers as a data source to graph construction, with the image on the right showing a zoomed-in view of the graph.  
**Panels b and c**: Two distinct approaches are presented. In **b**, a multi-agent system based on a pre-programmed sequence of interactions between agents ensures consistency and reliability. In **c**, a fully automated, flexible multi-agent framework adapts dynamically to the evolving research context. Both systems leverage a sampled path within a global knowledge graph as context to guide the research idea generation process. Each agent plays a specialized role:  
- **The Ontologist** defines key concepts and relationships.  
- **Scientist 1** crafts a detailed research proposal.  
- **Scientist 2** expands and refines the proposal.  
- **The Critic agent** conducts a thorough review and suggests improvements.  
- In the second approach, **The Planner** develops a detailed plan, and the **Assistant** checks the novelty of the generated research hypotheses.

This collaborative framework enables the generation of innovative and well-rounded scientific hypotheses that extend beyond conventional human-driven methods.
