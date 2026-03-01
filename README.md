# scCognito: Integrating semantic reasoning and spatial structure for single-cell closed-loop analysis

<p align="center">
  <img src="fig1.framework.png" alt="scCognito Framework">
</p>

**scCognito** is an advanced computational framework designed to execute a series of complex biological and computational tasks, particularly in the field of single-cell spatial transcriptomics. It features a closed-loop architecture (**Teacher -> Bridge -> PLM -> Agent**) that focuses on the synergistic learning process between a Pre-trained Language Model (PLM, acting as a structural world model) and a Large Language Model (LLM, serving as a semantic reasoning engine). By deeply integrating semantic knowledge with spatial structures, scCognito optimizes biological representations for comprehensive discovery.

The main capabilities of scCognito include:

* (1) **Spatial domain identification & Cross-modal alignment:** Identifying regions with similar gene expression patterns and aligning diverse biological data into unified representations. <br>
* (2) **Cognitive feedback & Autonomous scientific exploration:** Acting as an automated AI scientist to generate hypotheses, design experiments, and iteratively optimize models. <br>
* (3) **Mechanistic interpretability & Computational perturbation:** Inferring biological mechanisms like gene regulatory networks and simulating genetic or chemical perturbations. <br>
* (4) **Temporal evolution & Large-scale data integration:** Modeling the temporal dynamics of gene expression in development or disease progression across multi-resolution datasets. <br>

## Installation

Create a separate virtual environment for version control and to avoid potential conflicts. Please install the corresponding version of the package according to the `requirements.txt`, then the scCognito pipeline can be directly used.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate
# Activate the environment (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt