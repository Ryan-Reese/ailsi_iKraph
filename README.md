# iKraph: a comprehensive, large-scale biomedical knowledge graph for AI-powered, data-driven biomedical research

This repository contains code for paper "iKraph: a comprehensive, large-scale biomedical knowledge graph for AI-powered, data-driven biomedical research". It contains three parts:
1. Named Entity Recognition: code for training and inference on biomedical papers to extract named entities, such as genes, drugs, chemicals and diseases.
2. Relation Extraction: code for training and inference on extracting relations between the named entities.
3. Repurposing: code for drug repurposing.

## Setup

NER and RE have different python environment requirements. See the README.md file under each directory for instructions.

Repurposing data can be downloaded at [google drive](https://drive.google.com/file/d/1OliLj7OZ2M6f65Ws5ZSlU-tXdEQEIG9e/view?usp=sharing) and shoule be unzipped and put under `repurposing/data`.
* An overview of data is provided in the `README` file
* `iKraph_2020_CGD_correlationOnly` is a partial knowledge graph as of **2020-01-01**