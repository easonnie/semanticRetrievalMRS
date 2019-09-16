# semanticRetrievalMRS
This repo contains the source code for the following paper 
* Yixin Nie, Songhe Wang, Mohit Bansal, *Revealing the Importance of Semantic Retrievalfor Machine Reading at Scale.* in EMNLP, 2019.". (PDF Coming soon)

## Introduction
The paper proposes a simple but effective pipeline system for both Question Answering and Fact Verification, achieving start-of-the-art results on [HotpotQA](https://hotpotqa.github.io) and FEVER 1.0.

The system roughly consists of 4 components (see the figure below): Term-based/Heuristic Retrieval, Neural Paragraph Retrieval, Neural Sentence Retrieval and QA/NLI.

Each neural component is trained by sampling data using upstream components and supervised by intermediate annotations provided by the data set. (Find out more in the paper.)

![pipeline_figure](img/pipeline_figure.png "Pipeline System")


More importantly, the system is used as a testbed to analyze and reveal the importance of intermediate semantic retrieval and how the retrieval performance will affect the downstream tasks on different metrics.
## Results


## Usage
(Coming Soon)

## Intermediate Retrieval Data
(Coming Soon)
