# semanticRetrievalMRS
This repo contains the source code for the following paper 
* Yixin Nie, Songhe Wang, Mohit Bansal, *Revealing the Importance of Semantic Retrievalfor Machine Reading at Scale.* in EMNLP, 2019.". ([arxiv](http://arxiv.org/abs/1909.08041))

## Introduction
The paper proposes a simple but effective pipeline system for both Question Answering and Fact Verification, achieving start-of-the-art results on [HotpotQA](https://hotpotqa.github.io) and FEVER 1.0.

The system roughly consists of 4 components (see the figure below): Term-based/Heuristic Retrieval, Neural Paragraph Retrieval, Neural Sentence Retrieval and QA/NLI.

Each neural component is trained by sampling data using upstream components and supervised by intermediate annotations provided by the data set. (Find out more in the paper.)

![pipeline_figure](img/pipeline_figure.png "Pipeline System")


**More importantly**, the system is used as a testbed to analyze and reveal the importance of intermediate semantic retrieval and how the retrieval performance will affect the downstream tasks on different metrics.
We hope the analysis could be insightful and inspiring for future development on OpenDomain QA/NLI systems. 
## Results

## Requirement
* Python 3.6
* (More coming)

## Usage
#### Download Data
##### Dataset
In the repo directory, run the following commands. 
```bash
mkdir data
cd data
mkdir hotpotqa
cd hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
```
##### Wikipedia
In the repo directory, run the following commands. 
```bash
cd data
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
```
(Coming Soon)

## Intermediate Retrieval Data
(Coming Soon)
