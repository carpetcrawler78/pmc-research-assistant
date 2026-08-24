# Capstone1-text-prediction-for-scientific-papers
Building a model to predict keywords and abstracts for scientific papers. By Markus, Thomas and Viktor 

## The Goal
The goal of our project is to build a model that can summarize scientific findings in a meaningful way while also giving feedback about the reliability of the summary. Since Pubmed papers are easily accessible while they are also presented in XML format, only Pubmed papers are considered for our modeling approach. We intend to cover all scientific fields that did publish their research work on Pubmed. However, we might focus on only one scientific field if model performance does not increase fast enough. Potential stake holders are universities, science communicators, scientific publishers, students, science journalists, scientifically interested non professionals and further entities that have an interest in brief descriptions of scientific research. The project development aims for a incremental modeling approach were we start with a simple task and increase the capability with every model iteration. These are the sub goals leading to the final model:
* Prediction of keywords
* Prediction of abstract
* Implementation of prediction reliability
* Optional: Prediction of discussion
* Optional: Creation of a RAG system from the paper database

## Download Data
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz
* Big data set to extract training and test data

https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.incr.2026-03-04.tar.gz
* Small data set for proof of concept testing


## Set up your Environment

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the notebooks.

### **`macOS`** type the following commands : 

- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```



  

