#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:32:32 2024

@author: arindam
"""




import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
import ollama
import PyPDF2


def read_pdf(file_path, num_pages_to_read=4):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = min(num_pages_to_read, len(pdf_reader.pages))

        text = ''
        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()

    return text

pdf_file_path = '10.1515_mr-2023-0040-1.pdf'
pdf_text = read_pdf(pdf_file_path, num_pages_to_read=4)


#temperature = 0.0 


# Split the text into sentences
sentences = pdf_text.split('. ')

# Create a Pandas DataFrame
df = pd.DataFrame({'Sentences': sentences})

# Display the DataFrame
print(df)

sentences_list = df['Sentences'].tolist()


context_text = '```\n' + '```\n'.join(sentences_list) + '```\n'



    
    
    
    
system_prompt = (
    "This is the entire TEXT, START   {context}    END ."
    "You are a network graph maker specializing in medical terms and their relationships. Your goal is to extract a precise medical ontology from the TEXT."
    "Consider key medical entities such as medical objects, entities, conditions, locations, treatments, procedures, medications, etc."
    "Ensure atomicity and capture relations between terms in sentences and paragraphs only from the TEXT."
    "You are provided with context chunks of the TEXT(delimited by ```). Each context chunk is important"
    "Again mentioning that terms should be as atomistic as possible\n\n"
    "You should think about how these medical terms can have a one-on-one relation with other terms.\n"
    "All extracted terms should only come from inside the TEXT provided.\n"
    "Terms that are mentioned in the same sentence or the next sentences are typically related to each other.\n"    
    "Ensure that there are no repetitive node-to-node relationships, and each relationship is unique. \n\n"
    "Don't include anything else except medical entities in the output JSON. No long sentences. Try to generate as many relationships as possible but should be perfect and useful and only from within the TEXT \n\n"
    "Format your output as a list of JSON. Each element of the list contains a pair of medical terms "
    "and the relation between them, as shown in this example: ["                    
    "{{"
    '"sentence": "A portion of the TEXT which best depicts the relation between the nodes and edges for extracting medical terms", '
    '"node_1": "A medical concept or entity from extracted ontology", '
    '"node_2": "A related medical concept or entity from extracted ontology", '
    '"edge": "Relationship between the two entities or concepts, node_1 and node_2 in few words from the text provided"'
    "}}"
    "]"
    "Multiple nodes and edges relationships can be derived from one sentence."
    "Donot return empty values for JSON keys. If empty, skip"
    "All nodes and edges should come only from within the TEXT provided here in the prompt"
    "Iterate through each sentence and get as many samples as possible"
)






user_prompt = "Generate the JSON containing nodes and edges as requested in the context"

# Format the prompts
formatted_system_prompt = system_prompt.format(context=context_text)
formatted_user_prompt = user_prompt

# Define messages for Ollama
ollama_messages = [
    {'role': 'system', 'content': formatted_system_prompt},
    {'role': 'user', 'content': formatted_user_prompt}
]

# Use Ollama to chat
response = ollama.generate(model='zephyr', system=formatted_system_prompt, prompt=formatted_user_prompt)


# Print the response
print(response['message']['content'])


finaljson = response['message']['content']



jsonobj = json.loads(finaljson)

print(jsonobj)


df_export = pd.DataFrame(jsonobj)



# Create a NetworkX graph from the DataFrame
G = nx.from_pandas_edgelist(df_export, 'node_1', 'node_2', ['edge'])

# Use spring_layout with adjusted k parameter
pos = nx.spring_layout(G, k=17)

# Draw the graph
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)

# Add edge labels
edge_labels = nx.get_edge_attributes(G, 'edge')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

# Save the graph as an image (e.g., PNG)
plt.savefig("knowledge_graph7.png", format="PNG")

# Display the graph
plt.show()