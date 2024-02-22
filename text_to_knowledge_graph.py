

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: arindam
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import json  # Added import for JSON handling

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")


def extract_relations_from_model_output(text):
    relations = []
    node1, node2, edge = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if edge != '':
                relations.append({
                    'node1': node1.strip(),
                    'edge': edge.strip(),
                    'node2': node2.strip()
                })
                edge, node1, node2 = '', '', ''
        elif token == "<subj>":
            current = 's'
            if edge != '':
                relations.append({
                    'node1': node1.strip(),
                    'edge': edge.strip(),
                    'node2': node2.strip()
                })
            node2 = ''
        elif token == "<obj>":
            current = 'o'
            edge = ''
        else:
            if current == 't':
                node1 += ' ' + token
            elif current == 's':
                node2 += ' ' + token
            elif current == 'o':
                edge += ' ' + token
    if node1 != '' and edge != '' and node2 != '':
        relations.append({
            'node1': node1.strip(),
            'edge': edge.strip(),
            'node2': node2.strip()
        })
    return relations


class KB:
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["node1", "edge", "node2"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def to_json(self):  # Added method to convert KB to JSON
        return json.dumps(self.relations)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")


def from_text_to_kb(text, span_length=128, max_sequence_length=1024, verbose=False):
    # Split the text into chunks of max_sequence_length
    chunks = [text[i:i + max_sequence_length] for i in range(0, len(text), max_sequence_length)]

    kb = KB()
    for chunk in chunks:
        # Tokenize each chunk
        inputs = tokenizer([chunk], return_tensors="pt", max_length=max_sequence_length, truncation=True)

        # compute span boundaries
        num_tokens = len(inputs["input_ids"][0])
        if verbose:
            print(f"Input has {num_tokens} tokens")
        num_spans = math.ceil(num_tokens / span_length)
        if verbose:
            print(f"Input has {num_spans} spans")
        overlap = math.ceil((num_spans * span_length - num_tokens) /
                            max(num_spans - 1, 1))
        spans_boundaries = []
        start = 0
        for i in range(num_spans):
            spans_boundaries.append([start + span_length * i,
                                     start + span_length * (i + 1)])
            start -= overlap
        if verbose:
            print(f"Span boundaries are {spans_boundaries}")

        # transform input with spans
        tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                      for boundary in spans_boundaries]
        tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                        for boundary in spans_boundaries]
        inputs_spans = {
            "input_ids": torch.stack(tensor_ids),
            "attention_mask": torch.stack(tensor_masks)
        }

        # generate relations
        num_return_sequences = 3
        gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": num_return_sequences
        }
        generated_tokens = model.generate(
            **inputs_spans,
            **gen_kwargs,
        )

        # decode relations
        decoded_preds = tokenizer.batch_decode(generated_tokens,
                                               skip_special_tokens=False)

        # create kb
        i = 0
        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = extract_relations_from_model_output(sentence_pred)
            for relation in relations:
                kb.add_relation(relation)
            i += 1

    return kb

'''
text = """
Smart contracts are executed on blockchain, which means that the terms are stored in a distributed database and cannot be changed. Transactions are also processed on the blockchain, which automates payments and counterparties. Since the emergence of the digital currency Ethereum, the creation and execution of smart contracts has been simplified, as complex transactions can be programmed into the Ethereum protocol.
"""

kb = from_text_to_kb(text, verbose=True)
# kb.print()
print(kb.to_json())  # This will print the JSON representation of relations

'''

import PyPDF2
import pandas as pd


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




# Split the text into sentences
sentences = pdf_text.split('. ')

# Create a Pandas DataFrame
df = pd.DataFrame({'Sentences': sentences})

# Display the DataFrame
print(df)

sentences_list = df['Sentences'].tolist()


context_text = '```\n' + '```\n'.join(sentences_list) + '```\n'



kb = from_text_to_kb(context_text, verbose=True)

output_json = kb.to_json()