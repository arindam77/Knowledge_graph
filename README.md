Entity Relation Extraction from PDF Text:

This module facilitates the extraction of relationships between entities (nodes) from text embedded in a PDF file. The implementation leverages the Hugging Face Transformers library, specifically utilizing the "Babelscape/rebel-large" model and tokenizer tailored for sequence-to-sequence tasks. The primary objective is to tokenize the provided text, generate relationships and structure the data into a Knowledge Base (KB) with distinct relations.

Key Highlights:

Model and Tokenizer: Employs the "Babelscape/rebel-large" model and associated tokenizer from the Hugging Face Transformers library.

Knowledge Base (KB): Manages unique relations, ensuring the exclusion of duplicate entries within the knowledge base.

Span Extraction: Segments the input text at full stop and space boundaries, treating each sentence as an individual span.

Output Format: The script produces relations in a JSON format.

Dependencies:

transformers
torch
json
PyPDF2
pandas




