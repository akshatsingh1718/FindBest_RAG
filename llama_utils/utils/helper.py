from llama_index import (
    SimpleDirectoryReader,
    Document,
)
from typing import List


def get_single_document(input_files: List[str]) -> Document:
    if isinstance(input_files, str):
        input_files = [input_files]

    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    return document
