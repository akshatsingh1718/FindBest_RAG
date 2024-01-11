from llama_index import (
    SimpleDirectoryReader,
    Document,
)

def get_single_document(input_files: list) -> Document:
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    return document
