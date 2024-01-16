from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.agents import Tool
from typing import List

def setup_knowledge_base_from_text(product_catalog: str= None):
    text_splitter= CharacterTextSplitter(chunk_size= 400, chunk_overlap= 50, separator="\n")
    texts= text_splitter.split_text(product_catalog)
    print("splitter ------------------")
    print(texts[0])

    llm= ChatOpenAI(temperature= 0)
    embeddings= OpenAIEmbeddings()
    docsearch= Chroma.from_texts(
        texts, embeddings, collection_name= "product-knowledge-base"
    )
    # print(docsearch.as_retriever().get_relevant_documents("cloud comfort"))
    
    knowledge_base= RetrievalQA.from_chain_type(
        llm= llm, 
        chain_type= "stuff", 
        retriever= docsearch.as_retriever(search_kwargs= {"k" : 6}))
    return knowledge_base

def setup_knowledge_base(product_catalog: str = None, file_type: str= "text"):
    """
    Assuming that product catalog is simply a text string

    TODO Add things from pararms 
    """
    if file_type == "text":
        with open(product_catalog, "r") as f:
            product_catalog = f.read()
        
        return setup_knowledge_base_from_text(product_catalog)


def get_tools(knowledge_base: Chain)-> List[Tool]:
    """knowledge_base: it is RetrievalQA Chain"""
    tools= [
        Tool(
            name="Menu",
            func=knowledge_base.run,
            description= "useful for when you need to answer questions about the food and drinks menu"
        )
    ]

    return tools

def main():
    product_catalog = """
    Sleep Haven product 1: Luxury Cloud-Comfort Memory Foam Mattress
    Experience the epitome of opulence with our Luxury Cloud-Comfort Memory Foam Mattress. Designed with an innovative, temperature-sensitive memory foam layer, this mattress embraces your body shape, offering personalized support and unparalleled comfort. The mattress is completed with a high-density foam base that ensures longevity, maintaining its form and resilience for years. With the incorporation of cooling gel-infused particles, it regulates your body temperature throughout the night, providing a perfect cool slumbering environment. The breathable, hypoallergenic cover, exquisitely embroidered with silver threads, not only adds a touch of elegance to your bedroom but also keeps allergens at bay. For a restful night and a refreshed morning, invest in the Luxury Cloud-Comfort Memory Foam Mattress.
    Price: $999
    Sizes available for this product: Twin, Queen, King

    Sleep Haven product 2: Classic Harmony Spring Mattress
    A perfect blend of traditional craftsmanship and modern comfort, the Classic Harmony Spring Mattress is designed to give you restful, uninterrupted sleep. It features a robust inner spring construction, complemented by layers of plush padding that offers the perfect balance of support and comfort. The quilted top layer is soft to the touch, adding an extra level of luxury to your sleeping experience. Reinforced edges prevent sagging, ensuring durability and a consistent sleeping surface, while the natural cotton cover wicks away moisture, keeping you dry and comfortable throughout the night. The Classic Harmony Spring Mattress is a timeless choice for those who appreciate the perfect fusion of support and plush comfort.
    Price: $1,299
    Sizes available for this product: Queen, King

    Sleep Haven product 3: EcoGreen Hybrid Latex Mattress
    The EcoGreen Hybrid Latex Mattress is a testament to sustainable luxury. Made from 100% natural latex harvested from eco-friendly plantations, this mattress offers a responsive, bouncy feel combined with the benefits of pressure relief. It is layered over a core of individually pocketed coils, ensuring minimal motion transfer, perfect for those sharing their bed. The mattress is wrapped in a certified organic cotton cover, offering a soft, breathable surface that enhances your comfort. Furthermore, the natural antimicrobial and hypoallergenic properties of latex make this mattress a great choice for allergy sufferers. Embrace a green lifestyle without compromising on comfort with the EcoGreen Hybrid Latex Mattress.
    Price: $1,599
    Sizes available for this product: Twin, Full

    Sleep Haven product 4: Plush Serenity Bamboo Mattress
    The Plush Serenity Bamboo Mattress takes the concept of sleep to new heights of comfort and environmental responsibility. The mattress features a layer of plush, adaptive foam that molds to your body's unique shape, providing tailored support for each sleeper. Underneath, a base of high-resilience support foam adds longevity and prevents sagging. The crowning glory of this mattress is its bamboo-infused top layer - this sustainable material is not only gentle on the planet, but also creates a remarkably soft, cool sleeping surface. Bamboo's natural breathability and moisture-wicking properties make it excellent for temperature regulation, helping to keep you cool and dry all night long. Encased in a silky, removable bamboo cover that's easy to clean and maintain, the Plush Serenity Bamboo Mattress offers a luxurious and eco-friendly sleeping experience.
    Price: $2,599
    Sizes available for this product: King"""

    load_dotenv()
    knowledge_base= setup_knowledge_base_from_text(product_catalog)


    res = knowledge_base.invoke("What type of matress you have ?")
    print(res)



if __name__ == "__main__":
    main()
