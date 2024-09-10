from ScienceDiscovery.utils import *
import os

data_dir_source='./graph_giant_component/'

embeddings_name='embeddings_simple_giant_ge-large-en-v1.5.pkl'
graph_name='large_graph_simple_giant.graphml'
tokenizer_model="BAAI/bge-large-en-v1.5"

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model) 
embedding_model = AutoModel.from_pretrained(tokenizer_model,  ) 

G = load_graph_with_text_as_JSON (data_dir=data_dir_source, graph_name=graph_name)
G = return_giant_component_of_graph  (G)
G = nx.Graph(G)
try:
    node_embeddings = load_embeddings(f'{data_dir_source}/{embeddings_name}')
except:
    print ("Node embeddings not loaded, need to regenerate.")
    node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )