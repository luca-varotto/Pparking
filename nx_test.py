import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

n = 50
G = nx.erdos_renyi_graph(n, 0.1, seed=None, directed=False)
pos = nx.spring_layout(G)

for u,v,d in G.edges(data=True):
    d['weight'] = 1 + 10*np.random.rand()

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

path = nx.shortest_path(G,source=0,target=G.number_of_nodes()-1)
path_edges = set(zip(path,path[1:]))
nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
nx.draw(G, pos,node_color='b', node_size=10, edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Blues)
plt.show()