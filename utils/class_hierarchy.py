import networkx as nx
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import self_utils


graph_dir = './nabirds-data/nabirds/'

G = nx.DiGraph()
nodes = nx.read_adjlist(graph_dir + 'classes.txt', comments=' ')
edges = nx.read_edgelist(graph_dir + 'hierarchy.txt', nodetype=int)
print(edges)
G.add_nodes_from(nodes)
G.add_edges_from(edges.edges())


plt.figure(figsize=(20, 20))
pos = graphviz_layout(G, 'twopi', 0) #'-Gsplines=true -Goverlap=false')
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=8, font_size=5)
plt.savefig("graph.png", dpi=300)

leaves = [x for x in G.nodes() if G.in_degree(x) == 1 and G.out_degree(x) == 0]
#print(len(leaves))

G.remove_edge(0,22)
np_depth = nx.shortest_path_length(G,0)
p_depth = nx.shortest_path_length(G,22)

G_u = G.to_undirected()

#print(np_depth)
#print(p_depth)
np_depths = {0:[], 1:[], 2:[], 3:[]}
for k, v in np_depth.items(): np_depths[v].append(k)
p_depths = {0:[], 1:[], 2:[], 3:[]}
for k, v in p_depth.items(): p_depths[v].append(k)
print('Layer 1 classes: ' + str(len(np_depths[0] + p_depths[0])))
print('Layer 2 classes: ' + str(len(np_depths[1] + p_depths[1])))
print('Layer 3 classes: ' + str(len(np_depths[2] + p_depths[2])))
print('Layer 4 classes: ' + str(len(np_depths[3] + p_depths[3])))

#depth = [depth[x] for x in leaves]
#print(depth)

full_depths = []
full_depths.append(np_depths[0] + p_depths[0])
full_depths.append(np_depths[1] + p_depths[1])
full_depths.append(np_depths[2] + p_depths[2])
full_depths.append(np_depths[3] + p_depths[3])

label_normalized = {}
label_normalized[0] = {full_depths[0][idx]: idx for idx in range(len(full_depths[0]))}
label_normalized[1] = {full_depths[1][idx]: idx for idx in range(len(full_depths[1]))}
label_normalized[2] = {full_depths[2][idx]: idx for idx in range(len(full_depths[2]))}
label_normalized[3] = {full_depths[3][idx]: idx for idx in range(len(full_depths[3]))}

#print(label_normalized)


c12 = np.zeros((2,50))
c23 = np.zeros((50,404))
c34 = np.zeros((404,555))

hierarchy_list = [c12, c23, c34]

for i in range(3):
    d = full_depths[i]
    for node in d:
        parent_id = label_normalized[i][node]
        children = G.successors(node)
        for child in children:
            child_id = label_normalized[i+1][child]
            hierarchy_list[i][parent_id][child_id] = 1
        



with open('hierarchical_logit_mappings.pkl', 'wb') as f:
    pickle.dump(hierarchy_list, f)











print(list(G.predecessors(610)))
print(list(G.predecessors(24)))
print(list(G.predecessors(1)))

np_labels = {}

for node in np_depths[3]:
    path = nx.shortest_path(G_u, source=node, target=0)
    np_labels[node] = path[::-1]

p_labels = {}

for node in p_depths[3]:
    path = nx.shortest_path(G_u, source=node, target=22)
    p_labels[node] = path[::-1]

labels = np_labels | p_labels



#print(labels[1003])
#print(normalized_labels[1003])


true_class_names = self_utils.load_class_names('./nabirds-data/nabirds')

print(true_class_names)
print(len(true_class_names))

class_mappings = []
class_mappings.append([true_class_names['class'][i] for i in full_depths[0]])
class_mappings.append([true_class_names['class'][i] for i in full_depths[1]])
class_mappings.append([true_class_names['class'][i] for i in full_depths[2]])
class_mappings.append([true_class_names['class'][i] for i in full_depths[3]])

#print(class_mappings)

with open('hierarchical_class_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

#with open('hierarchical_class_labels.pkl', 'rb') as f:
#    print(pickle.load(f))

normalized_labels = {}
i = 0
for key, value in labels.items():
    new = []
    old = value
    new.append(label_normalized[0][old[0]])
    new.append(label_normalized[1][old[1]])
    new.append(label_normalized[2][old[2]])
    new.append(label_normalized[3][old[3]])
    normalized_labels[i] = new
    i = i + 1

with open('normalized_class_labels.pkl', 'wb') as f:
    pickle.dump(normalized_labels, f)

with open('normalized_class_mappings.pkl', 'wb') as f:
    pickle.dump(class_mappings, f)
