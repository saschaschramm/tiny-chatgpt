import random
from typing import Optional


from chatgpt.mdp import MDP
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx

random.seed(0)

def _hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def draw(tree: MDP, path: Optional[str]) -> None:
    plt.clf()
    fig: Figure = plt.gcf()
    fig.set_size_inches(30.0, 18.0)
    graph = tree.get_networkx_graph()
    
    types = nx.get_node_attributes(graph, 'type')
    pos = _hierarchy_pos(graph, tree._root.node_id)

    node_list_action = {}
    node_list_state = {}
    for key,value in pos.items():
        if key in types:
            if types[key] == "action":
                node_list_action[key] = value
            elif types[key] == "state":
                node_list_state[key] = value
            else:
                raise NotImplementedError
    
    nx.draw_networkx_nodes(graph, pos, nodelist=node_list_action, node_shape="s", node_color="white", edgecolors="black", node_size=8000) 
    nx.draw_networkx_nodes(graph, pos, nodelist=node_list_state, node_shape="s", node_color="white", edgecolors="black", node_size=8000)

    node_labels = nx.get_node_attributes(graph,'label')
    node_labels_state = {k: v for k, v in node_labels.items() if k in node_list_state}
    node_labels_action = {k: v for k, v in node_labels.items() if k in node_list_action}
    nx.draw_networkx_labels(graph, pos, labels=node_labels_action, font_color='black')
    nx.draw_networkx_labels(graph, pos, labels=node_labels_state, font_color='black')

    probs = nx.get_edge_attributes(graph, 'prob')
    alphas = []
    for edge in graph.edges():
        prob = probs[edge]
        alphas.append(prob)

    alphas = [0.2 + (0.8-0.2)*alpha for alpha in alphas]
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), alpha=alphas, edge_color='black', width=2)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=nx.get_edge_attributes(graph,'label'), font_color='black')
    plt.axis('off')
    plt.margins(0.0, 0.0)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    
