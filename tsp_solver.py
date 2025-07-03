import networkx as nx
import matplotlib.pyplot as plt

def tsp_route(locations, distance_matrix):
    G = nx.complete_graph(len(locations))
    
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                G[i][j]['weight'] = distance_matrix[i][j]

    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True)
    
    route = [locations[i] for i in tsp_path]
    return route
