# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Sophie Gee>
<Section 3>
<10/28/21>
"""

import collections
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n in self.d.keys():
            pass
        else:
            self.d[n] = set()

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u not in self.d.keys():
            self.add_node(u)
        if v not in self.d.keys():
            self.add_node(v)
        self.d[u].update(v)
        self.d[v].update(u)
        

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        for node in self.d.values():
            node.discard(n)
        self.d.pop(n)
        

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        
        self.d[v].remove(u)
        self.d[u].remove(v)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        
        if source not in self.d.keys():
            raise KeyError("Source node not in graph.")
        else:
            V = []
            Q = collections.deque()
            M = {source}

            Q.appendleft(source)
            while Q:
                A = Q.pop()
                V.append(A)
                edges = self.d[A]
                for edge in edges:
                    if edge not in M:
                        Q.appendleft(edge)
                        M.update(edge)
            return V


    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        #check to see if the source is in the graph
        if source not in self.d.keys():
            raise KeyError("Source node not in graph")

        #check to see if the target is in the graph
        if target not in self.d.keys():
            raise KeyError("Target node not in graph")

        #both in the graph, create the variables assigned below
        else:
            V = []
            Q = collections.deque()
            M = {source}
            P = {}

            #put the source on the beginning of a queue
            Q.appendleft(source)

            #while not everything is popped off of the Q
            while Q:
                #set A to last element in the queue
                A = Q.pop()
                #add A to list of visited nodes
                V.append(A)
                if A == target:
                    #if A is the target, it has been found and the path is A
                    S = [A]
                    #go through S to 
                    while source not in S: 
                        S.append(P[A])
                        A = P[A]
                    return S[::-1]
                edges = self.d[A]
                for edge in edges:
                    if edge not in M:
                        Q.appendleft(edge)
                        M.update(edge)
                        P[edge] = A

           


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.titles = set()
        self.chars = set()
        self.graph = nx.Graph()
        with open(filename, 'r') as myfile:
            lines = myfile.read()
        movies = lines.split('\n')
        for movie in movies:
            new_movies = movie.split('/')
            self.graph.add_nodes_from(new_movies)
            self.graph.add_edges_from([(new_movies[0], char) for char in new_movies[1:]])
            self.titles.add(new_movies.pop(0))
            for actor in new_movies:
                self.chars.add(actor)
        

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        shortest_path = list(nx.shortest_path(self.graph, source,target))
        length = len(shortest_path)//2
        return shortest_path,length

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        all_paths = nx.shortest_path_length(self.graph,target)               
        new_lens = [all_paths[i] // 2 for i in self.chars]
        plt.hist(new_lens, bins=[i-.5 for i in range(8)])
        plt.show()
        return np.mean(new_lens)


if __name__ == "__main__":
    g = MovieGraph()
    print(g.average_number('Kevin Bacon'))