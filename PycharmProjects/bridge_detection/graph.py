class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        #self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    print(graph[start])
    for node in graph[start].get_connections():
        if node.get_id() not in path:
            newpath = find_path(graph, node.get_id(), end, path)
            if newpath: return newpath
    return None


if __name__ == '__main__':

    g = Graph()
    for i in range(0,49):
        if (i == 20 or i == 27 or i == 28 or i == 46 ):
            continue
        g.add_vertex(str(i))

    for i in range(0,19):
        g.add_edge(str(i),str(i+1),1)

    for i in range(21,26):
        g.add_edge(str(i),str(i+1),1)

    for i in range(29,48):
        if (i == 46 or i==45):
            continue
        g.add_edge(str(i),str(i+1),1)

    g.add_edge('0', '29', 1)
    g.add_edge('1', '45', 1)
    g.add_edge('45', '1', 1)
    g.add_edge('1', '47', 1)
    g.add_edge('44', '48', 1)
    g.add_edge('9', '21', 1)
    g.add_edge('21', '9', 1)
    g.add_edge('22', '26', 1)
    g.add_edge('47', '1', 1)
    for i in range(9,1,-1):
        g.add_edge(str(i),str(i-1),1)


    print(g.vert_dict.items())
    if ('1' not in g.vert_dict):
        print ('yes')
    lst = find_path(g.vert_dict,'21','48')
    print(lst)

    # for v in g:
    #     for w in v.get_connections():
    #         vid = v.get_id()
    #         wid = w.get_id()
    #         print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))
    #
    # for v in g:
    #     print('g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))