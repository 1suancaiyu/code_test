
# 无向图邻接矩阵的表示

class GraphAX:
    def __init__(self, vertx, mat): #vertx 顶点表；mat邻接矩阵
        self.vnum = len(vertx)
        self.vertx = vertx
        self.mat = mat #[mat[i][:] for i in range(vnum)]

def creat_matrix(): #create graph by adjacency matrix
    nodes = ['v0', 'v1', 'v2', 'v3', 'v4']
    matrix = [[0,1,0,1,0],
              [1,0,1,0,1],
              [0,1,0,1,0],
              [1,0,1,0,0],
              [0,1,1,0,0]]
    mygraph = GraphAX(nodes, matrix)
    return mygraph

def DFS1(graph, cur_vertx_indx=0): #递归方式
    global visited_list
    dd = float("inf")
    print(graph.vertx[cur_vertx_indx], end=" ")  # 访问顶点v
    visited_list[cur_vertx_indx] = 1  # 置已访问标记
    for w in range(graph.vnum):
        if graph.mat[cur_vertx_indx][w] != 0 and graph.mat[cur_vertx_indx][w] != dd:
            if visited_list[w] == 0:  # 存在边<v,w>并且w没有访问过
                DFS1(graph, w)



class Sstack():
    def __init__(self):
        self.slist = []

    def is_empty(self):
        if self.slist == []:
            return 1
        else:
            return 0

    def pushstack(self, data):
        self.slist.append(data)

    def popstack(self):
        self.slist.pop()

def DFS2(graph, cur_vertx_indx=0): #非递归方式
    visited_li = [0] * graph.vnum
    s = Sstack()
    s.pushstack(cur_vertx_indx)
    print(graph.vertx[cur_vertx_indx], end=' ')
    visited_li[cur_vertx_indx] = 1
    i = 0
    while s.is_empty() != 1:
        for j in range(len(graph.vertx)):
            if graph.mat[i][j] == 1 and visited_li[j] == 0:
                print(graph.vertx[j], end=' ')
                visited_li[j] = 1
                s.pushstack(j)
                i = j
        if s.is_empty() != 1:
            i = s.popstack()




if __name__ == '__main__':
    graph = creat_matrix()
    print('图的顶点表为：')
    print(graph.vertx)
    print('\n图的邻接矩阵为：')
    print(graph.mat)

    print('\n深度遍历（递归）:')  
    visited_list = [0] * graph.vnum
    DFS1(graph, 0)

    print('\n\n深度遍历（非递归）:')
    DFS2(graph, 0)