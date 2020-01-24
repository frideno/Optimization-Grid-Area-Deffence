import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd

epsilon = 0.1**3
phases_to_save = 3
prob_to_sensor = 0.9

def get_k_argmax(vec, k):
    """
    find the k argmaxes of the vector
    :param vec: vector
    :param k: the number of arg max we want to perform.
    :return: the k argmaxes of the vector
    """

    argmax_list = []
    for i in range(k):
        index = np.argmax(vec)
        argmax_list.append(index)
        vec[index] = -1

    return argmax_list

def check_sub(arrs, e):

    """
    check if there is a convergence
    :param arrs: array of vectors.
    :return: if there is two vectors in here that their diffrence is less than epsilon
    """
    if len(arrs) == 0: 
        return False

    es = np.array([e] * len(arrs[0]))
    for i in range(len(arrs)):
        for j in range(len(arrs)):
            cmprtr =  np.less(np.abs(arrs[i] - arrs[j]), es)
            if np.all(cmprtr):
                return True

    return False

def node_prob(M, r):
    '''
    find the esstimation of the probabilities of being in each node
    :param M: the transition matrix, M((0,1)^nXn), sum(row(i)) = 1
    :param r: the start vector, r((0,1)^n), sum(r) = 1
    :return: the node ranking
    '''

    curr_r = np.dot(r, M)
    rs = []
    while(not check_sub(rs, epsilon)):
        rs = []
        for i in range(phases_to_save):
            rs.append(curr_r)
        curr_r = np.dot(curr_r, M)


    return curr_r



def find_sensors(M, r, k):
    '''
    find the places where the sensores should be located
    :param M: the transition matrix, M((0,1)^nXn), sum(row(i)) = 1
    :param r: the start vector, r((0,1)^n), sum(r) = 1
    :param k: the number of sensors
    :return: list of the nodes that sensors should be placed there
    '''

    places = node_prob(M, r)
    return get_k_argmax(places, k)

def calc_prob_to_caught(r_prob, sensors):
    '''
    calculate the probability to be caught
    :param r_prob: the probability to be in each node
    :param sensors: the sensors places
    :return: the probability to be caught
    '''
    sum = 0
    for s in sensors:
        sum += r_prob[s]
    return sum * prob_to_sensor


def block_edge(M, edge):
    '''
    block edge in M
    :param M: the transition matrix, right markovian (sum of rows is 1)
    :param edge: edge to remove
    :return: M after remove the edge
    '''
    v = edge[0]
    u = edge[1]

    p = 1 - M[v][u]
    if (p == 0):
        return M
    M[v][u] = 0
    for i in range(M[0].size):
        M[v][i] = M[v][i] / p
    return M

def get_sensors_with_roadblocks(M, r, k, l):
    '''
    find the places where the sensores should be located
    :param M: the transition matrix, right markovian (sum of rows is 1)
    :param r: the start vector, r((0,1)^n), sum(r) = 1
    :param k: the number of sensors
    :param l: the number of roadblocks
    :return: D - the places to set sensors, E - the edges to put road blocks, max_prob - the probability at the end of the algorithm.
    '''
    n = r.size # number of nodes
    M_next = M.copy()
    E = []    

    for roadblock in range(l):
        D = find_sensors(M_next, r, k)
        A = node_prob(M_next, r)
        prob_to_cuaght = calc_prob_to_caught(A, D)

        worst_edge = None
        max_prob = prob_to_cuaght
        # run on edges
        for i in range(n):
            for j in range(n):
                if i == j or M_next[i][j] == 0:
                    continue

                #remove edge
                edge = (i, j)
                M_temp = M_next.copy()
                M_temp = block_edge(M_temp, edge)

                # calculate with the removed edge
                s = find_sensors(M_temp, r, k)
                pr = node_prob(M_temp, r)
                temp_prob = calc_prob_to_caught(pr, s)
                if (temp_prob > max_prob):
                    worst_edge = edge
                    max_prob = temp_prob


        # if there is no need to remove the edge, probabilty didn't change:
        if (worst_edge == None): 
            return (D, E, max_prob)
        # append worst_edge to roadblocks list we eventually return.         
        else:
            E.append(worst_edge)

        # remove the edge
        M_next = block_edge(M_next, worst_edge)

    # return the new M and sensors
    return (D, E, max_prob)

def neigbors(n, nq, i):
    neigbors = []
    # not in left frame:
    if i >= nq:
       neigbors.append(i-nq)
    # not in right frame:
    if i < n - nq:
       neigbors.append(i+nq)
    # not in button frame:
    if i % nq != 0:
        neigbors.append(i-1)
    # not in ceiling frame:
    if (i+1) % nq != 0:
        neigbors.append(i+1)
    return neigbors

def generate_random_example(n, target):
    nq = int(np.sqrt(n))
    M = np.zeros((n, n))
    for i in range(n):
        nbors = neigbors(n, nq, i)
        # genreate random probs ammong neighbors that sums to 1.
        rv = [rnd.randint(1, 4) for g in nbors]
        prv = np.array(rv)/sum(rv)
        # make that the closer that i to the target, it gets higher prob, so it will converge. by sorting.
        nbors_distance_to_target = [g-target for g in nbors]
        for gd, p in zip(sorted(nbors_distance_to_target), sorted(list(prv))):
            g = gd + target
            M[i][g] = p
    r = np.zeros(n)
    num_enters = nq//2 + 1
    frame = [i for i in range(n) if len(neigbors(n, nq, i)) != 4]
    # choose random enternces:
    rnd.shuffle(frame)
    enters = frame[:num_enters]
    # devide the probs between them
    rv = [rnd.randint(1, 4) for g in enters]
    prv = np.array(rv)/sum(rv)
    for i, p in zip(enters, prv):
        r[i] = p
    
    return (M, r)


def draw_graph_and_results(M, r, sensors, block_roads, filename):
    """
    draw the graph to plot, then color  in purple the results.  enternece nodes are in triangle shape.
    save the plot to filename.png
    if blockroads != 2 it also draws the edges to remove,else only sensors.  
    """

    n = len(r)

    # add nodes, edges and lables for them.
    Gd = nx.DiGraph()
    Gd.add_nodes_from(list(range(n)))
    for i in range(n):
        for j in range(n):
            if M[i][j] > 0:
                Gd.add_weighted_edges_from([(i, j, M[i][j])])

        lables1 = {(i,j):round(M[i][j],3) for i in range(n) for j in range(i, n) if M[i][j] > 0} 
        lables2 = {(i,j):round(M[i][j],3) for i in range(n) for j in range(i, -1, -1) if M[i][j] > 0}
        nodelables = {}
        for i in range(n):
            if r[i] > 0:
                nodelables[i] = "{}\n p = {}".format(i,round(r[i], 2))
            else:
                nodelables[i] = str(i) 

        
        # create rectangular positioning.
        stx, sty = -1, -1
        pos = []
        for i in range(int(np.sqrt(n))):
             for j in range(int(np.sqrt(n))):
               pos.append([stx+i*0.25, sty+j*0.25])
     
        shapes = []
        for i in range(n):
            if r[i] > 0:
                shapes.append('v')
            else:
                shapes.append('o')


    lables = {(i,j):M[i][j] for i in range(n) for j in range(n) if M[i][j] > 0} 
    nx.draw_networkx_nodes(Gd, pos, nodelist= [i for i in range(n) if r[i] == 0], node_color='green', node_size = 1200)
    nx.draw_networkx_nodes(Gd, pos, nodelist=[i for i in range(n) if r[i]> 0], node_color='green', node_size = 1200, node_shape='v')

    # draw sensors in purple without ruing the shape it is.
    for s in sensors:
        nx.draw_networkx_nodes(Gd, pos, nodelist = [s], node_color='purple', node_size=1200, node_shape = shapes[s])
    nx.draw_networkx_edges(Gd, pos, edge_color='black')
    
    if block_roads != None:
        nx.draw_networkx_edges(Gd, pos, edge_color='yellow', edgelist=block_roads)

    nx.draw_networkx_edge_labels(Gd, pos, edge_labels=lables1, font_color='red', font_size=6, label_pos=0.5)
    nx.draw_networkx_edge_labels(Gd, pos, edge_labels=lables2, fond_color='orange', font_size=6, label_pos=0.7)
    nx.draw_networkx_labels(Gd, pos, labels=nodelables, font_size=6)


    plt.savefig(filename + '.png')

def main():
    N = int(input('enter size N so the board would be area NxN\n'))
    if N > 50:
        print('too big for your time, try other N <= 50')
        exit(1)
    n = N*N
    # generate random NxN board when the target is in the middle of the board (enternces in the frame of it).
    M, r = generate_random_example(n, n//2)
    print('part 1: k sensors')
    k = int(input('enter k = number of sensors you allow\n'))

    print("M:")
    print(M)

    print("\nr:")
    print(r)

    print("\nD = the {} sensors:".format(k))
    sensors = find_sensors(M, r, k)
    print(sensors)

    print("\nprobability to catch theifs: " + str(calc_prob_to_caught(node_prob(M, r), sensors)))

    ans = input('do you want to save the (problem, solution) graph to image? y/n\n')
    if ans == 'y':
        filename = input('insert file name to save to\n')
        draw_graph_and_results(M, r, sensors, None, filename)
        print('saved graph in {}.\ngraph explained:\ntriangle nodes are the enterence, the r[i] probability written on them.\npurple nodes are the ones with the sensors.\n'.format(filename + '.png'))


    print('part 2: k sensors, l road blocks')
    l = int(input('enter l = number of road blocks you allow\n'))
    sensors2, blocked_roads, new_prob = get_sensors_with_roadblocks(M, r, k, l)
    print('E = the {} road blocks:')
    print(blocked_roads)    
    print("probability to catch theifs now:", new_prob)

    ans = input('do you want to save the (problem, solution) graph to image? y/n\n')
    if ans == 'y':
        filename = input('insert file name to save to\n')
        draw_graph_and_results(M, r, sensors2, blocked_roads, filename)
        print('saved graph in {}.\ngraph explained:\ntriangle nodes are the enterence, the r[i] probability written on them.\npurple nodes are the ones with the sensors.\npurple edges are the ones with the blocked edges.\n'.format(filename + '.png'))

    print('Tnx! Have a good day...')
if __name__== "__main__":
    main()
