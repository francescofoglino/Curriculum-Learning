import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as sci
import pandas as pd
import copy as cp
import threading

Mode = "JS"
EXPN = "2"
TabuLenght = 2
SearchFor = 43

f = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/rank_%s"%(EXPN,Mode),delimiter="\n",dtype="str")
f1 = []
f2 = []
f3 = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/%svalues"%(EXPN,Mode),delimiter="\n",dtype="float")
SSPACE = []
SSPACE_labels = []
SSPACE_labels_reg = []
SCOUNT = 0
ExperimentDictionary = {
    "1":[5,9],
    "2":[3,18],
    "3":[4,12],
    "4":[7,7]
}
curr_MAX_LENGTH = ExperimentDictionary[EXPN][0]
numSources = ExperimentDictionary[EXPN][1]
MinimizeDictionary = {
    "JS":False,
    "TTT":True,
    "REG":False,
    "MR":False
}
TRANSLATION_DICT = {
    "1":-34.2000000000000,
    "2":-32.3000000000000,
    "3":-224.400000000000,
    "4":-541.500000000000
}


MINIMIZE=MinimizeDictionary[Mode]


for a in range(len(f)):
    f1.append((f[a]+","+str(a)))
    f2.append(f[a])
for k in range(len(f1)):
    SSPACE_labels.append(f1[k].split(sep=","))
    SSPACE_labels_reg.append(f1[k].split(sep=",")+[f3[k]])
for k in f2:
    SSPACE.append((k.split(sep=",")))


#DEFINING PARAMETERS


def BY_SIZE(size, space):
    return [word for word in space if len(word) == size]

def LOOK_AHEAD(pol,words):
    return [word for word in words if len(word) == len(pol)+1 and word[:-1] == pol]

def FITNESS_FUNCTION(curr):
    if curr in SSPACE:
        if Mode == "MR":
            return TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[SSPACE.index(curr)][-1])+0.00000001
        else:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
        if Mode == "MR":
            return  TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[-1][-1])+0.00000001
        else:
            return float(SSPACE_labels_reg[-1][-1])
def GET_SCORE(curr):
    if curr in SSPACE:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
        if Mode == "MR":
            return (float(SSPACE_labels_reg[-1][-1]))
        else:
            return float(SSPACE_labels_reg[-1][-1])
def GET_NEIGHBOURS(curr,space,size=curr_MAX_LENGTH):
    CURR_LIST = [curr]
    PRENEIGHBOURS = []
    NEIGHBOURS = []
    if len(curr)>1:
        CURR_LIST.append(curr[:-1])
    if len(curr)<curr_MAX_LENGTH:
        for a in LOOK_AHEAD(curr,SSPACE):
            CURR_LIST.append(a)
    PRENEIGHBOURS = CURR_LIST
    for i in PRENEIGHBOURS:
        for j in RETURN_SWAPS(i):
            NEIGHBOURS.append(j)
    NEIGHBOURS.remove(curr)
    NEIGHBOURS_SET = set()
    for i in NEIGHBOURS: NEIGHBOURS_SET.add(tuple(i))
    return [list(a) for a in NEIGHBOURS_SET]

def SCORE(curr):
    if curr in SSPACE:
        return(int(SSPACE_labels[SSPACE.index(curr)][-1]))
    else:
        return(len(SSPACE)+1)

def BEST_NEIGHBOUR(neighbours):
    if neighbours != []:
        if MINIMIZE:
            BN = neighbours[np.argmin([FITNESS_FUNCTION(x) for x in neighbours])]
        else:
            BN = neighbours[np.argmax([FITNESS_FUNCTION(x) for x in neighbours])]
        return BN
def SWAP(string,i,j):
    newstring = string
    newstring[i],newstring[j] = newstring[j],newstring[i]
    return newstring

def RETURN_SWAPS(string): # returns all possible combinations of pairwised swaps in string
    resultants = set()
    for i in range(len(string)):
        for j in range(len(string)):
            resultants.add(tuple(SWAP(string,i,j)))

    #print("resultants")
    #print(resultants)
    return [list(a) for a in resultants]


def CURR_INDEX(curr,space):
    try:
        return (space.index(curr))
    except:
        print("CURR NOT FOUND IN SEARCH SPACE")
#print(RETURN_SWAPS(["1","2","3","4"]))
#print(LOOK_AHEAD(["0","2"],SSPACE))
#[print(CURR_INDEX(a,SSPACE)) for a in RETURN_SWAPS(SSPACE[-40])]

#print(ADD_AND_DROP(["0","1","3"],SSPACE))


#print("------------")
#print(RETURN_SWAPS(["1","2","3","4","5"]))
def TABU_SEARCH(space,maxit,maxtabu):
    global SCOUNT
    initial_solution = np.random.choice(space,1)[0]
    best_fit = GET_SCORE(initial_solution)
    best_solution = initial_solution
    current_solution = initial_solution
    candidate_list = []
    tabu_list = []
    best_score = [GET_SCORE(initial_solution)]
    SCOUNT = 0
    i=0
    NSearched = [0]
    while i< maxit:
        best_fit = GET_SCORE(best_solution)

       # print("---------- new iterations ------ \n \n")
       # print(current_solution, "current solution")

        candidate_list = []
        for curr in GET_NEIGHBOURS(current_solution,space):
            if (not (curr in tabu_list)):
                candidate_list.append(curr)
        if candidate_list == []:
            print("No curriculum in candidate list")
            break
        if current_solution in candidate_list:
            candidate_list.remove(current_solution)
        random.shuffle(candidate_list)
        best_neigh = candidate_list[0]
        for curr in candidate_list:
            SCOUNT += 1
            best_score.append(best_fit)
            NSearched.append(SCOUNT)
            if MINIMIZE:
                if FITNESS_FUNCTION(curr)< FITNESS_FUNCTION(best_neigh):
                    best_neigh = curr

            else:
                if FITNESS_FUNCTION(curr)> FITNESS_FUNCTION(best_neigh):
                    best_neigh = curr

                    #print(candidate_list,"candidate list")
        #print(tabu_list,"tabu list")
        tabu_list.append(best_neigh)
        #print(best_neigh,"best neigh")
        current_solution = cp.deepcopy(best_neigh)
        #print(current_solution, "new solution")
        if MINIMIZE:
            if FITNESS_FUNCTION(current_solution) < FITNESS_FUNCTION(best_solution):
                best_solution = current_solution
                best_fit = GET_SCORE(best_solution)

        else:
            if FITNESS_FUNCTION(current_solution) > FITNESS_FUNCTION(best_solution):
                best_solution = cp.deepcopy(current_solution)
                best_fit = GET_SCORE(best_solution)

        if len(tabu_list) > maxtabu:
            del tabu_list[0]
        i += 1  # for plotting
          # for plotting
          # for
         # for plotting
    print(best_score[-1],NSearched[-1])
    return [best_score[-1],NSearched[-1]]
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sci.sem(a)
    h = se * sci.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def GENERATE_DATA(maxtabu,iteration,repeats):
    datax= []
    datay= []
    for i in range(repeats):
        print(i)
        ploty , plotx = TABU_SEARCH(SSPACE,iteration,maxtabu)
        datax.append(plotx)
        datay.append(ploty)

    dfx = np.array(datax)
    dfy = np.array(datay)

    maxfx    = int(np.median(([len(a) for a in dfx])))
    print(maxfx)
    maxfy   = int(np.median(([len(a) for a in dfx])))
    print(dfy)
    with open("TABU_%s_E%s_Y"%(Mode,EXPN),mode ="w") as file:
        for i in range(len(dfy)):
            file.write("\n")
            for j in range(maxfy):
                try:
                    file.write(str(dfy[i][j]))
                except:
                    file.write("")
                if j < maxfy-1:
                    file.write(",")
        file.close()
    with open("TABU_%s_E%s_X"%(Mode,EXPN),mode ="w") as file:
        for i in range(len(dfx)):
            file.write("\n")
            for j in range(maxfx):
                try:
                    file.write(str(dfx[i][j]))
                except:
                    file.write("")
                if j < maxfx-1:
                    file.write(",")
        file.close()

def TEST_PARAMETERS():
    global alpha,beta,K,EvaporationRate,backproprate
    pltx = []
    plty = []
    alpha = 5
    beta = 8
    K = 10
    EvaporationRate = 0.9
    backproprate = 0.9
    for i in range(10):
        avgx = []
        avgy = []
        for k in range(40):
            maxtabu=i*4
            print(maxtabu,"maxtabu")
            result = TABU_SEARCH(SSPACE,2000,maxtabu)[0][-1]
            avgx.append(maxtabu)
            avgy.append(result)
        pltx.append(np.mean(avgx))
        plty.append(np.mean(avgy))
    #print(plty)
    #print(pltx)
    plt.plot(pltx,plty)
    plt.show()

def GENERATE_DATA2(repeats,iterations):
    results = np.zeros((repeats,2))
    for i in range(repeats):
        a,b = (TABU_SEARCH(SSPACE,iterations,TabuLenght))
        results[i,0] = a
        results[i,1] = b

    print(Mode,EXPN)
    print("Iterations :",SearchFor)
    print(mean_confidence_interval(results[:,1]),"steps")
    print(mean_confidence_interval(results[:,0]),"metric")

GENERATE_DATA2(50,SearchFor)

