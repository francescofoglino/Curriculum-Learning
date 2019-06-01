import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as sci
import pandas as pd

##############################################################CHOOSE PARAMETERS################################################

Mode = "MR"
EXPN = "4"
Repeats = 70
Population = 20
SearchFor = 7
Prob_Mut = 0.5
GRAPHS = False


###################################################################################################################INITIALIZING VARIABLES################################################################
f = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/rank_%s"%(EXPN,Mode),delimiter="\n",dtype="str")
f3 = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/%svalues"%(EXPN,Mode),delimiter="\n",dtype="float")
# <- change to true for JS


ExperimentDictionary = {
    "1":[5,9],
    "2":[3,18],
    "3":[4,12],
    "4":[7,7]
}

curr_MAX_LENGTH = ExperimentDictionary[EXPN][0]
numSources = ExperimentDictionary[EXPN][1]
exploredCOUNT = 0

print(curr_MAX_LENGTH,numSources)



MinimizeDictionary = {
    "JS":False,
    "TTT":True,
    "REG":False,
    "MR":False
}

MINIMIZE=MinimizeDictionary[Mode]

print(MINIMIZE)

f1 = []
f2 = []
SSPACE = []
SSPACE_labels = []
SSPACE_labels_reg = []
SCOUNT = 0

TRANSLATION_DICT = {
    "1":34.2000000000000,
    "2":32.3000000000000,
    "3":224.400000000000,
    "4":541.500000000000
}

MINIMUN_FITNESS_DICT= {
}


for a in range(len(f)):
    f1.append((f[a]+","+str(a)))
    f2.append(f[a])
for k in range(len(f1)):
    SSPACE_labels.append(f1[k].split(sep=","))
    SSPACE_labels_reg.append(f1[k].split(sep=",")+[f3[k]])
for k in f2:
    SSPACE.append((k.split(sep=",")))


def by_size(size, space):
    return [word for word in space if len(word) == size]

def Look_Ahead(pol,words):
    return [word for word in words if len(word) == len(pol)+1 and word[:-1] == pol]

def GENERATE_POPULATION(N,space):
    return np.random.choice(space,N)
def SWAP(string,i,j):
    newstring = list(string)
    newstring[i],newstring[j] = newstring[j],newstring[i]
    return newstring

def ELITE_N(pop,N):
    return sorted(pop[:N],key=lambda x: int(x[-1]))

def FITNESS_FUNCTION(curr):
    if curr in SSPACE:
        if Mode == "MR":
            #print( TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[SSPACE.index(curr)][-1])+0.00000001)
            return TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[SSPACE.index(curr)][-1])+0.00000001
        else:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
        if Mode == "MR":
            return 0.00000000001
        else:
            return float(SSPACE_labels_reg[-1][-1])

def GET_SCORE(curr):
    if curr in SSPACE:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
            return float(SSPACE_labels_reg[-1][-1])

def FITNESS_PROPORTIONAL_SELECTION(pop):
    sumFIT = 0
    probability = []
    for a in pop:
        sumFIT += FITNESS_FUNCTION(a)
    for i in range(len(pop)):
        probability.append(abs(FITNESS_FUNCTION(pop[i])/sumFIT))
    id1 = np.random.choice(len(pop), 1, p=probability)[0]
    id2 = np.random.choice(len(pop), 1, p=probability)[0]
    selection = [pop[id1],pop[id2]]
    return(selection)

def CROSS(a,b,N):
    NewGen = []
    i = 0
    while len(NewGen) < N:
        if len(a) > 1 and len(b) > 1:
            ax = random.randint(0,len(a))
            bx = random.randint(0,len(b))
            c = list(a[0:ax]) + list(b[bx:-1])
            d = list(b[0:bx]) + list(a[ax:-1])
            if len(c)>0:
                NewGen.append(c)
                i+=1
            if len(d) > 0:
                NewGen.append(d)
                i+=1
        else:
            NewGen.append(a)
            i+= 1
            NewGen.append(b)
            i+= 1
    return NewGen

def MUTATE(pop,probability):
    global MEMORY
    Mutated_POP = []
    for a in pop:
        mut_a = a
        if random.random() <= probability:
            if random.random() <= 0.5: ## change
                for k in range(len(mut_a)):
                    if len(mut_a) > 1:
                        if random.random() <= 1/len(mut_a):
                            random_pool = [item for item in [str(x) for x in range(1, numSources)] if item not in mut_a]
                            if random_pool != [] : mut_a[k] = str(random.choice(random_pool))

            else: ## add or remove
                random_pool = [item for item in [str(x) for x in range(1, numSources)] if item not in mut_a]
                if random.random() <= 0.5:
                    if random_pool != []:
                        if len(mut_a) > 1: index = int(np.random.choice(range(0,len(mut_a)),1))
                        else: index = 0
                        string = str(np.random.choice(random_pool,1)[0])
                        mut_a.insert(index,string)
                else:
                    if len(mut_a)>1:
                        if random_pool != []:
                            mut_a.pop(random.choice(range(len(mut_a))))
        #print(mut_a)
        Mutated_POP.append(mut_a)
    return Mutated_POP

def TEST_SUBROUTINES():
    testpop = [["1","2","3","4"],["5","6","7"],["7","6","5","4"],["3","2"],["1","1","1"]]
    #print(FITNESS_PROPORTIONAL_SELECTION(CROSS(["0","7","3","5"],["21","3","2"],10)))

    #print(MUTATE(testpop,1))
    #print([FITNESS_FUNCTION(x) for x in FITNESS_PROPORTIONAL_SELECTION(MUTATE(testpop,1))])

def MAX_POPULATION_FITNESS(pop):
    return max([FITNESS_FUNCTION(curr) for curr in pop])

def PRINT_RANK(pop): print([GET_RANK(a) for a in pop])


def BEST_NEIGHBOUR(neighbours):
    if MINIMIZE:
        return neighbours[np.argmin([FITNESS_FUNCTION(x) for x in neighbours])]
    else:
        return neighbours[np.argmax([FITNESS_FUNCTION(x) for x in neighbours])]

def GENETIC_ALGORITHM(population_size,probability_of_mutation,iteration_number):
    global SCOUNT
    population = GENERATE_POPULATION(population_size,SSPACE)
    i = 0
    SCOUNT = 0
    fitness_graph = []
    best_fit = FITNESS_FUNCTION(BEST_NEIGHBOUR(population))
    best_global_curr = []
    best_score = []
    NSearched = []
    terminationcount= 0
    while i < iteration_number:


        [parent1,parent2] = FITNESS_PROPORTIONAL_SELECTION(population)
        #PRINT_RANK([parent1,parent2])
        #PRINT_RANK(population)
        newpopulation = CROSS(parent1,parent2,population_size)
        newpopulation.append(parent1)
        newpopulation.append(parent2)
        newpopulation = MUTATE(newpopulation,probability_of_mutation)
        population = newpopulation
        SCOUNT+= len(newpopulation)
        best_score.append(best_fit)
        NSearched.append(SCOUNT)
        population_best = BEST_NEIGHBOUR(newpopulation)
        if MINIMIZE:
            if FITNESS_FUNCTION(population_best) < FITNESS_FUNCTION(best_global_curr):
                best_global_curr = population_best
                best_fit = FITNESS_FUNCTION(population_best)
        else:
            if FITNESS_FUNCTION(population_best) > FITNESS_FUNCTION(best_global_curr):
                best_global_curr = population_best
                best_fit = FITNESS_FUNCTION(population_best)
        i += 1

        #print(str(bestsol)+"  Fitness: "+str(FITNESS_FUNCTION(bestsol)))
    #plt.scatter(iteration,best_score,c="red")
    #plt.plot(iteration,best_score)
    #plt.xlabel("Number of iterations")
    #plt.ylabel("Rank of best found solution")
    #plt.xticks(range(0,i), labels="")
    #plt.title("P(Mutation): %f , PopulationSize : %d , Max non improving steps allowed: %d"%(probability_of_mutation,population_size,iteration_number))
    #plt.show()
    print(GET_SCORE(best_global_curr),NSearched[-1])
    #return(best_score,NSearched) RETURN FUNCTION USED FOR GRAPHS
    return(GET_SCORE(best_global_curr),NSearched[-1])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sci.sem(a)
    h = se * sci.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def GENERATE_DATA(population,prob,iteration, repeats):
    datay =[]
    datax =[]
    for i in range(repeats):
        print(i)
        ploty , plotx = GENETIC_ALGORITHM(population,prob,iteration)
        datay.append(ploty)
        datax.append(plotx)
    dfy = np.array(datay)
    dfx = np.array(datax)
    print(dfx)
    xlist = []
    meanlist = []
    lowerlist = []
    upperlist = []
    print(dfy.shape[-1])
    print(dfy[-1,-1])
    with open("GEN_%s_E%s_Y"%(Mode,EXPN),mode ="w") as file:
        for i in range(dfy.shape[0]):
            file.write("\n")
            for j in range(dfy.shape[1]):
                file.write(str(dfy[i,j]))
                if j < dfy.shape[1]-1:
                    file.write(",")
        file.close()
    with open("GEN_%s_E%s_X"%(Mode,EXPN),mode ="w") as file:
        for i in range(dfy.shape[0]):
            file.write("\n")
            for j in range(dfx.shape[1]):
                file.write(str(dfx[i,j]))
                if j < dfx.shape[1]-1:
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
        for k in range(20):
            mutationrate = 0.1*i
            print(mutationrate,"mutationrate")
            result = GENETIC_ALGORITHM(20,mutationrate,100)[0][-1]
            avgx.append(mutationrate)
            avgy.append(result)
        pltx.append(np.mean(avgx))
        plty.append(np.mean(avgy))
    #print(plty)
    #print(pltx)
    plt.ylabel("Average %s"%(Mode))
    plt.title("Tuning exp%s_%s"%(EXPN,Mode))
    plt.xlabel("MutationRate")
    plt.plot(pltx,plty)
    plt.show()

#GENERATE_DATA(20,0.5,100,70)

def GENERATE_DATA2(repeats,iterations):
    results = np.zeros((repeats,2))
    for i in range(repeats):
        a,b = (GENETIC_ALGORITHM(Population,Prob_Mut,iterations))
        results[i,0] = a
        results[i,1] = b

    print(Mode,EXPN)
    print(mean_confidence_interval(results[:,1]),"steps")
    print(mean_confidence_interval(results[:,0]),"metric")
    print("Iterations: ",SearchFor)


GENERATE_DATA2(50,SearchFor)