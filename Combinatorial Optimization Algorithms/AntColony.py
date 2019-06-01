import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as sci
import pandas as pd
import copy

Mode = "MR"
EXPN = "4"
Repeats = 70
Population = 20
SearchFor = 145
PARAMETER_DICT={
    ("REG","1"):[0.6524740257662120,0.1932],
    ("REG","2"):[0.5756,0.2040],
    ("REG","3"):[0.8149,0.6389],
    ("REG","4"):[0.7224,0.5051],
    ("JS","1"):[-44.7700,-49.9000],
    ("JS","2"):[-49.00,-50],
    ("JS","3"):[-601.7400,-2.2838e+03],
    ("JS","4"):[-360.9000,-2.6248e+03],
    ("TTT","1"):[2.7493e+03,2.9817e+03],
    ("TTT","2"):[3828,4986],
    ("TTT","3"):[315.6000,1.3516e+03],
    ("TTT","4"):[1.6437e+03,2.7881e+03]
}


f = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/rank_%s"%(EXPN,Mode),delimiter="\n",dtype="str")
f3 = np.genfromtxt("C:/Users/Chris/PycharmProjects/chessai/venv/Experiment%s/%svalues"%(EXPN,Mode),delimiter="\n",dtype="float")


global_best = 0
global_score_best = 0
best_list = []
f1 = []
f2 = []
SSPACE = []
SSPACE_labels = []
SSPACE_labels_reg = []
SCOUNT = 0
NSearched = []

TRANSLATION_DICT = {
    "1":-34.2000000000000,
    "2":-32.3000000000000,
    "3":-224.400000000000,
    "4":-541.500000000000
}


MinimizeDictionary = {
    "JS":False,
    "TTT":True,
    "REG":False,
    "MR":False
}

MINIMIZE=MinimizeDictionary[Mode]

print(MINIMIZE)


for a in range(len(f)):
    f1.append((f[a]+","+str(a)))
    f2.append(f[a])
for k in range(len(f1)):
    SSPACE_labels.append(f1[k].split(sep=","))
    SSPACE_labels_reg.append(f1[k].split(sep=",")+[f3[k]])
for k in f2:
    SSPACE.append((k.split(sep=",")))


PHEROMONESPACE = copy.deepcopy(SSPACE)
for curr in PHEROMONESPACE:
    curr.append(0)

AntsList = []


def getPheromone(curr):
    return float(PHEROMONESPACE[SSPACE.index(curr)][-1])

def getFitness(curr):
    if curr in SSPACE:
        if Mode == "MR":
            return -TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[SSPACE.index(curr)][-1])+0.00000001
        else:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
        if Mode == "MR":
            return  -TRANSLATION_DICT[EXPN]+float(SSPACE_labels_reg[-1][-1])+0.00000001
        else:
            return float(SSPACE_labels_reg[-1][-1])
def getScore(curr):
    if curr in SSPACE:
            return (float(SSPACE_labels_reg[SSPACE.index(curr)][-1]))
    else:
        return float(SSPACE_labels_reg[-1][-1])

def bestCurriculum(list):
    bestCurriculum = random.choice(list)
    for curr in list:
        if MINIMIZE:
            if getFitness(curr) < getFitness(bestCurriculum):
                bestCurriculum = curr
        else:
            if getFitness(curr) > getFitness(bestCurriculum):
                bestCurriculum = curr
    return bestCurriculum

class Ant():

    def __init__(self,iterations,space=SSPACE,initial_state = [],greedy = False):
        AntsList.append(self)
        self.memory = []
        self.state = initial_state
        self.space = space
        self.memory = []
        self.terminated = False
        self.greedy = greedy
        self.iterations = iterations

    def getCandidateSteps(self):
        if self.state == []:
            return[curriculum for curriculum in self.space if len(curriculum) == 1]
        else:
            return [curriculum for curriculum in self.space if len(curriculum) == len(self.state) + 1 and curriculum[:-1] == self.state]

    def Distance(self,curr):
        if MINIMIZE:
            Distance = getFitness(self.state)-getFitness(curr)
            if Distance <0:
                Distance = 0
        else:
            Distance = getFitness(curr)-getFitness(self.state)
            if Distance < 0:
                Distance = 0
        return abs(Distance)

    def selectNextStep(self):
        global SCOUNT
        candidate_steps = self.getCandidateSteps()
        probability = []
        denominator = 0
        if self.greedy == False:
            for a in candidate_steps:
                denominator+= (getPheromone(a)+K)**alpha + (self.Distance(a)**beta)
                SCOUNT += 1

                best_list.append(global_score_best)
                NSearched.append(SCOUNT)
            for a in candidate_steps:



                probability.append(((getPheromone(a)+K)**alpha + (self.Distance(a)**beta))/denominator)
            #print(candidate_steps,"<- possible steps ",probability,"<-")

            #if len(candidate_steps[0])==1:
            #print("\n","\n""\n","\n",probability,"prob","\n",candidate_steps,"cand","\n",[getPheromone(a) for a in candidate_steps],"fitness")
            choice = candidate_steps[np.random.choice(range(len(candidate_steps)),p=probability)]
            #print(choice,"CHOICE")
            #print(candidate_steps,"\n",probability,"PROBABILITY")
            return choice
        else:
            try:
                return sorted(candidate_steps,key= lambda x : getPheromone(x),reverse=True)[0]
            except:
                self.terminateWalk()

    def takeStep(self):
        #print("\n")
        #print(self.state,"<- current position")
        if not(self.terminated):
            if self.getCandidateSteps() != []:
                self.state = self.selectNextStep()
                self.memory.append(self.state)

                #print(self.state,"<- Took a step")
            else:
                self.terminateWalk()

    def terminateWalk(self):
        global global_best,best_list,SCOUNT,global_score_best

        #print(self.memory,"current memory")
        #print("TERMINATED PATH")
        index_best = self.memory.index(bestCurriculum(self.memory))
        best_curr = self.memory[index_best]
        best_fitness = getFitness(best_curr)
        #print(bestCurriculum(self.memory),"best curriculum",getFitness(bestCurriculum(self.memory)),"fitness")
        if MINIMIZE:
            best_fitness = 1.0 / best_fitness

            if getFitness(best_curr) < global_best:
                global_best = getFitness(best_curr)
                global_score_best = getScore(best_curr)
        else:
            if getFitness(best_curr) > global_best:

                global_best = getFitness(best_curr)
                global_score_best = getScore(best_curr)

        AddPheremone(self.memory[index_best], abs(best_fitness))  ### addd pheremone implementation functio
        for i in range(0,index_best):
            AddPheremone(self.memory[i], abs(best_fitness))

        self.terminated = True
    def start(self):
        while self.terminated == False:
            self.takeStep()
    def restart(self):
        self.terminated = False
        self.memory = []
        self.state = []



def getConvergence():
        #print(greedAnt.selectNextStep(),"GREEDYSTEP")
    return(global_best,SCOUNT)


def Evaporate(rate):
    global PHEROMONESPACE
    for curr in PHEROMONESPACE:
        curr[-1] = curr[-1]*(1-rate)
def AddPheremone(curr,amount):
    pheroID = [SSPACE.index(curr)][-1]
    if getPheromone(curr) < PheremoneCeiling:
        PHEROMONESPACE[SSPACE.index(curr)][-1] += amount
    else:
        PHEROMONESPACE[pheroID][-1] = PheremoneCeiling


def Ant_Colony(N,Iterations):
    global PHEROMONESPACE,AntsList,global_best,SCOUNT,NSearched,best_list,global_score_best
    if Mode == "MR":
        global_best = float(SSPACE_labels_reg[-1][-1]) -TRANSLATION_DICT[EXPN]
    else:
        global_best = float(SSPACE_labels_reg[-1][-1])
    AntsList = []
    best_list = []
    global_score_best =float(SSPACE_labels_reg[-1][-1])
    NSearched = []
    SCOUNT= 0
    iter = 0
    for i in range(N):
        a = Ant(Iterations, SSPACE,[])
    while SCOUNT < Iterations:
        for ant in AntsList:
            if SCOUNT < Iterations:
                ant.start()
                ant.restart()
            else:
                break
        iter+= 1
    print(best_list[-1],NSearched[-1])
    return (best_list[-1],NSearched[-1],iter)

def GENERATE_DATA(population,iteration, repeats):
    datay =[]
    datax =[]
    for i in range(repeats):
        ploty , plotx = Ant_Colony(population,iteration)
        datay.append(ploty)
        datax.append(plotx)
    dfy = np.array(datay)
    print(dfy)
    dfx = np.array(datax)
    print(dfy.shape)
    xlist = []
    meanlist = []
    lowerlist = []
    upperlist = []
    with open("ANT_%s_E%s_Y"%(Mode,EXPN),mode ="w") as file:
        for i in range(dfy.shape[0]):
            file.write("\n")
            for j in range(dfy.shape[1]):
                file.write(str(dfy[i,j]))
                if j < dfy.shape[1]-1:
                    file.write(",")
        file.close()
    with open("ANT_%s_E%s_X"%(Mode,EXPN),mode ="w") as file:
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
    for i in range(6):
        avgx = []
        avgy = []
        for k in range(40):
            backproprate = i*0.2
            result = Ant_Colony(20, 600)[0][-1]
            avgx.append(backproprate)
            avgy.append(result)
        pltx.append(np.mean(avgx))
        plty.append(np.mean(avgy))
    #print(plty)
    #print(pltx)
    plt.plot(pltx,plty)
    plt.show()


PheremoneCeiling = 1000
K = 400
alpha = 1.5
beta = 1.2
EvaporationRate = 0.5
N = 1

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sci.sem(a)
    h = se * sci.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def GENERATE_DATA2(repeats,iterations):
    results = np.zeros((repeats,2))
    for i in range(repeats):
        a,b,c = (Ant_Colony(N,iterations))

        results[i,0] = a
        results[i,1] = b

    print(Mode,EXPN)
    print("Iterations: ",c)
    print(mean_confidence_interval(results[:,1]),"steps")
    print(mean_confidence_interval(results[:,0]),"metric")
    print("K= {}, alpha = {}, beta = {}, EvRate = {}, Ceiling= {} N_ants = {}".format(K,alpha,beta,EvaporationRate,PheremoneCeiling,N))
GENERATE_DATA2(50,SearchFor)