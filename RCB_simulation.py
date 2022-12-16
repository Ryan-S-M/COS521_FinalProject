#!/usr/bin/python
from random import randint
import numpy as np
from itertools import permutations
import math

# we assume that n is a power of 2

# tournament class: defines relationships between teams
# randomly generates
class Tournament:
    def __init__(self, n, arr=None):
        #print("making random tournament with ", n, "players")

        if arr is None:
            self.rankingMatrix = np.ones((n, n))
            for i in range(0, n):
                for j in range(i+1, n):
                    if randint(0, 1) == 1:
                        self.rankingMatrix.itemset((i, j), 0)
                    else:
                        self.rankingMatrix.itemset((j, i), 0)
        else:
            self.rankingMatrix = arr
        # randomize the matrix, making sure that players don't both win and lose against each other
        

        #print(self.rankingMatrix)
    # returns true if team1 beats team2, false otherwise
    def beats(self, team1, team2):
        if team1 == team2:
            raise(Exception("undefined to determine whether a team beats itself"))
        if self.rankingMatrix[team1][team2] == 1:
            # sanity check:
            if self.rankingMatrix[team2][team1] == 0:
                return True
            else:
                raise(Exception("Faulty matrix: team ", team1, " and team ", team2, " both claim to beat each other"))
        else:
            # sanity check:
            if self.rankingMatrix[team2][team1] == 1:
                return False
            else:
                raise(Exception("Faulty matrix: team ", team1, " and team ", team2, " both claim to lose to each other"))

    # swaps whether team1 beats team2 in this tournament
    def collude(self, team1, team2):
        if team1 == team2:
            raise(Exception("undefined for a team to collude with itself"))
        if self.rankingMatrix[team1][team2] == 1:
            self.rankingMatrix.itemset((team1, team2), 0)
            self.rankingMatrix.itemset((team2, team1), 1)
        else:
            self.rankingMatrix.itemset((team1, team2), 1)
            self.rankingMatrix.itemset((team2, team1), 0)

def RCB(tournament, prizeVec):
    if len(tournament.rankingMatrix) != len(tournament.rankingMatrix[0]):
        raise(Exception("RCB: tournament ranking matrix must be zero"))
    elif len(tournament.rankingMatrix) != len(prizeVec):
        raise(Exception("RCB: tournament ranking matrix must have same length as prize vector"))


    teamOrder = np.array(np.ones(len(prizeVec)), dtype=int)
    for i in range(0, n):
        teamOrder.itemset(i, i)

    # random initial ordering
    #print("RCB team order: ", teamOrder)

    np.random.shuffle(teamOrder)
    finalOrdering = RCB_helper(tournament, teamOrder)
    prizes = np.ones(len(prizeVec))

    for i in range(0, len(prizeVec)):
        prizes[finalOrdering[i]] = prizeVec[i]
    return prizes


# run bracket on a given ordering
def RunBracket(tournament, prizeVec, teamOrder):
    if len(tournament.rankingMatrix) != len(tournament.rankingMatrix[0]):
        raise(Exception("RCB: tournament ranking matrix must be zero"))
    elif len(tournament.rankingMatrix) != len(prizeVec):
        raise(Exception("RCB: tournament ranking matrix must have same length as prize vector"))

    finalOrdering = RCB_helper(tournament, teamOrder)
    prizes = np.ones(len(prizeVec))

    for i in range(0, len(prizeVec)):
        prizes[finalOrdering[i]] = prizeVec[i]
    return prizes


# ordering: teams with more wins are ranked closer to 0
def RCB_helper(tournament, ordering):
    #print("RCB_helper:", ordering)
    #base case: only one person left
    if len(ordering) == 1:
        #print(np.array(ordering))
        return np.array(ordering)
    
    # initial ordering: divide into winners and losers
    winners = np.array(np.ones(int(len(ordering) / 2)), dtype=int)
    losers = np.array(np.ones(int(len(ordering) / 2)), dtype=int)

    for i in range(0, len(ordering), 2):
        if tournament.beats(int(ordering[i]), int(ordering[i+1])):
            winners[int(i / 2)] = ordering[i]
            losers[int(i / 2)] = ordering[i+1]
        else:
            winners[i // 2] = ordering[i+1]
            losers[i // 2] = ordering[i]
    return np.concatenate((RCB_helper(tournament, winners), RCB_helper(tournament, losers)))

    
def getRandPrizeVector(n):
    randVec = np.random.rand(n)
    # print(randVec)
    maxVal = np.max(randVec)
    newVec= randVec / maxVal
    newVec = np.sort(newVec)[::-1]
    return newVec

def testManipulable(tournament, prizeVec):
    # for this tournament graph and prize vector, try it and see whether it is manipulable for all possible orderings
    n = len(tournament.rankingMatrix)
    # all possible permutations
    possibleOrderings = np.array(list(permutations(range(0, n))))
    numPossibleOrderings = len(possibleOrderings)
    
    #print(possibleOrderings)
    # for all possible pairs
    #print("There are ", len(possibleOrderings), " possible orderings")
    #print("The prize vector is", prizeVec)

    numPairs = n * (n-1) / 2
    numManipArr = np.zeros(int(numPairs))
    gainArr = np.zeros(int(numPairs))

    pairIndex = 0
    
    for i in range(0, n):
        for j in range(i+1, n):
            # for all possible orderings, count how many are manipulable
            numManip = 0
            sumGain = 0
            for elem in possibleOrderings:
                #print("Ordering: ", elem)
                # get honest tournament
                prizesHonest = RunBracket(tournament, prizeVec, elem)
                collabHonest = prizesHonest[i] + prizesHonest[j]
                #print("honest matrix\n", t.rankingMatrix)
                #print("\thonest prizes: ", prizesHonest)
                #print("\tcollabHonest: ", collabHonest)

                tournament.collude(i, j)
                
                prizesDishonest = RunBracket(tournament, prizeVec, elem)
                collabDishonest = prizesDishonest[i] + prizesDishonest[j]
                #print("Dishonest matrix\n", t.rankingMatrix)

                #print("\tDishonest prizes: ", prizesDishonest)
                #print("\tcollabDishonest: ", collabDishonest)

                tournament.collude(i, j)

                if collabHonest < collabDishonest:
                    #print("****************COLLABORATION********************")
                    numManip += 1
                    sumGain += collabDishonest - collabHonest
            #print("when", i, "and", j, "collude, there are", numManip, "orderings where they gain")
            #print("\t in total, they gain", sumGain)
            avgGain = sumGain / numPossibleOrderings
            numManipArr.itemset(pairIndex, numManip)
            gainArr.itemset(pairIndex, avgGain)
            # temporary
            pairIndex += 1
    return numManipArr, gainArr
            
# returns the binary representation of k as an array
def toBinStr(k, width):
    #print(np.binary_repr(k))
    # note: does not pad with ones
    strBin = np.binary_repr(k, width)
    return strBin
    
    
                
def testAll4PlayerMatrices(prizeVec):
    matrix = np.array(np.zeros((4, 4), dtype=int))
    # initialize: higher numbered players beat lower numbered players
    #for i in range(0, 4):
    #    for j in range(i+1, 4):
    #        matrix.itemset((j, i), 0)

    # try base matrix
    
            
    # try all possible tournaments
    # each tournament can be thought of as a 4 choose 2 (=6) length bit string
    numPairs = int((4 * (4 - 1)) / 2)
    numPossibleTournaments = int(math.pow(2, numPairs))
    #print(numPossibleTournaments)
    for numTourney in range(0, numPossibleTournaments):
        #print(numTourney)
        #print(toBinStr(numTourney, numPairs))
        binStr = toBinStr(numTourney, numPairs)
        # form matrix
        index = 0
        for i in range(0, 4):
            for j in range(i+1, 4):
                item = int(binStr[index])
                matrix.itemset((i, j), item)
                if item == 1:
                    matrix.itemset((j, i), 0)
                else:
                    matrix.itemset((j, i), 1)

                index += 1
        
        #print(matrix)
        # test if this tournament is manipulable under this prize vector
        tournament = Tournament(4, matrix)
        numManipArr, gainArr =  testManipulable(tournament, prizeVec)
        #print(numManipArr)
        #print(gainArr)

        numOrderings = math.factorial(4)
        # check if any of the pairs can collude more than 1/3 of the time
        # or whether they can gain more than 1/3 on average by colluding
        numManipArrNorm = numManipArr / numOrderings
        for i in range(0, numPairs):
            if numManipArrNorm[i] > 1 / 3:
                print("The following tournament was manipulable by > 1/3")
                print("matrix: \n", matrix)
                print("prize vector:", prizeVec)
                print("manipulation array:", numManipArr)
                print("normalized manipulation array:", numManipArrNorm)
                print("gain vector:", gainArr)
            elif gainArr[i] > 1 / 3:
                print("The following tournament had a pairing that gained > 1/3")
                print("matrix: \n", matrix)
                print("prize vector:", prizeVec)
                print("manipulation array:", numManipArr)
                print("normalized manipulation array:", numManipArrNorm)
                print("gain vector:", gainArr)



def testAllNPlayerMatrices(n, prizeVec):
    matrix = np.array(np.zeros((n, n), dtype=int))
    # initialize: higher numbered players beat lower numbered players
    #for i in range(0, 4):
    #    for j in range(i+1, 4):
    #        matrix.itemset((j, i), 0)

    # try base matrix
    
            
    # try all possible tournaments
    # each tournament can be thought of as a 4 choose 2 (=6) length bit string
    numPairs = int((n * (n - 1)) / 2)
    numPossibleTournaments = int(math.pow(2, numPairs))
    #print(numPossibleTournaments)
    for numTourney in range(0, numPossibleTournaments):
        #print(numTourney)
        #print(toBinStr(numTourney, numPairs))
        binStr = toBinStr(numTourney, numPairs)
        # form matrix
        index = 0
        for i in range(0, n):
            for j in range(i+1, n):
                item = int(binStr[index])
                matrix.itemset((i, j), item)
                if item == 1:
                    matrix.itemset((j, i), 0)
                else:
                    matrix.itemset((j, i), 1)

                index += 1
        
        #print(matrix)
        # test if this tournament is manipulable under this prize vector
        tournament = Tournament(n, matrix)
        numManipArr, gainArr =  testManipulable(tournament, prizeVec)
        #print(numManipArr)
        #print(gainArr)

        numOrderings = math.factorial(n)
        # check if any of the pairs can collude more than 1/3 of the time
        # or whether they can gain more than 1/3 on average by colluding
        numManipArrNorm = numManipArr / numOrderings
        for i in range(0, numPairs):
            if numManipArrNorm[i] >= 1 / 3:
                print("The following tournament was manipulable by > 1/3")
                print("matrix: \n", matrix)
                print("prize vector:", prizeVec)
                print("manipulation array:", numManipArr)
                print("normalized manipulation array:", numManipArrNorm)
                print("gain vector:", gainArr)
            elif gainArr[i] >= 1 / 3:
                print("The following tournament had a pairing that gained > 1/3")
                print("matrix: \n", matrix)
                print("prize vector:", prizeVec)
                print("manipulation array:", numManipArr)
                print("normalized manipulation array:", numManipArrNorm)
                print("gain vector:", gainArr)
                
        
            

if __name__ == "__main__":
    # basic testing
    n= 4
    t = Tournament(n, np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 1, 1]]))

    for i in range(0,n):
        for j in range(0, n):
            if i != j:
                t.beats(i, j)
    #print("Matrix:\n", t.rankingMatrix)
    #print("prize vector: ", getRandPrizeVector(n))


    #print("Results: ", RCB(t,getRandPrizeVector(n)))

    #print("nonrandom order:", RunBracket(t, getRandPrizeVector(n), [0, 3, 1, 2]))

    #should 

    #manipNums, gains = testManipulable(t, [1, 0, 0, 0])

    #print("manipulability numbers", manipNums)
    #print("gains", gains)
    #testAll4PlayerMatrices([1, 0, 0, 0])
    testAllNPlayerMatrices(8, [1, 0, 0, 0, 0, 0, 0, 0])
    
    
