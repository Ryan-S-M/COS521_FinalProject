#!/usr/bin/python
from random import randint
import numpy as np
from itertools import permutations
import math

# we assume that n is a power of 2

# tournament class: defines relationships between teams
class Tournament:
    def __init__(self, n, arr=None):
        # randomize the matrix, making sure that players don't both win and lose against each other
        if arr is None:
            self.rankingMatrix = np.ones((n, n))
            for i in range(0, n):
                for j in range(i+1, n):
                    if randint(0, 1) == 1:
                        self.rankingMatrix.itemset((i, j), 0)
                    else:
                        self.rankingMatrix.itemset((j, i), 0)
        else:
            # assumes that matrix is correct if given as a parameter
            self.rankingMatrix = arr
        

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

# run random complete bracket for a given tournament on a prize vector
def RCB(tournament, prizeVec):
    if len(tournament.rankingMatrix) != len(tournament.rankingMatrix[0]):
        raise(Exception("RCB: tournament ranking matrix must be zero"))
    elif len(tournament.rankingMatrix) != len(prizeVec):
        raise(Exception("RCB: tournament ranking matrix must have same length as prize vector"))

    teamOrder = np.array(np.ones(len(prizeVec)), dtype=int)
    for i in range(0, n):
        teamOrder.itemset(i, i)

    # random initial ordering
    np.random.shuffle(teamOrder)
    finalOrdering = RCB_helper(tournament, teamOrder)
    prizes = np.ones(len(prizeVec))

    for i in range(0, len(prizeVec)):
        prizes[finalOrdering[i]] = prizeVec[i]
    return prizes


# run bracket on a set ordering
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
    #base case: only one person left
    if len(ordering) == 1:
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

# returns a random prize vector (first prize entry is always 1)
def getRandPrizeVector(n):
    randVec = np.random.rand(n)
    maxVal = np.max(randVec)
    newVec= randVec / maxVal
    newVec = np.sort(newVec)[::-1]
    return newVec

# determine how manipulable a given tournament is on a prize vector
def testManipulable(tournament, prizeVec):
    # for this tournament graph and prize vector, try it and see whether it is manipulable for all possible orderings
    n = len(tournament.rankingMatrix)
    # all possible permutations
    # note: it should be possible to significantly reduce this sample space by accounting for symmetries
    # but this is not done here (so this code may take a long time to run)
    possibleOrderings = np.array(list(permutations(range(0, n))))
    numPossibleOrderings = len(possibleOrderings)
    
    # all possible pairs
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
                # get honest tournament
                prizesHonest = RunBracket(tournament, prizeVec, elem)
                collabHonest = prizesHonest[i] + prizesHonest[j]

                tournament.collude(i, j)
                
                prizesDishonest = RunBracket(tournament, prizeVec, elem)
                collabDishonest = prizesDishonest[i] + prizesDishonest[j]

                
                # have to switch back so we don't change the underlying tournament
                tournament.collude(i, j)

                if collabHonest < collabDishonest:
                    numManip += 1
                    sumGain += collabDishonest - collabHonest
            avgGain = sumGain / numPossibleOrderings
            numManipArr.itemset(pairIndex, numManip)
            gainArr.itemset(pairIndex, avgGain)
            pairIndex += 1
    return numManipArr, gainArr
            
# returns the binary representation of k as an array
def toBinStr(k, width):
    # note: does not pad with ones
    strBin = np.binary_repr(k, width)
    return strBin
    
    
# test all 4 player tournaments on a given prize vector
# print if find one that has manipulability > 1/3
def testAll4PlayerMatrices(prizeVec):
    matrix = np.array(np.zeros((4, 4), dtype=int))
    
            
    # try all possible tournaments
    # each tournament can be thought of as a 4 choose 2 (=6) length bit string
    numPairs = int((4 * (4 - 1)) / 2)
    numPossibleTournaments = int(math.pow(2, numPairs))
    for numTourney in range(0, numPossibleTournaments):
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
        
        # test if this tournament is manipulable under this prize vector
        tournament = Tournament(4, matrix)
        numManipArr, gainArr =  testManipulable(tournament, prizeVec)

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
    
            
    # try all possible tournaments
    # each tournament can be thought of as a n choose 2  length bit string
    numPairs = int((n * (n - 1)) / 2)
    numPossibleTournaments = int(math.pow(2, numPairs))
    for numTourney in range(0, numPossibleTournaments):
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
        
        # test if this tournament is manipulable under this prize vector
        tournament = Tournament(n, matrix)
        numManipArr, gainArr =  testManipulable(tournament, prizeVec)


        numOrderings = math.factorial(n)
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


def testNPlayerMatricesManyPrizes(n, prizeVecArr):
    matrix = np.array(np.zeros((n, n), dtype=int))
    
            
    # try all possible tournaments
    # each tournament can be thought of as a n choose 2 length bit string
    numPairs = int((n * (n - 1)) / 2)
    numPossibleTournaments = int(math.pow(2, numPairs))
    numOrderings = math.factorial(n)

    for numTourney in range(0, numPossibleTournaments):
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
        
        #count = 1
        # test if this tournament is manipulable under this prize vector

        for prizeVec in prizeVecArr:
            tournament = Tournament(n, matrix)
            numManipArr, gainArr =  testManipulable(tournament, prizeVec)

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
                    print("gain vector:", gainArr, flush=True)
                elif gainArr[i] > 1 / 3:
                    print("The following tournament had a pairing that gained > 1/3")
                    print("matrix: \n", matrix)
                    print("prize vector:", prizeVec)
                    print("manipulation array:", numManipArr)
                    print("normalized manipulation array:", numManipArrNorm)
                    print("gain vector:", gainArr, flush=True)
            #print("Done with prize vector", count, " of ", len(prizeVecArr))
            #count += 1
        print("Done with tournament", numTourney, "of", numPossibleTournaments)

# make a bunch of prize vectors
# num > 3
def makePrizeVecArr(n, num):
    # make some basic prize vectors
    prizeVecs = np.ones((num, n))
    firstVec = np.ones(n)
    for i in range(0, (n // 2) -1):
        firstVec.itemset(n - i - 1, 0)
    prizeVecs[0] =  firstVec
    secondVec = np.ones(n)
    for i in range(0, (n//2) +1):
        secondVec.itemset(n - i - 1, 0)
    prizeVecs[1] = secondVec
    prizeVecs[2] = makeBordaVec(n)

    #generate some random vectors
    for i in range(0, num - 3):
        prizeVecs[i+3] = getRandPrizeVector(n)
    return prizeVecs

def makeBordaVec(n):
    borda = np.ones(n)
    for i in range(0, n):
        borda.itemset(i, (n - i - 1) / (n - 1))
    return borda
            

if __name__ == "__main__":
    # basic testing
    n = 4
    num_trials = 1000
    testNPlayerMatricesManyPrizes(n, makePrizeVecArr(n, num_trials))
    
    
    
