from copy import copy
from operator import truediv
import numpy as np
import copy as cp
from time import time
from functools import partial


def readFile():
    # with open("crossword_CB_v2.txt", "r") as f, open("diccionari_CB_v2.txt") as f2:
    # with open("crossword_CB_v2.txt", "r") as f, open("diccionari_A.txt") as f2:
    with open("crossword_A_v2.txt", "r") as f, open("diccionari_A.txt") as f2:
        crossword = f.readlines()
        words = f2.readlines()

    crossword = [i.split() for i in crossword]
    crossword = np.array(crossword, dtype=str)

    words = [i[:-1] for i in words]
    words = np.array(words, dtype=str)
    return words, crossword


def findWords(crosswordMat):
    """
    Finds the index and length of all horizontal and vertical words that have two or more letters.
    return ((idx, length), (idx, length))
    """

    def filterWordList(wordsList):
        # Filter the dictionary by length

        wordDict = {}

        for word in wordsList:
            if len(word) in wordDict:
                wordDict[len(word)].append(word)
            else:
                wordDict[len(word)] = [word]

        for i in wordDict.keys():
            wordDict[i] = np.array(wordDict[i], dtype=str)

        return wordDict

    def findlengh(crosswordMat, vertical=False):
        words_idx = np.where(crosswordMat == "0")  # find the index of all zeros
        words_idx = np.array(words_idx).transpose()

        last_row, start, length, listVar = None, None, 1, []

        for row in words_idx:
            if last_row is None:
                start = row
                last_row = row
                continue

            if row[0] == last_row[0] and row[1] == last_row[1] + 1:
                length += 1
            else:
                if length > 1:
                    if vertical:  # translate the index of the transposed matrix
                        start[0], start[1] = start[1], start[0]
                    listVar.append(
                        [start, length, vertical, len(listVar), copy(wordDict[length])]
                    )
                length = 1
                start = row
            last_row = row

        if length > 1:
            if vertical:  # translate the index of the transposed matrix
                start[0], start[1] = start[1], start[0]
            listVar.append(
                [start, length, vertical, len(listVar), copy(wordDict[length])]
            )

        return listVar

    wordDict = filterWordList(words)
    horizontal = findlengh(crosswordMat)
    vertical = findlengh(crosswordMat.transpose(), True)

    return horizontal, vertical


def findSharedLetters(horizontal, vertical):
    """
    Creates a graph containing the position of the shared letter for each word and the letter:
     (pos horizontal, pos vertical, letter)
    """

    def intersection(a1, a2, b1, b2):
        """
        Finds the intersection point of two line segments defined by two points.
        Segment a: {a1, a2}
        Segment b: {b1, b2}

        Returns the intersection point or None if doesn't exist
        """
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1

        db = np.empty_like(da)
        db[0], db[1] = -da[1], da[0]

        denom = np.dot(db, db)
        num = np.dot(db, dp)

        p = ((num / denom) * db + b1).astype(int)

        # check if the point lies on the interval of the two segments

        x1 = max(min(a1[0], a2[0]), min(b1[0], b2[0]))
        x2 = min(max(a1[0], a2[0]), max(b1[0], b2[0]))

        y1 = max(min(a1[1], a2[1]), min(b1[1], b2[1]))
        y2 = min(max(a1[1], a2[1]), max(b1[1], b2[1]))

        if (x1 <= p[0] <= x2) and (y1 <= p[1] <= y2):
            return p
        else:
            return None

    structure = np.dtype(
        [("horizontal", np.int32), ("vertical", np.int32), ("letter", "<U1")]
    )

    dim = (len(horizontal), len(vertical))
    sharedLetterGraph = np.empty(dim, dtype=structure)

    for i, ival in enumerate(horizontal):
        for j, jval in enumerate(vertical):
            p1 = ival[0], ival[0].copy()  # start, end of a horizontal word
            p2 = jval[0], jval[0].copy()  # start, end of a vertical word

            # add the length of the word
            p1[1][1] += ival[1] - 1
            p2[1][0] += jval[1] - 1

            # intersection of the words (shared letter)
            p = intersection(*p1, *p2)

            if p is None:
                sharedLetterGraph[i][j] = (-1, -1, "")
            else:
                # find the index of the letter by substracting the start of the word to the point
                x = p[1] - ival[0][1]
                y = p[0] - jval[0][0]
                sharedLetterGraph[i][j] = (x, y, "-")

    return sharedLetterGraph


def order(var, graph):
    point, length, vertical, pos, wordlist = var

    if vertical:
        graph = graph.T
    row = graph[pos]
    count = 0
    for *_, i in row:
        if i == "-":
            count += 1
    return -count, -len(wordlist)


def backtracking(crossword):
    def meetsRequirements(word, crossword, var):
        varsArr, graph, (mat, wordsInMat) = crossword
        point, length, direction, row, wordlist = var

        if word in wordsInMat:
            return False

        # if the word is vertical transpose the matrix
        if direction:
            graph = graph.T

        # search if there are any letter mismatch
        row = graph[row]
        for x, y, letter in row:
            if x == -1 or letter == "" or letter == "-":
                continue

            idx = x if not direction else y

            if word[idx] != letter:
                return False

        return True

    def updateVariables(word, crossword, var):
        crossword = cp.deepcopy(crossword)

        varsArr, graph, (mat, wordsInMat) = crossword
        point, length, vertical, pos, wordlist = var

        # if the word is vertical transpose the matrix
        if vertical:
            graph = graph.T
            mat = mat.T

        # delete first variable
        varsArr = varsArr[1:]

        # update graph
        for i, val in enumerate(graph[pos]):
            x, y, _ = val
            if x == -1:
                continue
            idx = x if not vertical else y
            graph[pos, i] = (x, y, word[idx])

        # update dictionary
        wordlist = wordlist[wordlist != word]
        var[4] = wordlist

        # Fill the matrix
        x, y = point
        idx = x if not vertical else y
        y = x if vertical else y

        for i in range(len(word)):
            # assert mat[idx, y + i] != "#"
            mat[idx, y + i] = word[i]

        wordsInMat.append(word)

        # undo transposition
        if vertical:
            graph = graph.T
            mat = mat.T

        return [varsArr, graph, (mat, wordsInMat)]

    def isCompleted(crossword):
        mat = crossword[2][0]
        res = (mat == "0").any()
        return not res

    def updateDomain(crossword, var):
        def findModifiedRestrictions(graph):
            if vertical:
                graph = graph.T

            row = graph[pos]
            lst = []
            for i, (*_, letter) in enumerate(row):
                if letter != "":
                    lst.append(i)

            return lst

        point, length, vertical, pos, wordlist = var

        lst = findModifiedRestrictions(crossword[1])

        pMeetsReq = partial(meetsRequirements, crossword=crossword)
        varArr = crossword[0]
        for i, var2 in enumerate(varArr):
            point2, length2, vertical2, pos2, wordlist2 = var2
            if pos2 in lst and vertical2 != vertical or length2 == length:
                pF = partial(pMeetsReq, var=var2)
                vF = np.vectorize(pF)
                idx = vF(wordlist2)
                if idx.any() == False:
                    return None
                varArr[i][4] = wordlist2[idx]
        return crossword

    vars = crossword[0]

    if len(vars) == 0:
        return crossword

    first = vars[0]
    wordlist = first[4]

    for word in wordlist:
        # print(f"{first} {word} {meetsRequirements(word,crossword, first)}\n")
        if meetsRequirements(word, crossword, first):
            newCrossword = updateVariables(word, crossword, first)
            print(newCrossword[2][0], "\n")
            print(newCrossword[2][1], "\n")

            newCrossword = updateDomain(newCrossword, first)

            if newCrossword is None:
                print("skip")
                continue

            # newCrossword[0].sort(key=lambda x: order(x, newCrossword[1]))

            res = backtracking(newCrossword)
            if isCompleted(res):
                return res

    return crossword


if __name__ == "__main__":
    words, crosswordMat = readFile()
    horizontal, vertical = findWords(crosswordMat)
    sharedLetterGraph = findSharedLetters(horizontal, vertical)
    varArr = horizontal + vertical
    # varArr.sort(key=lambda x: order(x, sharedLetterGraph))
    crossword = varArr, sharedLetterGraph, (crosswordMat, [])

    start_time = time()
    res = backtracking(crossword)
    elapsed_time = time() - start_time
    print(elapsed_time)
    print(res[2])