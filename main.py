import numpy as np
import copy as cp
from time import time


def readFile():
    # with open("crossword_CB_v2.txt", "r") as f, open("diccionari_CB_v2.txt") as f2:
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
    Finds the index and lengh of all horizontal and vertical words that have two or more letters.
    return ((idx, lengh), (idx, lengh))
    """

    def findlengh(_crosswordMat, swap=False):
        words_idx = np.where(_crosswordMat == "0")  # find the index of all zeros
        words_idx = np.array(words_idx).transpose()

        last_row, start, lengh, listWords = None, None, 1, []

        for row in words_idx:
            if last_row is None:
                start = row
                last_row = row
                continue

            if row[0] == last_row[0] and row[1] == last_row[1] + 1:
                lengh += 1
            else:
                if lengh > 1:
                    if swap:  # translate the index of the transposed matrix
                        start[0], start[1] = start[1], start[0]
                    listWords.append([start, lengh, "", swap, len(listWords)])
                lengh = 1
                start = row
            last_row = row

        if lengh > 1:
            if swap:  # translate the index of the transposed matrix
                start[0], start[1] = start[1], start[0]
            listWords.append([start, lengh, "", swap, len(listWords)])

        return listWords

    horizontal = np.array(findlengh(crosswordMat))
    vertical = np.array(findlengh(crosswordMat.transpose(), True))

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

            # add the lengh of the word
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
                sharedLetterGraph[i][j] = (x, y, "")

    return sharedLetterGraph


def filterWordList(wordsList, horizontal, vertical):
    # Filter the dictionary by lengh

    wordDict = {}

    for _, lengh, *_ in np.vstack((horizontal, vertical)):
        if lengh not in wordDict:
            # wordDict[lengh] = np.empty((0,), dtype=str)
            wordDict[lengh] = []

    for word in wordsList:
        if len(word) in wordDict:
            wordDict[len(word)].append(word)

        # else:
        # wordDict[len(word)] = np.empty((0,), dtype=str)
        # wordDict[len(word)] = np.append(wordDict[len(word)], word)
    for i in wordDict.keys():
        wordDict[i] = np.array(wordDict[i], dtype=str)

    return wordDict


def backtracking(crossword, wordDict: dict):
    def meetsRequirements(word, crossword, var):
        varsArr, graph, mat = crossword
        *_, direction, row = var
        # if the word is vertical transpose the matrix
        if direction:
            graph = graph.T

        # search if there are any letter mismatch
        row = graph[row]
        for x, y, letter in row:
            if x == -1 or not letter:
                continue

            idx = x if not direction else y

            if word[idx] != letter:
                return False

        return True

    def updateVariables(word, crossword, var, wordDict):
        crossword = cp.deepcopy(crossword)
        wordDict = cp.deepcopy(wordDict)

        varsArr, graph, mat = crossword
        point, lengh, word2, direction, row = var

        # if the word is vertical transpose the matrix
        if direction:
            graph = graph.T
            mat = mat.T

        # update dictionary
        wordArray = wordDict[lengh]
        wordArray = wordArray[wordArray != word]
        wordDict[lengh] = wordArray

        # delete first variable
        varsArr = varsArr[1:]

        # update graph
        for i, val in enumerate(graph[row]):
            x, y, _ = val
            if x == -1:
                continue
            idx = x if not direction else y
            graph[row, i] = (x, y, word[idx])

        # Fill the matrix
        x, y = point
        idx = x if not direction else y
        y = x if direction else y

        for i in range(len(word)):
            # assert mat[idx, y + i] != "#"
            mat[idx, y + i] = word[i]

        # undo transposition
        if direction:
            graph = graph.T
            mat = mat.T

        return (varsArr, graph, mat), wordDict

    def isCompleted(crossword):
        mat = crossword[2]
        res = (mat == "0").any()
        # res = mat == []
        return not res

    vars = crossword[0]

    if vars.size == 0:
        return crossword

    first = vars[0]
    wordlist = wordDict[first[1]]

    for word in wordlist:
        # print(f"{first} {word} {meetsRequirements(word, crossword, first)}\n")
        # print(crossword[2], "\n")

        if not meetsRequirements(word, crossword, first):
            continue

        newCrossword, newWordDict = updateVariables(word, crossword, first, wordDict)
        print(newCrossword[2], "\n")

        res = backtracking(newCrossword, newWordDict)
        if isCompleted(res):
            return res

    return crossword


if __name__ == "__main__":
    words, crosswordMat = readFile()

    horizontal, vertical = findWords(crosswordMat)
    sharedLetterGraph = findSharedLetters(horizontal, vertical)
    wordDict = filterWordList(words, horizontal, vertical)
    varArr = np.vstack((horizontal, vertical))
    varArr = list(varArr)
    varArr.sort(key=lambda x: -x[1])
    varArr = np.array(varArr)
    crossword = varArr, sharedLetterGraph, crosswordMat

    start_time = time()
    res = backtracking(crossword, wordDict)
    elapsed_time = time() - start_time
    print(elapsed_time)
    print(res[2])