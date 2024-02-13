import cv2
import numpy as np
from grabber import Frame
import time
import os
from scipy.ndimage import shift
import pydirectinput as p

print('starting in 3...')
time.sleep(3)

BLOCK_SIZE = 21
BOARD_WIDTH = 4
PLAY_AREA_X_START = 107

PIECE_ID_MAP = { # based on first contour dimension
    6: 'o',
    7: 'i',
    11: 'l',
    12: 't',
    13: 'z',
    14: 's',
    18: 'j'
}



PIECE_BINARIES = {
    'o': np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]),
    'i': np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
    'l': np.array([
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
    ]),
    't': np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ]),
    'z': np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]),
    's': np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
    ]),
    'j': np.array([
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ]),
}

frame = Frame(444, -539, 1191, -1087)

prevBoardState = np.array(0)
prevPieceState = 0

def drop(pieceBinary, boardState):
    # drop to bottom
    while not np.any(pieceBinary[3]):
        pieceBinary = np.roll(pieceBinary, 1, axis=0)

    integerBoardState = np.where(boardState, 1, 0)
    boardHeightData = np.where(np.any(integerBoardState, axis=0), 30 - np.argmax(integerBoardState, axis=0), 0)

    pieceHeightData = np.where(np.any(pieceBinary, axis=0), np.argmax(np.flip(pieceBinary, axis=0), axis=0), np.inf)
    distances = pieceHeightData - boardHeightData

    collisionColumn = np.argmin(distances)
    boardCollisionHeight = boardHeightData[collisionColumn]
    pieceCollisionHeight = int(pieceHeightData[collisionColumn])

    boardStateCopy = boardState.copy()

    # todo : these go out of bounds before the game ends
    starter = 26-boardCollisionHeight+pieceCollisionHeight
    ender = 30-boardCollisionHeight+pieceCollisionHeight

    boardStateCopy[starter:ender] = np.bitwise_or(pieceBinary, boardState[starter:ender])
    return np.where(boardStateCopy, 1, 0)

def grader(boardState):
    numberOfClearedLines = np.sum(np.all(boardState, axis=1))
    heights = np.where(np.any(boardState, axis=0), 30 - np.argmax(boardState, axis=0), 0)
    maxHeight = np.max(heights)

    holes = 0
    zerosCounted = 0

    for colIndex in range(4):
        holes += np.sum(np.flip(boardState[:, colIndex])[:heights[colIndex]] == 0)

    return (numberOfClearedLines, maxHeight, holes)

while True:
    images = []
    originalImage = frame.grab_frame()
    
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    _, bwImage = cv2.threshold(grayImage, 76, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(bwImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    good_contours = []
    c = 0
    for contour, level in zip(contours, hierarchy):   
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if level[3] == -1 and area > 1000 and c < 2:
            good_contours.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'id': contour.shape[0]})
    
    if cv2.waitKey(1) & 0xFF == ord('`'):
            break

    if len(good_contours) < 2: continue

    bigBox, smallBox = good_contours[0], good_contours[1]

    if bigBox['area'] < smallBox['area']:
        bigBox, smallBox = smallBox, bigBox

    board = bwImage[bigBox['y']:bigBox['y']+bigBox['h'], bigBox['x']:bigBox['x']+bigBox['w']]
    shape = bwImage[smallBox['y']:smallBox['y']+smallBox['h'], smallBox['x']:smallBox['x']+smallBox['w']]

    playArea = board[:, PLAY_AREA_X_START:int(PLAY_AREA_X_START + (BLOCK_SIZE * BOARD_WIDTH))]

    boardState = [[np.sum(blockPixels == 255) > 10 for blockPixels in np.array_split(playArea[int(BLOCK_SIZE * layer):int(BLOCK_SIZE * layer) + 1][0], 4)] for layer in range(30)]
    pieceState = PIECE_ID_MAP.get(smallBox['id'], False)

    if (pieceState == False): continue

    if not np.array_equal(boardState, prevBoardState) or pieceState != prevPieceState or True:
        prevBoardState = boardState
        prevPieceState = pieceState

        # os.system('cls')
        # for layer in boardState:
        #     print(' '.join([str(int(i)) for i in layer]))
        # print(pieceState)


        pieceBinary = PIECE_BINARIES[pieceState]
        rotations = [np.rot90(pieceBinary, i) for i in range(4)] # todo : we could precompute these, dunno how important that is

        possiblePieceStates = [] # todo : okay, we should probably precompute these
        
        print('-----------------rotations-----------------')
        pieceDim = pieceBinary.shape[0]
        for i, rot in enumerate(rotations):
            fourWideRot = np.zeros((4, 4), dtype=int)
            fourWideRot[:pieceDim, :pieceDim] = rot
            possiblePieceStates.append((fourWideRot, i, 0)) # rotation, leftRotationCount, shiftAmount

            leftShifted = fourWideRot.copy()
            rightShifted = fourWideRot.copy()

            shifts = 0
            while np.all(leftShifted[:, 0] == 0):
                leftShifted = np.roll(leftShifted, -1)
                shifts -= 1
                possiblePieceStates.append((leftShifted, i, shifts))
            
            shifts = 0
            while np.all(rightShifted[:, rightShifted[0].size - 1] == 0):
                rightShifted = np.roll(rightShifted, 1)
                shifts += 1
                possiblePieceStates.append((rightShifted, i, shifts))
        
        try:
            grades = [grader(drop(pieceState, boardState)) for pieceState, _, _ in possiblePieceStates]
        except:
            grades = [0]

        bestPlacement = max(enumerate(grades), key=lambda x: (x[1][0], -x[1][1], -x[1][2]))[0]

        # print(possiblePieceStates[bestPlacement])
        state, rotations, shifts = possiblePieceStates[bestPlacement]
        rotationKey = ['', 'a', 'w', 'd'][rotations]

        p.press(rotationKey)
            
        direction = 'right' if shifts > 0 else 'left'
        for _ in range(abs(shifts)):
            p.press(direction)
        
        p.press('space')
            
cv2.destroyAllWindows()