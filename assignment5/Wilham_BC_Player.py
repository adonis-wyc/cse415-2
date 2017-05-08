'''Wilham_BC_Player.py
Wil-ham, implementation of an agent to play Baroque Chess by William Menten-Weil and Graham Kelly.

'''
from collections import defaultdict
import math
import time
import itertools
MoveTree = defaultdict(list)

# Constants
BLACK = 0
WHITE = 1

INIT_TO_CODE = {'p':2, 'P':3, 'c':4, 'C':5, 'l':6, 'L':7, 'i':8, 'I':9,
  'w':10, 'W':11, 'k':12, 'K':13, 'f':14, 'F':15, '-':0}

CODE_TO_INIT = {0:'-',2:'p',3:'P',4:'c',5:'C',6:'l',7:'L',8:'i',9:'I',
  10:'w',11:'W',12:'k',13:'K',14:'f',15:'F'}

INIT_TO_TEXT = {
    'p': 'Pincer',
    'i': 'Imitator',
    'l': 'Leaper',
    'w': 'Withdrawer',
    'k': 'King',
    'c': 'Coordinator',
    'f': 'Freezer',
    '-': 'empty square on the board'
}

COL_TO_LETTER = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
}

INIT_TO_PIECE_VALUES = {
    'p':lambda s :  1,
    'c':lambda s :  15,
    'l':lambda s :  25,
    'i':lambda s :  50,
    'w':lambda s :  50,
    'k':lambda s :  100,
    'f':lambda s :  15,
    '-':lambda s:  0
}

# End Constants

board_postions = 8**2
max_piece_code = max(INIT_TO_CODE.values()) + 1
# zobristnum = [[0]*max_piece_code]*board_postions
zobristnum= [[[0 for x in range(max_piece_code)] for y in range(board_postions)] for z in range(board_postions)]
    # for j range(max_piece_code):


from random import randint
# import numpy as np
# print(np.array(zobristnum).shape)
# import sys
# sys.exit(0)
def zobinit():
    global zobristnum
    for i in range(board_postions):
        for j in range(board_postions):
            for k in range(max_piece_code):
                zobristnum[i][j][k] = randint(0,  4294967296)
zobinit()

def zhash(state):
    whose = state.whose_move
    board = state.board
    global zobristnum
    val = 0
    for row_index in range(len(board)):
        row = board[row_index]
        for col_index in range(len(row)):
            piece_code = row[col_index]
            # board_pos = row_index*col_index
            # print("Row: %s Col: %s" % (row_index, col_index))
            # print("Board Pos: ", board_pos)
            zob_bp = zobristnum[row_index][col_index]
            zob_num = zob_bp[piece_code]
            val ^= zob_num
    # val ^= whose
    return val


class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf
  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)

valid_coords = list(itertools.product(range(8), range(8))) # board dim is 8
OPERATORS = [
    Operator("move from %s to %s", lambda s, start=start, end=end: can_move(s, start, end), lambda s, start=start, end=end: move(s, start, end))
    for start,end, in itertools.product(valid_coords, valid_coords)
  ]

def makeMove(currentState, currentRemark, timelimit):
    # newMoveDesc = 'No move'
    # return [[newMoveDesc, currentState], newRemark]
    print('Turn at start of makemove %s' % currentState.whose_move)
    # print(currentState)
    root = currentState

    roothash = zhash(root)
    print('Hash of received state %d' % roothash)
    inital_depth = 1
    max_depth = 10
    timelimit = timelimit - 2
    remaining = timelimit
    last_iter_time = time.time()
    max_ply_reached = inital_depth
    for depth in range(inital_depth, max_depth):
        # idfs(root, depth)
        max_ply_reached = depth
        cur_time = time.time()
        elapsed = cur_time - last_iter_time
        last_iter_time = cur_time
        remaining -= elapsed
        timeout(minimax, args=([None,root], depth), kwargs = {'whose':root.whose_move,'times':(cur_time, remaining)},timeout_duration=remaining)
        # if float(remaining) / elapsed < 1.7:
        cur_time = time.time()
        elapsed = cur_time - last_iter_time
        last_iter_time = cur_time
        remaining -= elapsed
        print("Time remaining: %s" % remaining)
        if remaining < 0:
            break
    canidate_state = None
    print("Max ply reached: %s" % max_ply_reached)
    # print([s[0] for s in MoveTree[roothash]])
    if root.whose_move == WHITE:
        canidate_state = max(MoveTree[roothash], key=lambda s: s[0])[1]
    else:
        canidate_state = min(MoveTree[roothash], key=lambda s: s[0])[1]
    newRemark = "Your Move!"
    print('Turn at end of makemove %s' % currentState.whose_move)
    print('Turn at end of makemove for new state %s' % canidate_state.whose_move)
    # print(canidate_state)
    print('Hash of new state %d' % zhash(canidate_state))
    return  [[canidate_state.move_description, canidate_state], newRemark]

def nickname():
    return "Wilham"

def introduce():
    return "I'm Wilham, I was created by William Menten-Weil (wtmenten) and Graham Kelly (grahamtk) to play in a Baroque Chess tournament."

def prepare(player2Nickname):
    return "Hello %s. Let's begin." % player2Nickname
    # pass

# depreciated in favor of minimax with alpha beta
# def idfs(state, cut):
#     if cut == 0: return
#     if zhash(state.board) in MoveTree:
#         for new in MoveTree[State]:
#             idfs(new, cut-1)
#     else:
#         for x, row in enumerate(state.board):
#             for y, col in enumerate(row):
#                 piece = CODE_TO_INIT[col]
#                 # from = (x,y)
#                 ops = OPERATORS
#                 for op in ops:
#                     if op.precond(state):
#                         new = op.state_transf(state)
#                         MoveTree[zhash(state.board)].append(new)
#                         idfs(new, cut-1)

def minimax(state,depth, whose=None, alphabeta=[-1*float('inf'), float('inf')], times=(0, 0)):
    # print(depth)
    # print('time 0 %d time 1 %d curtime %d' % (times[0], times[1], time.time()))
    if depth < 1 or times[0] + times[1] <= time.time(): # this is a leaf node
        cost = staticEval(state[1])
        state[0] = cost
        return cost
    else: # this is not a leaf node
        # print(zhash(state[1]))
        if zhash(state[1]) not in MoveTree:
            for x, row in enumerate(state[1].board):
                for y, col in enumerate(row):
                    piece = CODE_TO_INIT[col]
                    ops = OPERATORS
                    for op in ops:
                        if op.precond(state[1]):
                            new = op.state_transf(state[1])
                            MoveTree[zhash(state[1])].append([staticEval(new),new])

        if whose == WHITE: # Max move
            v = -1*float('inf')
            for ci,child in enumerate(MoveTree[zhash(state[1])]):
                sub_cost = minimax(child, depth-1, whose=BLACK, alphabeta=alphabeta, times=times)
                v = max(v, sub_cost)
                alpha = max(alphabeta[0], v)
                alphabeta[0] = alpha
                if alphabeta[1] <= alphabeta[0]:
                    break
            # print(v)
            MoveTree[zhash(state[1])][ci][0] = v
            return v
        else: # Min move
            v = float('inf')
            for ci,child in enumerate(MoveTree[zhash(state[1])]):
                sub_cost = minimax(child, depth-1, whose=WHITE, alphabeta=alphabeta, times=times)
                v = min(v, sub_cost)
                beta = min(alphabeta[1], v)
                alphabeta[1] = beta
                if alphabeta[1] <= alphabeta[0]:
                    break
            # print(v)
            MoveTree[zhash(state[1])][ci][0] = v

            # state[0] = v
            return v

def staticEval(state):
    value = 0
    me = state.whose_move
    for r, row in enumerate(state.board):
        for c, col in enumerate(row):
            modifier = (who(col) * 2) - 1
            for p in get_adj_pieces(state, r, c):
                if CODE_TO_INIT[p].lower() == 'f':
                    if who(p) != me:
                        modifier *= .75
                    else:
                        modifier *= 1.1
                elif CODE_TO_INIT[p].lower() == 'w':
                    if who(p) != me:
                        modifier *= .2
                elif CODE_TO_INIT[p].lower() == 'k':
                    if who(p) != me:
                        modifier *= .2
                elif CODE_TO_INIT[p].lower() == 'i':
                    if who(p) != me:
                        modifier *= .5
                elif CODE_TO_INIT[p].lower() == 'l':
                    if who(p) != me:
                        modifier *= .6
                elif CODE_TO_INIT[p].lower() == 'p':
                    if who(p) != me:
                        modifier *= .85
            value += modifier * INIT_TO_PIECE_VALUES[CODE_TO_INIT[col].lower()](state)
    return value

def get_adj_pieces(state, r, c):
    adj = []
    pos_deltas = [-1,0,1]
    deltas = itertools.product(pos_deltas, pos_deltas)
    for delta in deltas:
        if delta != [0,0]:
            try:
                piece = state.board[r + delta[0]][c + delta[1]]
                adj.append(piece)
            except IndexError:
                pass
    return adj

def to_piece(state,start):
    code = state.board[start[0]][start[1]]
    init = CODE_TO_INIT[code]
    return code, init

def can_move(state, start, end):
    if start == end: # no move
        return False
    code, piece = to_piece(state, start)
    e_code, e_piece = to_piece(state, end)
    if state.whose_move != who(code):
        return False
    if e_code != 0 and state.whose_move == who(e_code): #can't move into space occupied by ur own piece.
        return False
    if is_frozen(state, start) or (piece.lower() != 'i' and is_frozen(state, end)):
        return False
    if is_coordinated(state, end):
        return False
    piece = piece.lower()
    return can_move_piece(state, start, end, piece=piece)

def is_coordinated(state, loc):
    me = state.whose_move
    for i,vi in enumerate(state.board):
        for j, vj in enumerate(vi):
            if who(vj) != me and vj in [13,14]: k_idx = [i, j]
            if who(vj) != me and vj in [4,5]: c_idx = [i, j]

    dr = k_idx[0] - c_idx[0]
    dc = k_idx[1] - c_idx[1]
    if (k_idx[0] - dr,k_idx[1]) == loc or (k_idx[0],k_idx[1] - dc) == loc:
        return True
    return False

def is_frozen(state, start):
    surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    piece_who = who(state.board[start[0]][start[1]])
    for dr, dc in surrounding_space_deltas:
        if dr == dc and dr == 0:
            continue
        try:
            adj_x = start[0] + dr
            adj_y = start[1] + dc
            # print("Withdrawing from %s, %s" % (withdrawn_x,withdrawn_y))
            if adj_x >= 0 and adj_y >= 0:
                adj_code = state.board[start[0] + dr][start[1] + dc]
                if CODE_TO_INIT[adj_code].lower() == 'f' and who(adj_code) != piece_who:
                    return True
                elif CODE_TO_INIT[adj_code].lower() == 'i' and who(adj_code) != piece_who:
                    adj_loc = [start[0] + dr,start[1] + dc]
                    return is_frozen(state,adj_loc)
        except IndexError:
            pass
    return False

def can_move_piece(state, start, end, piece='-'):
    if piece == '-': # starting piece is blank
        return False
    elif piece == 'p':
        return can_move_pawn(state, start, end)
    elif piece == 'k':
        return can_move_king(state, start, end)
    elif piece in ['w', 'c', 'f']:
        return can_move_noble(state, start, end)
    elif piece == 'l':
        return can_move_leaper(state, start, end)
    elif piece == 'i':
        return can_move_imitator(state, start, end)
    else:
        raise ValueError("Unkown Piece %s" % piece)

def can_move_linear(state, start, end, jumps=0):
    x_dif = end[0] - start[0]
    y_dif = end[1] - start[1]

    endpiece = state.board[end[0]][end[1]]
    if CODE_TO_INIT[endpiece] != '-':
            return False

    if start[0] == end[0]: # same along the 1st dim
        direction = y_dif / abs(y_dif)
        row = state.board[start[0]]
        for col_dist in range(1,abs(y_dif)):
            col_index = start[1] + col_dist*direction
            col_index = int(col_index)
            col = row[col_index]
            if CODE_TO_INIT[col] != '-':
                if jumps > 0 and state.whose_move != who(col):
                    jumps -= 1
                else:
                    return False
        return True
    elif start[1] == end[1]: # same along the 2nd dim
        direction = x_dif / abs(x_dif)
        for row_dist in range(1,abs(x_dif)):
            row_index = start[0] + row_dist*direction
            row_index = int(row_index)
            row = state.board[row_index]
            col = row[start[1]]
            if CODE_TO_INIT[col] != '-':
                if jumps > 0 and state.whose_move != who(col):
                    jumps -= 1
                else:
                    return False
        return True

def can_move_pawn(state, start, end):
    x_dif = start[0] - end[0]
    y_dif = start[1] - end[1]

    # if move is on two axes
    if x_dif != 0 and y_dif != 0:
        return False
    return can_move_linear(state, start, end)


def can_move_king(state, start, end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if x_dif > 1 or y_dif > 1:
        return False

    # if CODE_TO_INIT[state.board[end[0]][end[1]]] != '-':
    #     return False
    # else:
    return True

def can_move_diag(state, start, end, jumps=0):
    x_dif = end[0] - start[0]
    y_dif = end[1] - start[1]
    # if not diag
    if abs(x_dif) != abs(y_dif):
        return False

    row_direction = x_dif / abs(x_dif)
    col_direction = y_dif / abs(y_dif)

    diag_dist = range(1,abs(x_dif)+1) # there the same
    # col_dist = range(1,abs(y_dif)+1)
    endpiece = state.board[end[0]][end[1]]
    if CODE_TO_INIT[endpiece] != '-':
            return False

    for diag_dist in range(1,abs(x_dif)):
        row_index = int(start[0] + diag_dist*row_direction)
        col_index = int(start[1] + diag_dist*col_direction)
        cell = state.board[row_index][col_index]
        if CODE_TO_INIT[cell] != '-':
            if jumps > 0 and state.whose_move != who(cell):
                jumps -= 1
            else:
                return False
    return True

def can_move_leaper(state,start,end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if can_move_linear(state, start, end, jumps=1) or can_move_diag(state, start, end, jumps=0):
        return True
    else:
        return False

def can_move_imitator(state,start,end):
    surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    can_moves = []
    for x_delt, y_delt in surrounding_space_deltas:
        if x_delt == y_delt and x_delt == 0: # to prevent infinite loop in check imitators own space (no imitation) as a noble
            can_moves.append(can_move_noble(state,start,end))
        adj_x, adj_y = start[0] +x_delt, start[1]+y_delt
        try:
            piece = state.board[adj_x][adj_y]
            adj_p = CODE_TO_INIT[piece]
            if adj_p.lower() == 'i':continue
            if who(piece) != state.whose_move:
                can_moves.append(can_move_piece(state,start,end,piece=adj_p.lower()))
        except IndexError:
            pass
    return any(can_moves)

def can_move_noble(state, start, end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if can_move_linear(state, start, end) or can_move_diag(state, start, end):
        return True
    else:
        return False

# TODO for imitator Move pick move which removes the best pieces
# TODO check for frozen pieces in can_move

def move(state, start, end):
    new_state = BC_state(old_board=state.board, whose_move=state.whose_move)
    piece_code = new_state.board[start[0]][start[1]]
    piece = CODE_TO_INIT[piece_code].lower()
    piece_text = INIT_TO_TEXT[piece]
    s_col = COL_TO_LETTER[start[1]]
    s_row = start[0]+1
    e_col = COL_TO_LETTER[end[1]]
    e_row = end[0]+1
    new_state.move_description = '%s from %s%s to %s%s' % (piece_text, s_col, s_row, e_col, e_row)

    new_state = move_piece(new_state, start, end, piece=piece)
    new_state.board[start[0]][start[1]] = 0
    new_state.board[end[0]][end[1]] = piece_code
    new_state.whose_move = (new_state.whose_move + 1) % 2
    return new_state


def move_piece(new_state, start, end, piece='-'):
    if piece == 'p':
        new_state = move_pawn(new_state, start, end)
    elif piece == 'k':
        new_state = move_king(new_state, start, end)
    elif piece == 'w':
        new_state = move_withdrawer(new_state, start, end)
    elif piece == 'c':
        new_state = move_coordinator(new_state, start, end)
    elif piece == 'f':
        new_state = move_freezer(new_state, start, end)
    elif piece == 'l':
        new_state = move_leaper(new_state, start, end)
    elif piece == 'i':
        new_state = move_imitator(new_state, start, end)
    else:
        raise ValueError("Unkown Piece %s" % piece)
    return new_state

#TODO fix for imitator. DONE
def move_pawn(state, start, end):
    whos_piece = who(state.board[start[0]][start[1]])
    piece_code = 3 if whos_piece == 1 else 2
    me = state.whose_move
    # piece_code = state.board[end[0]][end[1]]
    pos_deltas = [-2,2]
    for index in range(2):
        for delt in pos_deltas:
            init_delta = [0,0]
            init_delta[index] = delt
            try:
                if state.board[end[0] + init_delta[0]][end[1] + init_delta[1]] == piece_code and\
                who(state.board[end[0] + (init_delta[0]//2)][end[1] + (init_delta[1]//2)]) != me:
                    state.board[end[0] + (init_delta[0]//2)][end[1] + (init_delta[1]//2)] = 0
            except IndexError:
                pass
    return state

def move_king(state, start, end):
    return state

def move_leaper(state, start, end):
    if start[0] == end[0]:
        for i in range(start[1], end[1], 1 if end[1] > start[1] else -1):
            state.board[start[0]][i] = 0
    elif start[1] == end[1]:
        for i in range(start[0], end[0], 1 if end[0] > start[0] else -1):
            state.board[i][start[1]] = 0
    else:
        for x, y in zip(range(start[0], end[0], 1 if end[0] > start[0] else -1), range(start[1], end[1], 1 if end[1] > start[1] else -1)):
            state.board[x][y] = 0
    return state

def move_withdrawer(state, start, end):
    r_dif = end[0] - start[0]
    c_dif = end[1] - start[1]
    r_dir = int((r_dif * -1) / r_dif) if r_dif != 0 else 0
    c_dir = int((c_dif * -1) / c_dif) if c_dif != 0 else 0
    try:
        withdrawn_x = start[0] + r_dir
        withdrawn_y = start[1] + c_dir
        # print("Withdrawing from %s, %s" % (withdrawn_x,withdrawn_y))
        if withdrawn_x >= 0 and withdrawn_y >= 0:
            piece = state.board[withdrawn_x][withdrawn_y]
            if who(piece) != state.whose_move:
                state.board[start[0] + r_dir][start[1] + c_dir] = 0
    except IndexError:
        pass
    # surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    # for dr, dc in surrounding_space_deltas:
    #     try:
    #         state.board[start[0] + dr][start[1] + dc] = 0
    #     except IndexError:
    #         pass
    return state

def move_coordinator(state, start, end):
    me = state.whose_move
    for i,vi in enumerate(state.board):
        for j, vj in enumerate(vi):
            if who(vj) == me and vj in [13,14]: k_idx = [i, j]
    dr = k_idx[0] - end[0]
    dc = k_idx[1] - end[1]

    if who(state.board[k_idx[0] - dr][k_idx[1] ]) != me:
        state.board[k_idx[0] - dr][k_idx[1] ] = 0
    if who(state.board[k_idx[0]][k_idx[1] - dc]) != me:
        state.board[k_idx[0]][k_idx[1] - dc]
    return state

def move_freezer(state, start, end):
    return state

def move_imitator(state, start, end):
    value_iter = sorted(INIT_TO_PIECE_VALUES.items(), key=lambda x: x[1](state), reverse=True)
    for k, v in value_iter:
        if k in ['i', '-', 'f']:
            continue
        if can_move_piece(state, start, end, piece=k):
            # print("moving imitator like %s" % k)
            return move_piece(state, start, end, piece=k)
    return state

def who(piece): return piece % 2

def parse(bs): # bs is board string
  '''Translate a board string into the list of lists representation.'''
  b = [[0,0,0,0,0,0,0,0] for r in range(8)]
  rs9 = bs.split("\n")
  rs8 = rs9[1:] # eliminate the empty first item.
  for iy in range(8):
    rss = rs8[iy].split(' ');
    for jx in range(8):
      b[iy][jx] = INIT_TO_CODE[rss[jx]]
  return b

INITIAL = parse('''
c l i w k i l f
p p p p p p p p
- - - - - - - -
- - - - - - - -
- - - - - - - -
- - - - - - - -
P P P P P P P P
F L I W K I L C
''')


class BC_state:
  def __init__(self, old_board=INITIAL, whose_move=WHITE):
    new_board = [r[:] for r in old_board]
    self.board = new_board
    self.whose_move = whose_move

  def __repr__(self):
    s = ''
    for r in range(8):
      for c in range(8):
        s += CODE_TO_INIT[self.board[r][c]] + " "
      s += "\n"
    if self.whose_move==WHITE: s += "WHITE's move"
    else: s += "BLACK's move"
    s += "\n"
    return s

def test_starting_board():
  init_state = BC_state(INITIAL, WHITE)
  print(init_state)

# test_starting_board()





import sys
import time
from traceback import print_exc
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''This function will spawn a thread and run the given function using the args, kwargs and
    return the given default value if the timeout_duration is exceeded
    '''
    import threading
    class PlayerThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                print("Seems there was a problem with the time.")
                print_exc()
                self.result = default

    pt = PlayerThread()
    pt.start()
    started_at = time.time()
    pt.join(timeout_duration)
    ended_at = time.time()
    diff = ended_at - started_at
    print("Time used in minimax: %0.4f seconds out of " % diff, timeout_duration)
    if pt.isAlive():
        print("Took too long. to do next ply")
        # print("We are now terminating the game.")
        # print("Player "+PLAYER_MAP[CURRENT_PLAYER]+" loses.")
        # if USE_HTML: gameToHTML.reportResult("Player "+PLAYER_MAP[CURRENT_PLAYER]+" took too long (%04f seconds) and thus loses." % diff)
        # if USE_HTML: gameToHTML.endHTML()
        # exit()
    else:
        print("Within the time limit -- nice!")
        # return pt.result
