'''Wilham_BC_Player.py
Wil-ham, implementation of an agent to play Baroque Chess by William Menten-Weil and Graham Kelly.

'''
import math
import time
MoveTree = defaultdict(list)


def makeMove(currentState, currentRemark, timelimit):
    # newMoveDesc = 'No move'
    # return [[newMoveDesc, currentState], newRemark]
    root = currentState

    inital_depth = 1
    max_depth = 10
    remaining = timelimit
    last_iter_time = time.time()
    for depth in range(inital_depth, max_depth)
        # idfs(root, depth)
        minimax(root, depth)
        cur_time = time.time()
        elapsed = cur_time - last_iter_time
        last_iter_time = cur_time
        remaining -= elapsed
        if float(remaining) / elapsed < 1:
            canidate_state = None
            if state.whose_move == WHITE:
                canidate_state = max(MoveTree[root], key=lambda s: s.heur_val)
            else:
                canidate_state = min(MoveTree[root], key=lambda s: s.heur_val)
            newRemark = "Your Move!"
            return  [[canidate_state.move_description, canidate_state], newRemark]

def nickname():
    return "Wilham"

def introduce():
    return "I'm Wilham, I was created by William Menten-Weil (wtmenten) and Graham Kelly (grahamtk) to play in a Baroque Chess tournament."

def prepare(player2Nickname):
    return "Hello %s. Let's begin." % player2Nickname
    # pass

# depreciated in favor of minimax with alpha beta
def idfs(state, cut):
    if cut == 0: return
    if zhash(state.board) in MoveTree:
        for new in MoveTree[State]:
            idfs(new, cut-1)
    else:
        for x, row in enumerate(state):
            for y, col in enumerate(state):
                piece = CODE_TO_INIT[col]
                from = (x,y)
                ops = INIT_TO_OPERATORS[piece.lower()]
                for op in ops:
                    if op.precond(state, ):
                        new = op.state_trans(state)
                        MoveTree[zhash(state.board)].append(new)
                        idfs(new, cut-1)

def minimax(state,depth, alphabeta=[-math.inf, math.inf]):
    if zhash(state.board) not in MoveTree: # this is a leaf node
        cost = staticEval(state)
        state.heur_val = cost
        return cost
    else: # this is not a leaf node
        if state.whose_move == WHITE: # Max move
            v = -math.inf
            for child in MoveTree[zhash(state.board)]:
                sub_cost = minimax(child, depth, alphabeta)
                v = max(v, minimax(child, alphabeta))
                alpha = max(alphabeta[0], v)
                alphabeta[0] = alpha
                if alphabeta[1] <= alphabeta[0]:
                    break
            state.heur_val = v
            return v
        else: # Min move
            v = math.inf
            for child in MoveTree[zhash(state.board)]:
                sub_cost = minimax(child, depth, alphabeta)
                v = min(v, minimax(child, alphabeta))
                beta = min(alphabeta[1], v)
                alphabeta[1] = beta
                if alphabeta[1] <= alphabeta[0]:
                    break
            state.heur_val = v
            return v


def staticEval(state):
    value = 0
    for row in state.board:
        for col in row:
            modifer = (who(col) * 2) - 1
            value += modifer * INIT_TO_PIECE_VALUES[CODE_TO_INIT[col].lower()](state)
    return value




class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)

board_postions = 8**2
max_piece_code = max(INIT_TO_CODE.values())
zobristnum = [[0]*board_postions]*max_piece_code
from random import randint

def myinit():
    global zobristnum
    for i in range(board_postions):
        for j in range(max_piece_code):
            zobristnum[i][j]=\
            randint(0, \
            4294967296)

def zhash(board):
    global zobristnum
    val = 0;
    for row_index in range(board):
        row = board[row_index]
        for col_index in range(row):
            piece_code = row[col_index]
            board_pos = row_index*col_index
            val ^= zobristnum[board_pos][piece_code]
    return val

BLACK = 0
WHITE = 1

INIT_TO_CODE = {'p':2, 'P':3, 'c':4, 'C':5, 'l':6, 'L':7, 'i':8, 'I':9,
  'w':10, 'W':11, 'k':12, 'K':13, 'f':14, 'F':15, '-':0}

CODE_TO_INIT = {0:'-',2:'p',3:'P',4:'c',5:'C',6:'l',7:'L',8:'i',9:'I',
  10:'w',11:'W',12:'k',13:'K',14:'f',15:'F'}

INIT_TO_TEXT = {
    'p': 'Pincer'
    'l': 'Leaper'
    'i': 'Imitator'
    'w': 'Withdrawer'
    'k': 'King'
    'c': 'Coordinator'
    'f': 'Freezer'
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
    'p':lambda s : return 1,
    'c':lambda s : return 15,
    'l':lambda s : return 25,
    'i':lambda s : return 50,
    'w':lambda s : return 50,
    'k':lambda s : return 100,
    'f':lambda s : return 15,
    '-':lambda s: return 0
}

def to_piece(state,start):
    return CODE_TO_INIT[state.board[start[0]][start[1]]]

def can_move(state, start, end):
    if start == end: # no move
        return False
    piece = to_piece(state, start)
    if state.whose_move != who(piece) :
        return False
    if is_frozen(state, start):
        return False
    piece = piece.lower()
    return can_move_piece(state, start, end, piece=piece)

def is_frozen(state, start):

    surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    for dr, dc in surrounding_space_deltas:
        try:
            adj_code = state.board[start[0] + dx][start[1] + dy]
            if INIT_TO_CODE[adj_code].lower() == 'f' and who(adj_code) != state.whose_move:
                return True
            elif INIT_TO_CODE[adj_code].lower() == 'i' and who(adj_code) != state.whose_move:
                adj_loc = [start[0] + dx,start[1] + dy]
                return is_frozen(state,adj_loc)
        except KeyError:
            pass
    return False


def can_move_piece(state, start, end, piece='-'):
    if piece == '-' # starting piece is blank
        return False
    elif piece == 'p'
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
    x_dif = start[0] - end[0]
    y_dif = start[1] - end[1]

    endpiece = state.board[end[0]][end[1]]
    if CODE_TO_INIT[endpiece] != '-':
            return False

    if start[0] == end[0]: # same along the 1st dim
        direction = y_dif / abs(y_dif)
        row = state.board[start[0]]
        for col_dist in range(1,abs(y_dif)+1):
            col_index = start[1] + col_dist*direction
            col = row[col_index]
            if CODE_TO_INIT[col] != '-':
                if jumps > 0:
                    jumps -= 1
                else:
                    return False
        return True
    elif start[1] == end[1]: # same along the 2nd dim
        direction = x_dif / abs(x_dif)
        for row_dist in range(1,abs(x_dif)+1):
            row_index = start[0] + row_dist*direction
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
    x_dif = start[0] - end[0]
    y_dif = start[1] - end[1]
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

    for diag_dist in range(1,abs(x_dif)+1):
        row_index = start[0] + row_dist*x_dif
        col_index = start[1] + row_dist*y_dif
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
    else
        return False

def can_move_imitator(state,start,end):
    surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    can_moves = []
    for x_delt, y_delt in surrounding_space_deltas:
        if x_delt == y_delt and x_delt == 0: # to prevent infinite loop in check imitators own space (no imitation) as a noble
            can_moves.append(can_move_noble(state,start,end))
        adj_x, adj_y = start[0] +x_delt, start[1]+y_delt
        try:
            adj_p = CODE_TO_INIT[state.board[adj_x][adj_y]]
            can_moves.append(can_move_piece(state,start,end,piece=adj_p))
        except KeyError:
            pass
    return any(can_moves)

def can_move_noble(state, start, end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if can_move_linear(state, start, end) or can_move_diag(state, start, end):
        return True
    else
        return False

# TODO for imitator Move pick move which removes the best pieces
# TODO check for frozen pieces in can_move

def move(state, start, end)
    new_state = state.__copy__()
    piece_code = new_state.board[start[0]][start[1]]
    piece = CODE_TO_INIT[piece_code]
    piece_text = INIT_TO_TEXT[piece]
    s_col = COL_TO_LETTER[start[1]]
    s_row = start[0]+1
    e_col = COL_TO_LETTER[end[1]]
    e_row = end[0]+1
    new_state.move_description = '%s from %s%s to %s%s' % (piece_text, s_col, s_row, e_col, e_row)

    if piece == 'p'
        new_state = move_pawn(new_state, start, end)
    elif piece == 'k':
        # new_state = move_king(new_state, start, end)
        pass
    elif piece in ['w', 'c', 'f']: # TODO break these apart
        new_state = move_noble(new_state, start, end)
    elif piece == 'l':
        new_state = move_leaper(new_state, start, end)
    elif piece == 'i':
        new_state = move_imitator(new_state, start, end)
    else:
        raise ValueError("Unkown Piece %s" % piece)

     new_state.board[start[0]][start[1]] = 0
     new_state.board[end[0]][end[1]] = piece_code
     new_state.whose_move = (new_state.whose_move + 1) % 2


    #TODO fix for imitator. DONE
def move_pawn(state, start, end):
    whos_piece = whos(state.board[start[0]][start[1]])
    piece_code = 3 if whos_piece == 1 else piece_code = 2
    # piece_code = state.board[end[0]][end[1]]
    pos_deltas = [-2,2]
    for index in range(2):
        for delt in pos_deltas:
            init_delta = [0,0]
            init_delta[index] = pos_deltas
            try:
                if state.board[end[0] + init_delta[0]][end[1] + init_delta[1]] == piece_code:
                    state.board[end[0] + (init_delta[0]/2)][end[1] + (init_delta[1]/2)] = 0
            except KeyError:
                pass
    return state

def move_king(state, start, end):
    pass

def move_leaper(state, start, end):
    if start[0] == end[0]:
        for i in range(start[1], end[1], 1 if end[1] > start[1] else -1):
            state.board[start[0]][i] = 0
    elif start[1] == end[1]:
        for i in range(start[0], end[0], 1 if end[0] > start[0] else -1):
            state.board[i][start[1]] = 0
    else:
        for x, y in zip(start[0], end[0], 1 if end[0] > start[0] else -1, range(start[1], end[1], 1 if end[1] > start[1] else -1)):
            state.board[x][y] = 0

def move_withdrawer(state, start, end):
    surrounding_space_deltas = itertools.product([-1,0,1],[-1,0,1])
    for dr, dc in surrounding_space_deltas:
        try:
            state.board[start[0] + dx][start[1] + dy] = 0
        except KeyError:
            pass

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

def move_freezer(state, start, end):
    pass

def move_imitator(state, start, end):


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

test_starting_board()


Operators = [
    Operator("move from %s to %s", lambda s, start=start, end=end: can_move(s, start, end), lambda s, start=start, end=end: move(s, start, end))
    for start,end, in itertools.product(range(8), range(8)) # board dim is 8
  ]



