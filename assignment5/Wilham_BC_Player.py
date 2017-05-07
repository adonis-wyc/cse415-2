'''Wilham_BC_Player.py
Wil-ham, implementation of an agent to play Baroque Chess by William Menten-Weil and Graham Kelly.

'''
# import defaultdict
import math
import time
MoveTree = defaultdict(list)


def makeMove(currentState, currentRemark, timelimit):
    # newMoveDesc = 'No move'
    # return [[newMoveDesc, currentState], newRemark]
    root = currentState

    inital_cut = 3
    max_ply = 8
    remaining = timelimit
    last_iter_time = time.time()
    for cut in range(inital_cut, max_ply)
        idfs(root, cut)
        minimax(root)
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
    return "Wil-ham"

def introduce():
    return "I'm Wil-ham, I was created by William Menten-Weil (wtmenten) and Graham Kelly (grahamtk) to play in a Baroque Chess tournament."

def prepare(player2Nickname):
    pass

def idfs(state, cut):
    if cut == 0: return

    if state in MoveTree:
        for new in MoveTree[State]:
            # TODO add terminator for if a b pruning test is failed
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

def minimax(state, alphabeta=[-math.inf, math.inf]):
    if zhash(state.board) not in MoveTree: # this is a leaf node
        cost = staticEval(state)
        state.heur_val = cost
        return cost
    else: # this is not a leaf node
        if state.whose_move == WHITE: # Max move
            canidate = -math.inf
            for child in MoveTree[zhash(state.board)]:
                sub_cost = minimax(child, alphabeta)
                canidate = max(canidate, minimax(child, alphabeta))
                alpha = max(alphabeta[0], canidate)
                if alphabeta[1] <= alphabeta[0]:
                    break
            state.heur_val = v
            return v
        else: # Min move
            canidate = math.inf
            for child in MoveTree[zhash(state.board)]:
                sub_cost = minimax(child, alphabeta)
                canidate = min(canidate, minimax(child, alphabeta))
                beta = min(alphabeta[1], canidate)
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
    piece = piece.lower()
    return can_move_piece(state, start, end, piece=piece)


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
                if jumps > 0:
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

    if CODE_TO_INIT[state.board[end[0]][end[1]]] != '-':
        return False
    else:
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
            if jumps > 0:
                jumps -= 1
            else:
                return False


    return True

def can_move_leaper(state,start,end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if can_move_linear(state, start, end, jumps=1) or can_move_diag(state, start, end, jumps=1):
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
        adj_p = CODE_TO_INIT[state.board[adj_x][adj_y]]
        can_moves.append(can_move_piece(state,start,end,piece=adj_p))
    return any(can_moves)

def can_move_noble(state, start, end):
    x_dif = abs(start[0] - end[0])
    y_dif = abs(start[1] - end[1])

    if can_move_linear(state, start, end) or can_move_diag(state, start, end):
        return True
    else
        return False

# TODO for imitator Move pick move which removes the best pieces

def move(state, start, end):
    new_state = state.__copy__()
    new_state.whose_move = (new_state.whose_move + 1) % 2
    piece = CODE_TO_INIT[new_state.board[start[0]][start[1]]]
    piece = INIT_TO_TEXT[piece]
    s_col = COL_TO_LETTER[start[1]]
    s_row = start[0]+1
    e_col = COL_TO_LETTER[end[1]]
    e_row = end[0]+1
    new_state.move_description = '%s from %s%s to %s%s' % (piece, s_col, s_row, e_col, e_row)

    # TODO build the move router here

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



