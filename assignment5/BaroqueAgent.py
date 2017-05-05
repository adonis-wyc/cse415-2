'''BaroqueAgent.py
Wil-ham, implementation of an agent that can't play
Baroque Chess.

'''
# import defaultdict

MoveTree = defaultdict(list)


def makeMove(currentState, currentRemark, timelimit):
    # newMoveDesc = 'No move'
    # newRemark = "I don't even know how to move!"
    # return [[newMoveDesc, currentState], newRemark]
    root = currentState

    inital_cut = 3
    max_ply = 8
    for cut in range(inital_cut, max_ply)
        idfs(root, cut)
        minimax(root)
        # TODO add logic to see if we have enough time for another iteration

def nickname():
    return "Wil-ham"

def introduce():
    return "I'm Wil-ham, I was created by William Menten-Weil and Graham Kelly to play in a Baroque Chess tournament."

def prepare(player2Nickname):
    pass

def idfs(state, cut):
    if cut == 0: return

    if state in MoveTree:
        for new in MoveTree[State]:
            # TODO add terminator for if a b pruning test is failed
            idfs(new, cut-1)
    else:
        for row in state:
            for col in state:
                piece = CODE_TO_INIT[col]
                ops = INIT_TO_OPERATORS[piece.lower()]
                for op in ops:
                    if op.precond(state):
                        new = op.state_trans(state)
                        MoveTree[state].append([None, new])
                        idfs(new, cut-1)

def minimax(state):
    for child in MoveTree[state]:
        if child[1] in MoveTree:
            sub_cost = minimax(child[1])
            # TODO add A B pruning logic here
            child[0] = sub_cost
        else:
            cost = static_evaluate(child[1])
            child[0] = cost
            # TODO add A B pruning logic here
            return cost




BLACK = 0
WHITE = 1

INIT_TO_CODE = {'p':2, 'P':3, 'c':4, 'C':5, 'l':6, 'L':7, 'i':8, 'I':9,
  'w':10, 'W':11, 'k':12, 'K':13, 'f':14, 'F':15, '-':0}

CODE_TO_INIT = {0:'-',2:'p',3:'P',4:'c',5:'C',6:'l',7:'L',8:'i',9:'I',
  10:'w',11:'W',12:'k',13:'K',14:'f',15:'F'}

INIT_TO_OPERATORS {
    'p': ,
    'c': ,
    'l': ,
    'i': ,
    'w': ,
    'k': ,
    'f': ,
    '-': [],
}

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
    self.whose_move = whose_move;

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




