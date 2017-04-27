'''William Menten-Weil wtmenten
CSE 415, Spring 2017, University of Washington
Instructor:  S. Tanimoto.
Assignment 3 Part II.  2
'''

'''EightPuzzle.py
A QUIET Solving Tool problem formulation.
QUIET = Quetzal User Intelligence Enhancing Technology.
The XML-like tags used here serve to identify key sections of this
problem formulation.

CAPITALIZED constructs are generally present in any problem
formulation and therefore need to be spelled exactly the way they are.
Other globals begin with a capital letter but otherwise are lower
case or camel case.
'''
#<METADATA>
QUIET_VERSION = "0.2"
PROBLEM_NAME = "EightPuzzle"
PROBLEM_VERSION = "0.2"
PROBLEM_AUTHORS = ['W. Menten-Weil']
PROBLEM_CREATION_DATE = "17-APR-2017"
PROBLEM_DESC=\
'''This formulation of the Basic Eight Puzzle problem uses generic
Python 3 constructs and has been tested with Python 3.4.
It is designed to work according to the QUIET tools interface, Version 0.2.
'''
#</METADATA>


#<COMMON_DATA>
POSSIBLE_TO = {}
grid_size = 3
for i in range(grid_size):
  for j in range(grid_size):
    cur_tile = i*grid_size+j
    possibilities = []
    if j == 0:
      possibilities.append(cur_tile+1)
    elif j == grid_size-1:
      possibilities.append(cur_tile-1)
    else:
      possibilities.append(cur_tile-1)
      possibilities.append(cur_tile+1)
    if i == 0:
      possibilities.append(cur_tile+grid_size)
    elif i == grid_size-1:
      possibilities.append(cur_tile-grid_size)
    else:
      possibilities.append(cur_tile-grid_size)
      possibilities.append(cur_tile+grid_size)

    POSSIBLE_TO[cur_tile] = possibilities

GOAL_STATE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#</COMMON_DATA>

#<COMMON_CODE>

def can_move(s,From,To):
  '''Tests whether it's legal to move a tile in state s
     from the From tile to the To tile.'''

  try:
   sf=s.d[From] # space from
   st=s.d[To]   # space to
   if sf == 0: return False  # no tile to move.
   if st != 0: return False # dest not empty
   if From == To: return False # not a move
   if To in POSSIBLE_TO[From]: return True
   return False
  except (Exception) as e:
   print(e)

def move(s,From,To):
  '''Assuming it's legal to make the move, this computes
     the new state resulting from swapping From and To tiles'''
  news = s.__copy__() # start with a deep copy.
  news.d[To] = s.d[From]
  news.d[From] = s.d[To]
  return news # return new state

def goal_test(s):
  '''If the state matches the sorted list it is a goal state'''
  return s.d == GOAL_STATE

def goal_message(s):
  return "The Puzzle has been solved!"

class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)

def h_hamming(state):
  "Counts the number of items NOT at their destination."
  count = 0
  for i,v in enumerate(state.d):
    if GOAL_STATE[i] != v: count += 1
  return count

#</COMMON_CODE>

#<STATE>
class State():
  def __init__(self, d):
    self.d = d

  def __str__(self):
    # Produces a brief textual description of a state.
    d = self.d
    txt = ""
    for i in range(3):
      for j in range(3):
        txt += "%s " % d[i*3+j]
      txt += '\n'
    return txt

  def __eq__(self, s2):
    if not (type(self)==type(s2)): return False
    d1 = self.d; d2 = s2.d
    return d1 == d2

  def __hash__(self):
    return (str(self)).__hash__()

  def __copy__(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State({})
    news.d = [t for t in self.d]
    return news
#</STATE>

#<INITIAL_STATE>

INITIAL_STATE = State([1, 4, 2, 3, 7, 0, 6, 8, 5])
CREATE_INITIAL_STATE = lambda: INITIAL_STATE
#</INITIAL_STATE>

#<OPERATORS>
import itertools
tile_combinations = itertools.product(range(9), range(9))

OPERATORS = [Operator("Move tile from %s to %s" % (p,q),
                      lambda s,p1=p,q1=q: can_move(s,p1,q1),
                      # The default value construct is needed
                      # here to capture the values of p&q separately
                      # in each iteration of the list comp. iteration.
                      lambda s,p1=p,q1=q: move(s,p1,q1) )
             for (p,q) in tile_combinations]
#</OPERATORS>

#<GOAL_TEST>
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION>
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>

#<HEURISTICS> (optional)
HEURISTICS = {'h_hamming': h_hamming,}
#</HEURISTICS>
