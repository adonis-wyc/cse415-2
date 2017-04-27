'''William Menten-Weil wtmenten & Graham Kelly grahamtk
CSE 415, Spring 2017, University of Washington
Instructor:  S. Tanimoto.
Assignment 4
'''
'''EndPoverty.py
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
PROBLEM_NAME = "EndPoverty"
PROBLEM_VERSION = "1"
PROBLEM_AUTHORS = ['W. Menten-Weil', 'G. Kelly']
PROBLEM_CREATION_DATE = "26-APR-2017"
PROBLEM_DESC=\
'''This formulation of the End Poverty problem uses generic
Python 3 constructs and has been tested with Python 3.4.
It is designed to work according to the QUIET tools interface, Version 1.
'''
#</METADATA>
import copy
import math
import itertools

#<COMMON_DATA>
POVERTY_LEVEL = 35000
MIN_WAGE = 27000
INITIAL_POP_SIZE = 1000
INFLATION = 0.018
#</COMMON_DATA>

#<COMMON_CODE>

def can_adj_tax(s,i,delta):
  '''Tests whether it's legal to
    adjust the tax in state = s
    for bracket index = i
    by value = delta'''
  b_value_index = 1

  # will the cutoff/taxrate stay above 0?
  if s.b[i][b_value_index] + delta <= 0:
    return False # tax rate cannot fall below 0

  # now test if shifting by delta will invalidate the order of the brackets
  target_b = s.b[i][b_value_index] + delta
  if i == 0:
    post_b = s.b[i+1][b_value_index]
    return target_b < post_b
  elif i == len(s.b) - 1:
    prev_b = s.b[i-1][b_value_index]
    return prev_b < target_b
  else:
    prev_b = s.b[i-1][b_value_index]
    post_b = s.b[i+1][b_value_index]
    return prev_b < target_b < post_b

def can_adj_cutoff(s,i,delta):
  '''Tests whether it's legal to
    adjust the cuttoff in state = s
    for bracket index = i
    by value = delta'''
  b_value_index = 0

  if s.b[i][b_value_index] + delta <= 0:
    return False # cutoff cannot fall below 0

  target_b = s.b[i][b_value_index] + delta

  if i == 0:
    post_b = s.b[i+1][b_value_index]
    return target_b < post_b
  elif i == len(s.b) - 1:
    prev_b = s.b[i-1][b_value_index]
    # return prev_b < target_b
    return False # last tax brackets cutoff must be +inf
  else:
    prev_b = s.b[i-1][b_value_index]
    post_b = s.b[i+1][b_value_index]
    return prev_b < target_b < post_b

def adj_tax(s, i, delta):
  new_s = s.__copy__()
  print(new_s.p[0])
  new_s.b[i][1] += delta
  return advance(new_s)

def adj_cutoff(s, i, delta):
  new_s = s.__copy__()
  new_s.b[i][0] += delta
  return advance(new_s)

# progresses a year in the state
def advance(s):
  s = s.__copy__()
  taxes = 0
  for i, p in enumerate(s.p):
    unfound = True
    current_bracket = 0
    while unfound:
      cut, rate = s.b[current_bracket]
      if p <= cut:
        taxes += p * rate
        s.p[i] *= (1 + INFLATION)
        s.p[i] *= (1 -  rate)
        unfound = False
      else:
        current_bracket += 1

  per_p_subsidy = taxes / len(s.p)

  # print("Taxes: %s" % taxes)
  # print("Tax Return Per P: %s" % per_p_subsidy)
  for i, p in enumerate(s.p):
      s.p[i] += per_p_subsidy
  return s


def goal_test(s):
  '''If the state matches the sorted list it is a goal state'''
  return all(p >= POVERTY_LEVEL for p in s.p)

def goal_message(s):
  return "Poverty has ended!"

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
  "Counts the number of people below the poverty line."
  count = 0
  for i,p in enumerate(state.p):
    if p < POVERTY_LEVEL: count += 1

  return count + bracket_tax_dif(state)

def bracket_tax_dif(s):
  b_val_idx = 1
  difs = []
  for i in range(len(s.b)-1):
    difs.append(s.b[i+1][b_val_idx] - s.b[i][b_val_idx])
  return sum(difs) / float(len(difs))

# def h_euclidean(state):
#   import math
#   distances = []
#   for i,v in enumerate(state.d):
#     g_col = v % 3
#     g_row = v // 3 % 3
#     c_col = i % 3
#     c_row = i // 3 % 3
#     dist = math.sqrt((c_col-g_col)**2 + (c_row-g_row)**2)
#     distances.append(dist)
#   return sum(distances)

# def h_manhattan(state):
#   distances = []
#   for i,v in enumerate(state.d):
#     g_col = v % 3
#     g_row = v // 3 % 3
#     c_col = i % 3
#     c_row = i // 3 % 3
#     dist = abs(c_col-g_col) + abs(c_row-g_row)
#     distances.append(dist)
#   return sum(distances)

# def h_harmonic(state):
#   h1 = h_manhattan(state)
#   h2 = h_euclidean(state)
#   h3 = h_hamming(state)
#   h = 0 if h1 == 0 or h2 == 0 or h3 == 0 else 3 / (1/h1 + 1/h2 + 1/h3)
#   return h

#</COMMON_CODE>

#<STATE>
class State():
  def __init__(self, p, b):
    self.p = copy.copy(p)
    self.b = copy.copy(b)

  def __str__(self):
    # Produces a brief textual description of a state.
    return "People: %s\nTax Brackets: %s" % (self.p, self.b)

  def __eq__(self, s2):
    if not (type(self)==type(s2)): return False
    pop1, tax1 = self.p, self.b; pop2, tax2 = s2.p, s2.b
    return pop1 == pop2 and tax1 == tax2

  def __hash__(self):
    return (str(self)).__hash__()

  def __copy__(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State([],[])
    news.p, news.b = [copy.deepcopy(t) for t in self.p], [copy.deepcopy(t) for t in self.b]
    return news
#</STATE>

#<INITIAL_STATE>

INITAL_POPULATION = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 10000, 10000, 12000, 13000, 50000, 55000, 63000, 72000]
INITAL_TAX_BRACKETS = [[11000, .0], [20000, .10], [35000, .11], [50000, .12], [75000, .15], [100000, .16], [math.inf, .17]]
INITIAL_STATE = State(
    INITAL_POPULATION,
    INITAL_TAX_BRACKETS
  )
CREATE_INITIAL_STATE = lambda: INITIAL_STATE
#</INITIAL_STATE>

#<OPERATORS>

OPERATORS = [
  [
    Operator("Increase Cutoff", lambda s: can_adj_cutoff(s, i, 1000), lambda s: adj_cutoff(s,i, 1000))
    for i in range(len(INITAL_TAX_BRACKETS))
  ],
  [
    Operator("Decrease Cutoff", lambda s: can_adj_cutoff(s, i, -1000), lambda s: adj_cutoff(s,i, -1000))
    for i in range(len(INITAL_TAX_BRACKETS))
  ],
  [
    Operator("Increase Tax Rate", lambda s: can_adj_tax(s, i, 0.01), lambda s: adj_tax(s,i, 0.01))
    for i in range(len(INITAL_TAX_BRACKETS))
  ],
  [
    Operator("Decrease Tax Rate", lambda s: can_adj_tax(s, i, -0.01), lambda s: adj_tax(s,i, -0.01))
    for i in range(len(INITAL_TAX_BRACKETS))
  ],
]
OPERATORS = list(itertools.chain(*OPERATORS))

# tile_combinations = itertools.product(range(9), range(9))
# OPERATORS = [Operator("Move tile from %s to %s" % (p,q),
#                       lambda s,p1=p,q1=q: can_move(s,p1,q1),
#                       # The default value construct is needed
#                       # here to capture the values of p&q separately
#                       # in each iteration of the list comp. iteration.
#                       lambda s,p1=p,q1=q: move(s,p1,q1) )
#              for (p,q) in tile_combinations]
#</OPERATORS>

#<GOAL_TEST>
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION>
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>

#<HEURISTICS> (optional)
HEURISTICS = {'h_hamming': h_hamming,}
# HEURISTICS = {'h_hamming': h_hamming,'h_manhattan': h_manhattan, 'h_euclidean': h_euclidean, 'h_custom': h_harmonic}
#</HEURISTICS>
