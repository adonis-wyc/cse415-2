'''MDP.py
S. Tanimoto, May 2016, 2017.

Provides representations for Markov Decision Processes, plus
functionality for running the transitions.

The transition function should be a function of three arguments:
T(s, a, sp), where s and sp are states and a is an action.
The reward function should also be a function of the three same
arguments.  However, its return value is not a probability but
a numeric reward value -- any real number.

operators:  state-space search objects consisting of a precondition
 and deterministic state-transformation function.
 We assume these are in the "QUIET" format used in earlier assignments.

actions:  objects (for us just Python strings) that are
 stochastically mapped into operators at runtime according
 to the Transition function.


CSE 415 STUDENTS: Implement the 3 methods indicated near the
end of this file.

'''
import random
import copy
import itertools
from collections import defaultdict

REPORTING = True

class MDP:
    def __init__(self):
        self.known_states = set()
        self.succ = {} # hash of adjacency lists by state.

    def register_start_state(self, start_state):
        self.start_state = start_state
        self.known_states.add(start_state)

    def register_actions(self, action_list):
        self.actions = action_list

    def register_operators(self, op_list):
        self.ops = op_list

    def register_transition_function(self, transition_function):
        self.T = transition_function

    def register_reward_function(self, reward_function):
        self.R = reward_function

    def state_neighbors(self, state):
        '''Return a list of the successors of state.  First check
           in the hash self.succ for these.  If there is no list for
           this state, then construct and save it.
           And then return the neighbors.'''
        neighbors = self.succ.get(state, False)
        if neighbors==False:
            neighbors = [op.apply(state) for op in self.ops if op.is_applicable(state)]
            self.succ[state]=neighbors
            self.known_states.update(neighbors)
        return neighbors

    def random_episode(self, nsteps):
        self.current_state = self.start_state
        self.known_states = set()
        self.known_states.add(self.current_state)
        self.current_reward = 0.0
        for i in range(nsteps):
            self.take_action(random.choice(self.actions))
            if self.current_state == 'DEAD':
                print('Terminating at DEAD state.')
                break
        if REPORTING: print("Done with "+str(i)+" of random exploration.")

    def take_action(self, a):
        s = self.current_state
        neighbors = self.state_neighbors(s)
        threshold = 0.0
        rnd = random.uniform(0.0, 1.0)
        r = self.R(s,a,s)
        for sp in neighbors:
            threshold += self.T(s, a, sp)
            if threshold>rnd:
                r = self.R(s, a, sp)
                s = sp
                break
        self.current_state = s
        self.known_states.add(self.current_state)
        if REPORTING: print("After action "+a+", moving to state "+str(self.current_state)+\
                            "; reward is "+str(r))

    def assess_action(self, s, a):
        # s = self.current_state
        neighbors = self.state_neighbors(s)
        threshold = 0.0
        rnd = random.uniform(0.0, 1.0)
        r = self.R(s,a,s)
        for sp in neighbors:
            threshold += self.T(s, a, sp)
            if threshold>rnd:
                r = self.R(s, a, sp)
                s = sp
                break
        return s, r

    def generateAllStates(self):
        # IMPLEMENT THIS
        self.known_states = set()

        self.bfsSearch(self.start_state)
        print(self.known_states)

    def bfsSearch(self, s):
        self.known_states.add(s)
        neighbors = self.state_neighbors(s)
        # products = itertools.product(neighbors, self.actions)
        # unexplored = [n,a for n,a in products if self.succ.get(n, False) == False]
        unexplored = [n for n in neighbors if self.succ.get(n, False) == False]
        if len(unexplored) == 0:
            return
        for n in unexplored:
            self.bfsSearch(n)


    def valueIteration(self, discount, iterations):
        self.V = defaultdict(float)

        for it in range(iterations):
            for s in self.known_states:
                    n_vals = []
                    for n in self.succ.get(s):
                        for a in self.actions:
                            val =  self.T(s, a, n) * (discount*self.V[n] + self.R(s, a, n))
                            n_vals.append(val)
                    max_val = max(n_vals) if len(n_vals) > 0 else self.V[s] # rock/dead has no neighbors
                    self.V[s] =  max_val

    def QLearning(self, discount, nEpisodes, epsilon):
        # IMPLEMENT THIS
        N = defaultdict(int)
        self.Q = defaultdict(int)

        for it in range(nEpisodes):
            self.current_state = self.start_state
            while (self.current_state != "DEAD"):
                s = self.current_state
                q_vals = []
                for a in self.actions:
                    sp, r = self.assess_action(s, a)
                    val = (discount*self.V[sp] + r)
                    q_vals.append((val, a, sp))

                rand = random.random()
                cand_val, cand_action, cand_sp = None, None, None
                if rand > epsilon:
                    cand_val, cand_action, cand_sp = max(q_vals, key=lambda x: x[0])
                else:
                    cand_val, cand_action, cand_sp = random.choice(q_vals)
                N[(s,cand_action)] += 1
                alpha = 1.0 / N[(s,cand_action)]
                old_val = self.Q[(s,cand_action)]
                self.Q[(s,cand_action)] = old_val + alpha * (cand_val - old_val)
                print(cand_val)
                self.take_action(cand_action)
                # self.current_state =
        # print(self.Q)


    def extractPolicy(self):
        self.optPolicy = {}
        for s in self.known_states:
            opt_val, opt_a = max([(self.Q[(s, a)], a) for a in self.actions])
            self.optPolicy[s] = opt_a
        return self.optPolicy
