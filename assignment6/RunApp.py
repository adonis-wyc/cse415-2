'''RunApp.py

This program imports Grid.py and MDP.py and runs certain algorithms to
demonstrate aspects of reinforcement learning.

CSE 415  Students: Fill in the missing code where indicated.

'''

import MDP, Grid
import sys

def GW_Values_string(V_dict):
    out_str = ''
    for row in range(2,-1,-1):
        for col in range(4):
            state_key = (col,row)
            formated = " %04.3f " % V_dict[state_key]
            if V_dict[state_key] >= 0:
                formated = " "+formated
            out_str += formated
        out_str += '\n'
    return out_str

def format_key(source_dict, dict_key):
    formated = " %04.3f " % source_dict[dict_key]
    if source_dict[dict_key] >= 0:
        formated = " "+formated
    return formated

def GW_QValues_string(Q_dict):
    out_str = ''
    out_str += '---------------------------------------------------------------------------\n'
    for row in range(2,-1,-1):
        # print North
        out_str += '    '
        for col in range(4):
            state_key = (col,row)
            state_key = (state_key, 'North')
            out_str += format_key(Q_dict,state_key)
            out_str += '     |     '
        out_str += '\n'

        # print West, East
        for col in range(4):
            state_key = (col,row)
            state_key = (state_key, 'West')
            out_str += format_key(Q_dict,state_key)

            state_key = (col,row)
            state_key = (state_key, 'East')
            out_str += format_key(Q_dict,state_key)
            out_str += ' | '

        out_str += '\n'

        # print South
        out_str += '    '
        for col in range(4):
            state_key = (col,row)
            state_key = (state_key, 'South')
            out_str += format_key(Q_dict,state_key)
            out_str += '     |     '

        out_str += '\n'
        out_str += '---------------------------------------------------------------------------\n'
    return out_str


def GW_Policy_string():
    # IMPLEMENT THIS
    pass


def test():
    '''Create the MDP, then run an episode of random actions for 10 steps.'''
    grid_MDP = MDP.MDP()
    grid_MDP.register_start_state((0,0))
    grid_MDP.register_actions(Grid.ACTIONS)
    grid_MDP.register_operators(Grid.OPERATORS)
    grid_MDP.register_transition_function(Grid.T)
    grid_MDP.register_reward_function(Grid.R)


    grid_MDP.generateAllStates()
    # grid_MDP.valueIteration(0.7, 20)

    # grid_MDP.random_episode(100)

    # Uncomment the following, when you are ready...

    grid_MDP.valueIteration( 0.6, 15)
    print(GW_Values_string(grid_MDP.V))

    grid_MDP.QLearning( 0.6, 100, 0.5)
    print(GW_QValues_string(grid_MDP.Q))

    # print(GW_Policystring())

test()
