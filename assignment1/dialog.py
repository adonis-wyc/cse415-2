# dialog.py
# This program runs a dialog between two agents, which must be defined
# elsewhere as separate modules.

import grahamtk
import wtmenten
N_TURNS = 10

turn = 0
agentB = grahamtk.CrazyJoe()
agentA = wtmenten.Jokester()

print(str(turn)+"A: "+agentA.agentName() + ': ' + agentA.introduce()+"\n")
print(str(turn)+"B: "+agentB.agentName() + ': ' + agentB.introduce()+"\n")
remark = "Good morning."
for i in range(N_TURNS):
    turn += 1
    remark = agentA.respond(remark)
    print(str(turn)+"A: "+agentA.agentName() + ': ' + remark+"\n")
    remark = agentB.respond(remark)
    print(str(turn)+"B: "+agentB.agentName() + ': ' + remark+"\n")
