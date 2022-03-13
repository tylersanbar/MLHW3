from math import inf


class MarkovDecisionProcess():
    def __init__(self) -> None:
        self.states = {}

        #Initialize Nodes
        mount_olympus    = State("Mount Olympus")
        oracle_of_delos  = State("Oracle of Delos")
        oracle_of_delphi = State("Oracle of Delphi")
        oracle_of_dodoni = State("Oracle of Dodoni")
        
        #Initialize Actions
        olympus_fly = Action("fly")
        olympus_fly.paths.append(Path( .9, 2, oracle_of_delphi))
        olympus_fly.paths.append(Path( .1, -1, mount_olympus))
        mount_olympus.actions["fly"] = olympus_fly
        
        olympus_walk = Action("walk")
        olympus_walk.paths.append(Path( .8,  2, oracle_of_dodoni))
        olympus_walk.paths.append(Path( .2, -2, oracle_of_delphi))
        mount_olympus.actions["walk"] = olympus_walk

        delos_fly = Action("fly")
        delos_fly.paths.append(Path( .4, -1, oracle_of_delphi))
        delos_fly.paths.append(Path( .4, -1, oracle_of_dodoni))
        delos_fly.paths.append(Path( .2, -1, oracle_of_delos))
        oracle_of_delos.actions["fly"] = delos_fly

        delphi_fly = Action("fly")
        delphi_fly.paths.append(Path( .7,  4, oracle_of_delos))
        delphi_fly.paths.append(Path( .3, -1, oracle_of_delphi))
        oracle_of_delphi.actions["fly"] = delphi_fly

        delphi_horse = Action("horse")
        delphi_horse.paths.append(Path( .8,  1, mount_olympus))
        delphi_horse.paths.append(Path( .2,  1, oracle_of_dodoni))
        oracle_of_delphi.actions["horse"] = delphi_horse

        dodoni_fly = Action("fly")
        dodoni_fly.paths.append(Path( .7,  2, mount_olympus))
        dodoni_fly.paths.append(Path( .3, -1, oracle_of_dodoni))
        oracle_of_dodoni.actions["fly"] = dodoni_fly

        dodoni_horse = Action("horse")
        dodoni_horse.paths.append(Path( .7,  0, mount_olympus))
        dodoni_horse.paths.append(Path( .3,  1, oracle_of_delphi))
        oracle_of_dodoni.actions["horse"] = dodoni_horse

        #Set states to MDP
        self.states["Mount Olympus"]    = mount_olympus
        self.states["Oracle of Delos"]  = oracle_of_delos
        self.states["Oracle of Delphi"] = oracle_of_delphi
        self.states["Oracle of Dodoni"] = oracle_of_dodoni
    
    def __str__(self) -> str:
        s = "MDP Graph\n"
        for node in self.states:
            s = s + str(node)
        return s

class State():
    def __init__(self, name) -> None:
        self.name = name
        self.actions = {}
        self.eligibility_trace = 0
        self.value = 0
    
    def __str__(self) -> str:
        s = self.name + "\n"
        return s

class Action():
    def __init__(self, name) -> None:
        self.name = name
        self.paths = []
        self.value = 0
        self.eligibility_trace = 0

class Path():
    def __init__(self, probability, reward, nextState) -> None:
        self.probability = probability
        self.reward = reward
        self.nextState = nextState

class Sequence():
    def __init__(self, mdp) -> None:
        self.terminal = False
        self.mdp = mdp
        #Olympus, walk, 2 Dodoni
        #Dodoni, fly, 2 olympus
        #Olympus, fly, -1, Olympus
        #Olympus, fly, 2, Delphi
        #Delphi, fly, 4 Delos
        #Delos, fly -1, 
        self.episodes = {
        "1" : 
            {"1" : ["Mount Olympus", "walk", 2, "Oracle of Dodoni"],
            "2" : ["Oracle of Dodoni", "fly", 2, "Mount Olympus"],
            "3" : ["Mount Olympus", "fly", -1, "Mount Olympus"]},
        "2" : 
            {"1" : ["Mount Olympus", "fly", 2, "Oracle of Delphi"],
            "2" : ["Oracle of Delphi", "fly", 4, "Oracle of Delos"],
            "3" : ["Oracle of Delos", "fly", -1, None]}
        }
    
    def getResult(self, episode_number, step_number):
        if (episode_number == 1 and step_number == 3) or (episode_number == 2 and step_number == 2): self.terminal = True 
        return self.episodes[str(episode_number)][str(step_number)]

    def getAction(self, episode_number, step_number):
        state = self.episodes[str(episode_number)][str(step_number)][0]
        action = self.episodes[str(episode_number)][str(step_number)][1]
        return mdp.states[state].actions[action]

    def getNextAction(self, episode_number, step_number):
        if(episode_number == 1 and step_number == 3):
            state = self.episodes["2"]["1"][0]
            action = self.episodes["2"]["1"][1]
            return mdp.states[state].actions[action]
        state = self.episodes[str(episode_number)][str(step_number+1)][0]
        action = self.episodes[str(episode_number)][str(step_number+1)][1]
        return mdp.states[state].actions[action]

def TemporalDifference(alpha, discount_rate, mdp, sequence, iterations, trace_decay = 0):
    #Initialize states with V = 0, Q = 0, e = 0
    mdp = mdp
    state = mdp.states["Mount Olympus"]
    episode_number = 0
    while(episode_number < iterations):
        #Re-initialize eleigibility traces vector to 0.0 at beginning of each episode
        sequence.terminal = False
        for state in mdp.states.values():
            state.eligibility_trace = 0
        episode_number += 1
        state = mdp.states["Mount Olympus"]
        step_number = 0
        
        while(not sequence.terminal):
            step_number += 1
            result = sequence.getResult(episode_number, step_number)
            reward = result[2]
            nextState = mdp.states[result[3]]
            error = reward + discount_rate * nextState.value - state.value
            state.eligibility_trace += 1
            for state in mdp.states.values():
                state.value = state.value + alpha * error * state.eligibility_trace
                state.eligibility_trace = discount_rate * trace_decay * state.eligibility_trace
            state = nextState
        #Print value of states after each sequence
        print("Episode ", episode_number)
        for state in mdp.states.values():
            print(state.name, " ", state.value)
        print("\n")

def Sarsa(alpha, discount_rate, mdp, sequence, iterations, trace_decay = 0):
    #Initialize states with V = 0, Q = 0, e = 0
    state = mdp.states["Mount Olympus"]
    episode_number = 0
    while(episode_number < iterations):
        #Re-initialize eligibility traces vector to 0.0 at beginning of each episode
        sequence.terminal = False
        for state in mdp.states.values():
            state.eligibility_trace = 0
            for action in state.actions.values():
                action.eligibility_trace = 0
        episode_number += 1
        state = mdp.states["Mount Olympus"]
        step_number = 0
        
        while(not sequence.terminal):
            step_number += 1
            result = sequence.getResult(episode_number, step_number)
            reward = result[2]
            nextState = mdp.states[result[3]]
            action = sequence.getAction(episode_number, step_number)
            nextAction = sequence.getNextAction(episode_number, step_number)
            error = reward + discount_rate * nextAction.value - action.value
            action.eligibility_trace += 1
            for state in mdp.states.values():
                for action in state.actions.values():
                    action.value = action.value + alpha * error * action.eligibility_trace
                    action.eligibility_trace = discount_rate * trace_decay * action.eligibility_trace
            state = nextState
        #Print value of states after each sequence
        print("Episode ", episode_number)
        for state in mdp.states.values():
            print(state.name)
            for action in state.actions.values():
                print(action.name, " ", action.value)
        print("\n")

def QLearning(alpha, discount_rate, mdp, sequence, iterations, trace_decay = 0):
    #Initialize states with V = 0, Q = 0, e = 0
    mdp = mdp
    state = mdp.states["Mount Olympus"]
    episode_number = 0
    while(episode_number < iterations):
        #Re-initialize eligibility traces vector to 0.0 at beginning of each episode
        sequence.terminal = False
        for state in mdp.states.values():
            state.eligibility_trace = 0
            for action in state.actions.values():
                action.eligibility_trace = 0
        episode_number += 1
        state = mdp.states["Mount Olympus"]
        step_number = 0
        
        while(not sequence.terminal):
            step_number += 1
            result = sequence.getResult(episode_number, step_number)
            reward = result[2]
            nextState = mdp.states[result[3]]
            action = sequence.getAction(episode_number, step_number)
            nextAction = sequence.getNextAction(episode_number, step_number)
            maxAction = nextAction
            maxValue = nextAction.value
            for a in nextState.actions.values():
                if a.value > maxValue: maxAction = a

            error = reward + discount_rate * maxAction.value - action.value
            action.eligibility_trace += 1
            for state in mdp.states.values():
                for action in state.actions.values():
                    action.value = action.value + alpha * error * action.eligibility_trace
                    if(nextAction == maxAction): 
                        action.eligibility_trace = discount_rate * trace_decay * action.eligibility_trace
                    else: action.eligibility_trace = 0
            state = nextState
        #Print value of states after each sequence
        print("Episode ", episode_number)
        for state in mdp.states.values():
            print(state.name)
            for action in state.actions.values():
                print(action.name, " ", action.value)
        print("\n")

def TemporalDifferenceD(alpha, discount_rate, mdp, sequence, iterations, trace_decay = 0.6):
    #Initialize states with V = 0, Q = 0, e = 0
    mdp = mdp
    state = mdp.states["Mount Olympus"]
    episode_number = 0
    while(episode_number < iterations):
        #Re-initialize eleigibility traces vector to 0.0 at beginning of each episode
        sequence.terminal = False
        for state in mdp.states.values():
            state.eligibility_trace = 0
        episode_number += 1
        state = mdp.states["Mount Olympus"]
        step_number = 0
        
        while(not sequence.terminal):
            step_number += 1
            result = sequence.getResult(episode_number, step_number)
            reward = result[2]
            nextState = mdp.states[result[3]]
            error = reward + discount_rate * nextState.value - state.value
            state.eligibility_trace += 1
            for state in mdp.states.values():
                state.value = state.value + alpha * error * state.eligibility_trace
                state.eligibility_trace = discount_rate * trace_decay * state.eligibility_trace
            state = nextState
        #Print value of states after each sequence
        print("Episode ", episode_number)
        for state in mdp.states.values():
            print(state.name, " V = ", state.value, "| e = ", state.eligibility_trace)
        print("\n")
if __name__ == "__main__":
    α = .05
    γ = .9
    #Question 1
    print("Question 1\n")
    mdp = MarkovDecisionProcess()
    sequence = Sequence(mdp)
    TemporalDifference(α, γ, mdp, sequence, 2)

    #Question 2
    print("Question 2\n")
    mdp = MarkovDecisionProcess()
    sequence = Sequence(mdp)
    Sarsa(α, γ, mdp, sequence, 2)

    print("Question 3\n")
    mdp = MarkovDecisionProcess()
    sequence = Sequence(mdp)
    QLearning(α, γ, mdp, sequence, 2)

    print("Question 4\n")
    mdp = MarkovDecisionProcess()
    sequence = Sequence(mdp)
    TemporalDifferenceD(α, γ, mdp, sequence, 2)

# 7.5 Accumulating Eligibility Trace
# if (s != s_t): e_t(s) = discount_rate^trace_decay_parameter * e_t-1(s) 
# if (s == s_t): e_t(s) = discount_rate^trace_decay_parameter * e_t-1(s) + 1

# 7.6 TD error for state-value prediction
# d_t = r_t+1 + discount_rate * Vt(s_t+1) - Vt(s_t)

#Complete algorithm for on-line TD(trace_decay_parameter)
#Initialize V(s) arbitrarily and e(s)=0 for all states
#Repeat (for each episode)
#   a = action given by policy for state
#   Take action a, observe reward r and next state s'
#   d = reward + discount_rate * V(s') - V(s)
#   e(s) = e(s) + 1
#   for all s:
#       V(s) = V(s) + ade(s)
#   s = s'
#Until s is terminal

#Agent Hermes has following experience over the following two sequences:
# σ1 : Olympus(walk, 2) -> Dodoni(fly, 2) -> Olympus(fly, -1) -> Olympus
# σ2 : Olympus(fly, 2) -> Delphi(fly, 4) -> Delos
# α = 0.05, γ = 0.9
# V and Q initialized to 0 for all states and state-action pairs before these two sequences σ1 and σ2 are observed
# should NOT re-initialize the V and Q values to 0 after you are done processing σ1 and you are about to start processing σ2
# for TD(λ) you should re-initialize the elibility traces vector to 0.0 at the beginning of each episode/sequence, not just at the beginning of each part/question
#HW3 Questions
# a) What is the value of V for all states after each sequence using TD-learning?
# b) What is the value of Q for all state-action pairs after each sequence using Sarsa?
# c) What is the value of Q for all state-action pairs after each sequence using Q-learning?
# d) What is the value of V and e for all states after each sequence using TD(0.6) using accumulating traces?
# e) Explain why the middle values of λ are better for learning in general with eligibility traces.