# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            tempValues = self.values.copy()
            for state in self.mdp.getStates():
                # check for terminal state
                if self.mdp.isTerminal(state):
                    tempValues[state] = 0
                # get actions with best qValue otherwise
                else:
                    max_qVal = float('-inf')
                    for a in self.mdp.getPossibleActions(state):
                        qVal = self.computeQValueFromValues(state, a)
                        if qVal >= max_qVal:
                            max_qVal = qVal
                            tempValues[state] = max_qVal
            self.values = tempValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        for sAndP in self.mdp.getTransitionStatesAndProbs(state, action):
            probVal = sAndP[1]
            rewardVal = self.mdp.getReward(state, action, sAndP[0])
            discountVal = self.discount
            # implement the equation for calculating the qValue
            qVal +=  probVal * (rewardVal + discountVal*self.values[sAndP[0]])
        return qVal

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # check for terminal state
        if self.mdp.isTerminal(state):
            return None
        # get action otherwise
        maxReward = float('-inf')
        actionTaken = ""
        for a in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, a)
            # action that has the best reward (max) should be taken
            if qVal >= maxReward:
                actionTaken = a
                maxReward = qVal
        return actionTaken

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            # if state is terminal, nothing happens
            if not self.mdp.isTerminal(state):
                max_qVal = float('-inf')
                for a in self.mdp.getPossibleActions(state):
                    qVal = self.getQValue(state, a)
                    if qVal >= max_qVal:
                        max_qVal= qVal
                self.values[state] = max_qVal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # get predecessors
        predecessors = self.getPredecessors()
        # initialize empty priority queue
        priorityQ = util.PriorityQueue()
        # for each non-terminal state,
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                # find the absolute value of the difference between current
                # value of s in self.values and the highest Q-value across all
                # possible actions from s; this is diff
                max_qVal = self.getMax_qValue(s)
                diff = abs(max_qVal - self.values[s])
                # Push s into the priority queue with priority -diff
                priorityQ.update(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1,
        for i in range(self.iterations):
            # if priority queue is empty, terminate
            if priorityQ.isEmpty():
                break
            # pop a state s off the priority queue.
            temp_s = priorityQ.pop()
            # update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(temp_s):
                max_qVal = self.getMax_qValue(temp_s)
                self.values[temp_s] =  max_qVal
            # For each predecessor p of s, do:
            for p in predecessors[temp_s]:
                if not self.mdp.isTerminal(p):
                    # Find the absolute value of the difference between the
                    # current value of p in self.values and the highest
                    # Q-value across all possible actions from p; this is diff
                    max_qVal = self.getMax_qValue(p)
                    diff = abs(max_qVal - self.values[p])
                    # If diff > theta,
                    if diff > self.theta:
                        # push p into the priority queue with priority -diff
                        priorityQ.update(p, -diff)

    def getPredecessors(self):
        """
        Function to compute predecessors of all states, which are all states
        that have a nonzero probability of reaching s by taking some action a
        """
        predecessors = {} # set to avoid duplicates
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                for a in self.mdp.getPossibleActions(s):
                    for sAndP in self.mdp.getTransitionStatesAndProbs(s, a):
                        # if state is in predecessors
                        if sAndP[0] in predecessors:
                            predecessors[sAndP[0]].add(s)
                        # if state is not in predecessors
                        else:
                            predecessors[sAndP[0]] = {s}
        return predecessors

    def getMax_qValue (self, state):
        """ Function to compute the max Q value for a given state """
        max_qVal = float('-inf')
        for a in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, a)
            if qVal >= max_qVal:
                max_qVal= qVal
        return max_qVal
