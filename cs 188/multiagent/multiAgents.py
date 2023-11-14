# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()
        foods = currentGameState.getFood().asList()
        x, y = newPos
        capsule = currentGameState.getCapsules()

        foods_dist = []
        for x in foods:
            foods_dist.append(manhattanDistance(newPos, x))

        ghost_dist = []
        for x in newGhostStates:
            ghost_dist.append(manhattanDistance(newPos, x.getPosition()))

        if len(foods_dist):
            score -= min(foods_dist)

        if ghost_dist:
            if min(ghost_dist) < 2 and max(newScaredTimes) < 1:
                score -= (min(ghost_dist) + 2)




        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        currindex = 0
        depth = self.depth

        def value(state, currindex, currdepth):
            if currdepth == 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), "")
            if currindex == 0:
                return maxvalue(state, currindex, currdepth)
            else:
                return minvalue(state, currindex, currdepth)
        def maxvalue(state, currindex, currdepth):
            currindex += 1
            actions = state.getLegalActions(0)
            v = -float('inf')
            besta = ''
            succ = []
            for a in actions:
                succ.append(state.generateSuccessor(0, a))
            for s in range(len(succ)):
                val = value(succ[s], currindex, currdepth)
                if val[0] > v:
                    besta = actions[s]
                    v = val[0]
            return v, besta
        def minvalue(state, currindex, currdepth):
            actions = state.getLegalActions(currindex)
            currindex += 1
            if currindex == state.getNumAgents():
                currdepth -= 1
                currindex = 0
            v = float('inf')
            besta = ''
            succ = []
            for a in actions:
                succ.append(state.generateSuccessor(currindex - 1, a))
            for s in range(len(succ)):
                val = value(succ[s], currindex, currdepth)
                if val[0] < v:
                    besta = actions[s]
                    v = val[0]
            return v, besta
        return value(gameState, currindex, depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currindex = 0
        depth = self.depth

        def value(state, currindex, currdepth, alph, bet):
            if currdepth == 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), "")
            if currindex == 0:
                return maxvalue(state, currindex, currdepth, alph, bet)
            else:
                return minvalue(state, currindex, currdepth, alph, bet)
        def maxvalue(state, currindex, currdepth, alph, bet):
            currindex += 1
            actions = state.getLegalActions(0)
            v = -float('inf')
            besta = ''
            for a in actions:
                val = value(state.generateSuccessor(0, a), currindex, currdepth, alph, bet)
                if val[0] > v:
                    besta = a
                    v = val[0]
                if v > bet:
                    return v, besta
                alph = max(alph, v)
            return v, besta
        def minvalue(state, currindex, currdepth, alph, bet):
            actions = state.getLegalActions(currindex)
            currindex += 1
            if currindex == state.getNumAgents():
                currdepth -= 1
                currindex = 0
            v = float('inf')
            besta = ''
            for a in actions:
                val = value(state.generateSuccessor(currindex - 1, a), currindex, currdepth, alph, bet)
                if val[0] < v:
                    besta = a
                    v = val[0]
                if v < alph:
                    return v, besta
                bet = min(bet, v)
            return v, besta
        return value(gameState, currindex, depth, -float('inf'), float('inf'))[1]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        currindex = 0
        depth = self.depth

        def value(state, currindex, currdepth):
            if currdepth == 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), "")
            if currindex == 0:
                return maxvalue(state, currindex, currdepth)
            else:
                return expecti(state, currindex, currdepth)
        def maxvalue(state, currindex, currdepth):
            currindex += 1
            actions = state.getLegalActions(0)
            v = -float('inf')
            besta = ''
            succ = []
            for a in actions:
                succ.append(state.generateSuccessor(0, a))
            for s in range(len(succ)):
                val = value(succ[s], currindex, currdepth)
                if val[0] > v:
                    besta = actions[s]
                    v = val[0]
            return v, besta
        def expecti(state, currindex, currdepth):
            currindex += 1
            if currindex == state.getNumAgents():
                currdepth -= 1
                currindex = 0
            v = 0
            actions = state.getLegalActions(currindex - 1)
            succ = []
            for a in actions:
                succ.append(state.generateSuccessor(currindex - 1, a))
            for s in range(len(succ)):
                prob = 1/len(actions)
                v += prob * value(succ[s], currindex, currdepth)[0]
            return v, ''
        return value(gameState, currindex, depth)[1]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <consider distance to nearest food, ghost, and capsule with weight on eliminating ghost if possible>
    """

    """pos = list(currentGameState.getPacmanPosition())
    food = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    score = currentGameState.getScore()

    foods_dist = []

    ghost_dist = []

    capsules_dist = []


    for f in food:
        a = abs(f[0] - pos[0])
        b = abs(f[1] - pos[1])
        foods_dist.append(manhattanDistance(pos, f))

    for x in ghost_states:
        ghost_pos = x.getPosition()
        c = abs(ghost_pos[0] - pos[0])
        d = abs(ghost_pos[1] - pos[1])
        ghost_howfar = c+d
        ghost_dist.append(manhattanDistance(pos, x.getPosition()))

    if min(ghost_dist) < min(scared_times):
        score += min(ghost_dist)

    if min(ghost_dist) < 80 and min(scared_times) < 1:
        score -= min(ghost_dist)

    if len(foods_dist):
        score += min(foods_dist)

    if capsules:
        for i in capsules:
            capsules_dist.append(manhattanDistance(pos, i))
        score += min(capsules_dist)




    return score"""

    pos_rn = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    food_dist = [manhattanDistance(x, pos_rn) for x in food_list]

    capsules = currentGameState.getCapsules()
    capsules_dist = []

    ghost_st = currentGameState.getGhostStates()

    score = currentGameState.getScore() - 2.5 * len(food_list)

    for g in ghost_st:
        ghost_pos = g.getPosition()
        ghost_dist_rn = util.manhattanDistance(ghost_pos, pos_rn)

        timer = g.scaredTimer

        if ghost_dist_rn < timer:
            score += 100 - ghost_dist_rn

    if len(food_dist):
        score -= min(food_dist)
    else:
        score += 10

    return score



# Abbreviation
better = betterEvaluationFunction
