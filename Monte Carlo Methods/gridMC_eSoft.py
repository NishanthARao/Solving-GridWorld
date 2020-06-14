#Solving a 4x4 gridworld using Monte Carlo Every-visit On Policy method (epsilon-greedy).
#The policy pi_(a|s) takes 4 actions equiprobably: Left, Right, Up, Down. Thus pi_(a|s) = 0.25 for all states
#The agent is transferred deterministically, hence p(s',r|s,a) = 1 for all states

import random
import numpy as np
from numpy.random import choice

#GRID:
#-------------
#| 0  1  2  3|
#| 4  5  6  7|
#| 8  9 10 11|
#|12 13 14 15|
#-------------
#With states 0,15 being the terminal states

#provide a reward of -1 to all the transitions.
r = -1
actions = ['<', '>', '^', 'v']
iterations = int(input("Please enter the number of iterations to be made: "))

#Assign Q(s,a) randomly, except the terminal states as they have 0 values for all actions.
#Each row represents state(1-16) with state1 and state16 being terminal states. Each column represents action in the order: left, right, up, down
epsilon = 0.01
Q_sa = np.random.rand(16,4)
Q_sa[0,:] = np.zeros(4)
Q_sa[15,:] = np.zeros(4)
pi_as = [[0.25]*4]*16 #16x4 list, with each rows representing probabilities of choosing l,r,u,d actions respectively for each state
pi_as[0] = [0]*4    #These are terminal states. Thus no actions are taken here, and the probabilities are all zeros.
pi_as[15] = [0]*4 
grid_actions = [['', '', '', ''], ['', '', '', ''], ['', '', '', ''], ['', '', '', '']]
#=======================================================================================================================================
#Generate Trajectories based on pi_(a|s) given state and the action took in that state
def generate_episode(state, action, pi_as):
    trajectory = []
    while(state != 0 and state != 15):
        if(action == 1):#Left
            state_ = state - 1
            if(state_ == 3 or state_ == 7 or state_ == 11): #If the agent hits the left wall, it stays in its previous location
                state_ = state_ + 1
        if(action == 2):#Right
            state_ = state + 1
            if(state_ == 4 or state_ == 8 or state_ == 12): #If the agent hits the right wall, it stays in its previous location
                state_ = state_ - 1
        if(action == 3):#Top
            state_ = state - 4
            if(state_ == -3 or state_ == -2 or state_ == -1): #If the agent hits the top wall, it stays in its previous location
                state_ = state_ + 4
        if(action == 4):#Down
            state_ = state + 4
            if(state_ == 16 or state_ == 17 or state_ == 18): #If the agent hits the bottom wall, it stays in its previous location
                state_ = state_ - 4
        trajectory.append([state, action]) #Append s, a to the trajectory. [[S1, A1], [S2, A2], ...] and so on. Reward is -1 for all transitions, thus not appended in the trajectory.
        action = choice([1,2,3,4], p=pi_as[state]) #Choose any action from 1-4 i.e left, right, top, or down based on their probabilities
        state = state_ #Previous state becomes current state in the next iteration
    return trajectory
#=====================================================================================================================================
#Generalised Policy Iteration:
for _ in range(0, iterations):
    state = np.random.randint(1,15) #Explore all the state-action pairs with equal probability
    action = choice([1,2,3,4], p=pi_as[state])
    value = 0.0
    j = 0.0
    trajectory = generate_episode(state, action, pi_as) #Generate a sample trajectory
    G = 0
    for i in range(1, len(trajectory)):
        G += -1
        if([state, action] == trajectory[i]):
            j += 1.0
            value += (1.0/j)*(G - value)    #Update the value of the state-action pair incrementally
            Q_sa[state, action-1] = value
            A = np.argmax(Q_sa[state,:])
            probability_actions = [epsilon/len(actions)]*4
            probability_actions[A] = 1 - epsilon + epsilon/len(actions)
            pi_as[state] = probability_actions
#=======================================================================================================================================    
#Show visually what actions to be taken in the gridworld
for i in range(0, len(pi_as)):
    if i < 3:
        grid_actions[0][i+1] = actions[pi_as[i+1].index(max(pi_as[i+1]))]
    if 4 <= i < 8:
        grid_actions[1][i-4] = actions[pi_as[i].index(max(pi_as[i]))]
    if 8<= i < 12:
        grid_actions[2][i-8] = actions[pi_as[i].index(max(pi_as[i]))]
    if 12<= i < 15:
        grid_actions[3][i-12] = actions[pi_as[i].index(max(pi_as[i]))]
print("The actions to be taken after " + str(iterations) + " iterations are:")
print(np.asarray(grid_actions))
#=========================================================================================================================================
