#Solving a 4x4 gridworld using Dynamic Programming.
#The policy is assumed to take 4 actions equiprobably: Up, Down, Left, Right
#The agent is transferred deterministically, hence p(s',r|s,a) = 1 for all states

import random
import numpy as np

#GRID:
#-------------
#| 0  1  2  3|
#| 4  5  6  7|
#| 8  9 10 11|
#|12 13 14 15|
#-------------
#With states 0,15 being the terminal states

iterations = int(input("Please enter the number of iterations to be made: "))
#provide a reward of -1 to all the transitions. Assign pi(a|s) = 0.25
r = -1
pi_as = 0.25
actions = ['<', '>', '^', 'v']
#create a 4x4 grid with the values being randomly assigned in each cell, with terminal states having a value of 0.
grid_value = np.random.rand(4,4)
prev_grid_value = np.random.rand(4,4)
grid_value[0,0] = 0
grid_value[3,3] = 0

#============================================== Policy Iteration ====================================================================
for _ in range(0, iterations):
    #================================= Policy Evaluation ===================================================
    #In each state, the bellman equation for value function is computed. The if-else construct takes care of index errors
    for i in range(0,4):
        for j in range(0,4):        
            if(i != 0 and i != 3 and j != 0 and j!=3): #If you are in (1,1), or (1,2) or (2,1) or (2,2) positions. (Note: index values are from (0-3) 
                grid_value[i,j] = pi_as*((r + grid_value[i+1, j]) + (r + grid_value[i,j+1]) + (r + grid_value[i-1, j]) + (r + grid_value[i, j-1]))
                
            elif(i == 0): #If you are in the first row
                if(j == 3): #If you are in the first row and last column
                    grid_value[i,j] = pi_as*((r + grid_value[i+1, j]) + (r + grid_value[i,j]) + (r + grid_value[i, j]) + (r + grid_value[i, j-1]))
                elif(j ==0): #If you are in (0,0) which is a terminal state
                    pass
                else: #If you are in the first row, except the first and last columns
                    grid_value[i,j] = pi_as*((r + grid_value[i+1, j]) + (r + grid_value[i,j+1]) + (r + grid_value[i, j]) + (r + grid_value[i, j-1]))
                    
            elif(i == 3): #If you are in the last row
                if(j == 0): #If you are in the last row and first column
                    grid_value[i,j] = pi_as*((r + grid_value[i, j]) + (r + grid_value[i,j+1]) + (r + grid_value[i-1, j]) + (r + grid_value[i, j]))
                elif(j == 3): #If you are in (3,3) which is a terminal state
                    pass
                else: #If you are in the last row, except the first and last columns
                    grid_value[i,j] = pi_as*((r + grid_value[i, j]) + (r + grid_value[i,j+1]) + (r + grid_value[i-1, j]) + (r + grid_value[i, j-1]))
            
            else: #If you are in the first or second row
                if(j == 0): #If you are in the first column
                    grid_value[i,j] = pi_as*((r + grid_value[i+1, j]) + (r + grid_value[i,j+1]) + (r + grid_value[i-1, j]) + (r + grid_value[i, j]))
                elif(j == 3): #If you are in the last column
                    grid_value[i,j] = pi_as*((r + grid_value[i+1, j]) + (r + grid_value[i,j]) + (r + grid_value[i-1, j]) + (r + grid_value[i, j-1]))

    #=========================== POLICY IMPROVEMENT =========================================================
    grid_actions = [['', '', '', ''], ['', '', '', ''], ['', '', '', ''], ['', '', '', '']]
    for i in range(0,4):
        for j in range(0,4):
            if(i != 0 and i != 3 and j != 0 and j!=3):
                #Find the action that yields the maxmimum return
                a = [(r + grid_value[i,j-1]), (r + grid_value[i,j+1]), (r + grid_value[i-1, j]), (r + grid_value[i+1,j])]
                indices = [ab for ab, cd in enumerate(a) if cd == max(a)] #Returns a list of indices, in case two or more max elements
                for gh in indices:
                    grid_actions[i][j] = grid_actions[i][j] + actions[gh]
                
            elif(i == 0):
                if(j == 3):
                    a = [(r + grid_value[i,j-1]), (r + grid_value[i,j]), (r + grid_value[i, j]), (r + grid_value[i+1,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh]
                elif(j ==0):
                    pass
                else:
                    a = [(r + grid_value[i,j-1]), (r + grid_value[i,j+1]), (r + grid_value[i, j]), (r + grid_value[i+1,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh]
                    
            elif(i == 3):
                if(j == 0):
                    a = [(r + grid_value[i,j]), (r + grid_value[i,j+1]), (r + grid_value[i-1, j]), (r + grid_value[i,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh]
                elif(j == 3):
                    pass
                else:
                    a = [(r + grid_value[i,j-1]), (r + grid_value[i,j+1]), (r + grid_value[i-1, j]), (r + grid_value[i,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh]
            
            else:
                if(j == 0):
                    a = [(r + grid_value[i,j]), (r + grid_value[i,j+1]), (r + grid_value[i-1, j]), (r + grid_value[i+1,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh]
                elif(j == 3):
                    a = [(r + grid_value[i,j-1]), (r + grid_value[i,j]), (r + grid_value[i-1, j]), (r + grid_value[i+1,j])]
                    indices = [ab for ab, cd in enumerate(a) if cd == max(a)]
                    for gh in indices:
                        grid_actions[i][j] = grid_actions[i][j] + actions[gh] 
#======================== End of Policy Iteration ===============================================================================
print("==============================================================================")
print("The values for each of the states after " + str(iterations) + " iterations are: ")
print grid_value
print("\n")
print("The actions to be taken in each state after " + str(iterations) + " iterations are: ")
print np.asarray(grid_actions)
print("==============================================================================")

