# autonomous_trading




# STOCK TRADING MACHINE DEEP Q-LEARNING/REINFORCEMENT LEARNING PROJECT

This project addresses and aims to work on the importance of and the essential aspects of reinforcement and deep Q learning in the field of autonomous product machine training in order to maximize the reward and minimize the potential risks or losses. In the development of this project, several other projects were looked into in order to understand the working of different real world, gaming scenarios and situations. Through the detailed use of deep Q learning and reinforcement learning concepts, I have essentially broken them down to create an idea of a stock/price trading reinforcement learning project prompt, where I aim to trade stocks between 'n' number of companies or entities in a way through my investing amount that I maximize the profits and minimize the risk of loss through any action. Throughout the research work and the generation of information of this project, the actions, agents, states and the time frames were kept into mind while simultaneously handling the trading environment. For this, I have looked into several online videos and have noted down the essential aspects of both of the concepts and several underlining concepts through the completion of two courses : "Modern Reinforcement Learning - Actor Critic Algorithms" and "MATLAB and Simulink - programming use in the field of Autonomous systems and artificial intelligence". This ReadMe file contains all the research notes, the code, the detailed use and working of different algorithms, code, formulae, concepts, and theoretical assumptions of potentially solving reinforcement learning and Deep Q-Learning problems. Most importantly, this readme file contains the problem statement for this stock trading project work and the use of several concepts and approaches that I and others reading this can take to approach other similar problems in day to day lives. 

Before I give below all the notes created by me, I would like to state the IMPORTANCE of why I chose to approach and create this project problem and that is because - THE MOST IMPORTANT USE OF DEEP Q-LEARNING and REINFORCEMENT learning till date is in the coorporate financial industry/Trading industry to solve stock problems for any INDUSTRIAL GROWTH.

# RESEARCH WORK - NOTES GIVEN FOR ALL OF THE CONCEPTS USED IN THE WORKING OF EXAMPLE PROBLEMS AND THE CREATION OF THIS PROJECT PROBLEM AND ITS SOLUTION

1. A-I Reinforcement Learning 

    1. Reinforcement Learning : A general path towards artificial intelligence.
    2. RL : Multiple real world cases, there are tons of practical cases. Getting an algorithms as to how these cases work gets you on the top of the pile in order to get started with a start up.
    3. Will know all the policy gradients and all the critical algorithms creation from the scratch.
    4. How to read algorithms, To see how algorithms operate
    5. To learn deep reinforcement learning papers into code.
1. Fundamentals of Reinforcement Learning :

-> Reinforcement -> Deep Reinforcement

=>   ‘From paper into code’ -> Mathematical in logic

1. Not replicating numbers, not cheating with the world’s best scores
2. Training machines
3. Next video : Required background in software to complete this course


2. Required Background, Software, Hardware

  1. Python 3
2. Core features -> List comprehensions, looping, classes & inheritance (OOP)
3. Calculus, chain rule, deep neural networks.
4. To install : Torch, numpy, matplotlib, gym, box2d-py
5. Vim : Sublime
6. Familiar with python, mathematical background, python 3
7. CHECK THE COURSE FORUMS if you have questions
8. Answers : www.github.com/philtabor/


3. Review of Fundamental Concepts

Deals with an agent in contact with the environment changing its state from one state to another.

1. Various algorithms looks for these attributes. Algorithms are not possible without the Markov Decision Process:

Process that obeys the property that the current state of the item depends on the previous state of the item and the action that the agent took.

2. There are episodic returns and we can evaluate the rewards and calculate the state at which the agent stands. Some states have more value than the others.
3. We discount the VALUE OF FUTURE REWARDS.
4. Sum of discounted rewards -> Episodic return.
5. Agent needs probability of the actions and mapping states to actions. (Probability : 100%)
Learning from experience:

Interacts with the environment, tracking the rewards received, Need to have a relation between V and Q.

Bellman : Actions that lead to a valuable state such that the agent makes complete use of the rewards attained over time.

On Policy : One same policy to explore and engage
Off Policy : Separate policy

Agents keep a value of present rewards to discount the value of the future functions




4. Teaching an AI about Black Jack with Monte Carlo Prediction


1. Monte Carlo (MC) methods : Class of algorithms learning from experience with no prior knowledge
2. Agent will play games and calculate the value for each states and the rewards received.
Generalized policy iteration :  Start with a policy, calculate the value and making it more greedy.

3. First vs. Every Visit MC : Tracking rewards received after visiting states.

Rewards received after first visit -> First visit MC

Black Jack : 

Rewards : +1 for winning, 0 for draw, -1 for loss

Algorithm Overview :

Initialize the :
Policy to be evaluated, value function arbitrarily, list of returns for all states in the state space.

Repeat for large number of episodes.


￼<img width="763" alt="Screen Shot 2021-04-29 at 9 56 09 AM" src="https://user-images.githubusercontent.com/86114245/122622731-a85fed80-d067-11eb-839b-9b05ba31bd4f.png">


CODE SOLUTION: Watch the black jack code of the agent and the main class again for understanding later on.

Probability shows : of winning -> 0.888
				of losing 	 -> -0.201


Conclusion :

1. MC uses agent experience to estimate the value of the policy
2. Iterate the value of the policy
3. MC lets us calculate the value of the given policy



5. Monte Carlo Control Problem:

-> Improving the policy.
-> We will not have a complete model of our environment, without a model, we do not know how to transition to the next most valuable state

q. How to handle explore exploit dilemma?

1. Exploring Starts

-> With the start of each episode we are going to select the state, action, pair at random and get the accurate estimate of the action at a given policy.
-> In the case of the black jack, we pick an action at random and use to determine the policy.
-> We get a random total estimate and get a good coverage of state and action space.
-> This can be limiting : Even though we can select the actions at random, the agent might not be able to visit them. 
FOR THIS : USE EPSILON SOFT ACTION SELECTION :

-> Replaces the deterministic policy and helps us solve it. Allows efficient exploration.

-> Greed and exploration are taken into consideration

ALGORITHM OVERVIEW :



￼



=> Watch the coded solution part 5 of the course again later on for deeper understanding.




6. Temporal Difference Learning

-> There are some MC method drawbacks : 
1. They are of episodic in nature so hence, difficult for longer episodes
2. TDL : Perform the MC with updating the value.
3. The Updates will improve the agents policy and action value.
4. MC -> Had to wait until the end of the episode to have the correct estimate of the value.
5. TDL : The update on the other hand is performed every single time and is updated. We have updated the V every time step.
6. There is a TD error as well.
7. TDL methods are Bootstrapped that is that we are using updates to estimate the values of other estimates.

Bootstrapping methods help in modern reinforcement learning.

Conclusion : TDL is a model free bootstrapped learning and online method that updates the value function with each time steps for good long episodes with help in predictions and control problems.


7. Temporal Difference Prediction

-> For example, the cart pole environment, the epos ends based on the given conditions.
-> The state space : observation is a 4-tuple
-> The details given in the video for the course
-> To take different spaces for the state space and classify them into different intervals.
-> The Action Space and Policy : 

1. Push left -> Represented by a 0, Push right -> represented by a 1.
2. If the pole is left to the center, move left and vice versa.

How to compute the value of such a system? Given below is the algorithm :





Initialize policy to be evaluated
Initialize arbitrary V, terminal state gets 0
Repeat for large number of episodes:
	Initialize the environment
	For each step of the episode:
	Select action a according to policy for current 	state
	Take action and get reward and new state from 	env.
	*Given formula for V*
	Set current state to new state
np.digitze & np.linspace to discretize pole angle into 10 bins
If theta < 0 move left, else move right
5000 games, print V for all 10 states to terminal
Alpha = 0.1 and gamma = 0.99


—> CODE REFERRED TO IN THE VIDEO OF THE COURSE



Result : 0 and 10 states have the lowest values, when the agent is closest to losing the value, the rest are higher.

Observation : There is a lead up on one of the sides than the other one. —> Successful run.


Conclusion : 

1. Can focus on improving policies
2. Can move from digitizing to handle continuous state spaces
3. Next video : Control problem with queue learning.







8. Temporal Difference Control: Q Learning

In the previous model, we saw that the problem was soluble.
Moving on to a control problem and for that we will be focusing on Q learning as given below:


-> Q LEARNING:

1. Off policy, model free, bootstrapped method.
2. Don’t need a full model.
3. Using the epsilon greedy to see how often the agent selects the action in random
4. If random number < epsilon: explore else exploit
5. Decrease epsilon over time to a minimum value
6. Update the action value for greedy action

Update rule : Given formula in the video lecture. We compute the action value Q using the epsilon greedy policy.


ALGORITHM :



Initialize arbitrary Q, terminal state gets 0
Repeat for large number of episodes:
	Initialize the environment
	For each step of the episode:
	Select action a according to an epsilon greedy 	policy
	Take action and get reward and new state from 	env.
	*Given formula for Q*
	Set current state to new state
Epsilon start at 1, go down to 0.01 halfway through
If theta < 0 move left, else move right
5000 games, plot running average reward over last 100 games
Alpha = 0.01, Gamma = 0.99
Digitize for the other 3 variables


CODE given in the lecture, refer to it!


Results :  A linear increase till the epsilon goes down to 250000 games. 


Conclusion :

1. Q learning : Off policy, model free learning in real time, 
2. Can be used to beat non trivial environments
3. TD with actor critic methods.




9. Policy approximation and its advantages

In the previous unit, we used the MC and TD methods to solve prediction problems.

In this unit, we will learn a different alternative approach to learning the reinforcement learning problems called the ->

Q. What is a Policy?

Policy -> Probability distribution with the agent selecting each action.

In the case of MC, policy was completely deterministic
We had to learn the value functions. We have to figure out another method:

1. Policy Approximation :

Here, we parametrize the policy with neural networks, the policy parts are a function a theta. We use gradient ascent in J to use to calculate this.

All the methods that follow this are called policy gradient methods.

-> Advantages :

1. Policy is continuous and finite function. 
2. Works for continuous action spaces
3. All actions are sampled so no dilemma

All actions get sampled at least some of the times.
-> Go through all of the formulae


The algorithms are for high valued actions. Strong benefit over epsilon greedy policies.

Limitations of (Some) policy gradient methods:

1. Sample inefficiency
2. The agent keeps a long memory. In MC methods, the previous episodes are considered
3. Credit assignment problem
4. Unstable solutions : leads to large shifts into cliffs of poor performances
5. May still need too learn V, Q

Policy Gradient Theorem 

1. Refer to the formula. J is the performance of the agent. Defined by the parameters theta.
2. Are the updates helping? -> Depends on the environment. J is some complicated function. Since we have no way of knowing that, we are stuck.


Conclusion:

Can learn from experience to improve performance!

1. Approximating the policy with deep N.N.
2. Stochastic policy solves explore exploit dilemma
3. Handles continuous action spaces
4. Policy gradient theorem.







10. REINFORCE: Monte Carlo Policy Gradients

In this, we will cut up to the first policy algorithm. The gradient in J will give the gradients and the gradients are calculated by deep learning PyTorch.

Refer to all of the formulae from the lecture 


1. Algorithm:


Initialize deep N.N. to model agent’s policy
Repeat for large number of episodes:
	Generate episode using policy, keep track of 		probabilities
	For each step of the agent’s memory:
	Calculate the return G for the episode
	* FORMULA FOR THETA FOR T+1 *

One step at a time; will come back to this for review



Conclusion :

1. Don’t need distribution of states
2. Update rule in terms of known quantities.




11. Updating the LunarLander-v2 Environment

 Our goal : to land the lunar module to be landed on the moon. The episode ends and lands on the landing pad. This has an incentive.

The four actions : do nothing, fire the left, right engine.


Code solution : given in the video lecture.



12. Coding the policy network



Input layer : 128 neurons hidden layers -> Output layer
Adam optimizer : Use GPU


# ------------- X ------------------- X --------------------

1. NEXT STEP:
2. To execute a python program for a simple reinforcement problem -> for example, there are people having preferences of different kinds of coffee based off of different preferences using the reinforcement learning to learn the preferences of the people 
3. Get a data using which we can train a reinforcement model
4. State, Action, Reward -> Pick any problem, try to find it, out it in a data set


——> Take 2-3 days.
——> New -> Challenging -> So many data sets available from the internet -> Download a data set for reinforcement problems from the internet.
——> If you can figure the internet —> figure out a code, get a reference of the code and understand the code and create the environment on your laptop.




REINFORCEMENT LEARNING AND Q LEARNING - AN EXAMPLE OF A TAXI PROBLEM IN PYTHON


THEORY IN BRIEF :

1. Going from machine learning to deep learning is often more intuitive, however, when proceeding to reinforcement learning problems it could be more confusing.

Given below is a revision of reinforcement learning and Q-Learning, after which there is an example given using the technique of Q-Learning to solve a reinforcement problem - “The taxi problem” given in PYTHON 


1. What is reinforcement learning?

Reinforcement learning refers to most importantly DIFFERS FROM “input X leading to output Y” as it involves an agent interacting with its surrounding environment considering the best action to take. In this scenario the environment could be complex, uncertain, and also in this case, the agent’s behavior can be probabilistic and not deterministic.

Given below are the key aspects of reinforcement learning and there definitions for reference :



1. STATE SPACE :

In this case, we need to note the fact that the space state could be either discrete or finite or continuous or infinite. The state space can be defined as all the possible states that an agent could be in. We must be extremely careful and define all the states carefully so that the definition we have contains all the states that we essentially need.
For example, if we are solving a car problem, we might need the x-direction velocity, the y-direction velocity, or even the x and y direction acceleration.



2. ACTIONS SPACE :

The actions space can be defined as all the possible actions that an agent could possibly take. For example, in the case of having a robot example, a robot might take a left, right, forwards or backwards turn and therefore, due to this, the actions space is discrete or finite and constitutes much lesser elements than the state space.



3. TRANSITION PROBABILITY :

The transition probability as the name suggests is the function or the probability of getting to the next state (St + 1) where the given current state of the agent is St and the execution of that given action is At. This shows that the action of the agent’s behavior is probabilistic. THERE IS ALWAYS A POSSIBILITY THAT THE ROBOT WILL NOT DO WHAT YOU INSTRUCTED IT TO DO! This part of the concept is used in the Markov Decision Process but it is a necessary concept in the aspect of Q-Learning.



4. REWARD :

The reward can be defined as a function of the state and the action R (s, a). We can think of this as the desired goal that the agent aims to reach more quickly and efficiently! We often place a positive reward for the state function that we want the agent to get to and place a small negative reward/penalty for each extra step taken by the agent! 
FOR SEVERAL CONDITIONS, THE REWARD FUNCTION IS NOT WELL DEFINED. FOR THIS, THE AGENT HAS TO GO THROUGH A RUN OF A LARGE SAMPLE OF RANDOM ACTION EXECUTED BY THE AGENT AND GATHER THE INFORMATION TO ESTIMATE THE REWARDS.



5. POLICY :

The essential aspect is that A POLICY MAPS AND CREATES A STATE INTO AN ACTION - IT TAKES IN THE CURRENT STATE AS THE INPUT AND THEN OUTPUTS AN ACTION FOR THE AGENT TO TAKE.

IN REINFORCEMENT LEARNING PROBLEMS :

In reinforcement learning problems, the agent’s main goal is to maximize the total rewards. The agent in this case constantly interacts with the environment and explores to gather several information about what kind of rewards it obtains from executing actions at different states and then, using that, searches for an optimal action to be taken at each state to maximize the rewards function!





Q-Learning (Type of VALUE-BASED LEARNING ALGORITHM)

This can be essentially defined as a Value-based learning algorithm. In this case, the main goal of the agent is to maximize the “Value-function” suited to the given problem that the agent faces. Just like the REWARD function in Reinforcement Learning, the VALUE FUNCTION assesses the particular action in a given state for a given policy. IT NOT ONLY TAKES INTO ACCOUNT THE CURRENT REWARD BUT ALSO TAKES INTO ACCOUNT ALL OF THE FUTURE REWARDS RESULTING FROM THE PARTICULAR ACTION.

In this learning we have :
1. Q-TABLE that stores the Q-VALUES
2. Q-VALUES ARE stored for each of the state and each of the possible action
3. In this case, the agent explores the given environment and makes changes and updates the Q-VALUES with every single iteration!





LEARNING PROCESS FOR PROBLEM SOLVING INVOLVING Q-LEARNING 


STEP 1 : INITIALIZATION

This is the step in the learning process of Q-Learning were all of the Q-Values in the Q-Table are initialized to 0 due to which, The agent has no knowledge about what environment it is in.

STEP 2 : EXPLORE THE PLACE

As of now, we know that the agent keeps exploring the environment by executing actions in the states that it is in. 
IN THIS CASE, WE HAVE A PROBLEM THAT IS OF EXPLORATION VS EXPLOITATION!

There are two scenario cases to be covered :

1. We can define the action as the agents who keeps executing the action that returns the highest value function, therefore, generating a global optimum -> THIS PROCESS COULD BE SLOW AND PAINFUL!
2. Instead, what we can to for the action to take place by the agent is for the agent in the large action space, can occasionally select or choose its action at random, and therefore, doing this there is a chance that the agent by choosing the action at random will find the optimal value faster.

EXAMPLE : Amazon keeps showing the customer the recommended products based on the products that the customer has searched up on or previously bought. There is always a change that out of the given list, there could be a product that matched the customer’s preference. THIS GIVES THE RECOMMENDER SYSTEM NEW INFORMATION ABOUT THE CUSTOMER THAT WOULD OTHERWISE TAKE LONGER IN ORDER TO BECOME CLEARER.

THE WAY TO DO IT : The way to do this or use the exploration strategies might be to use the epsilon greedy strategy, this way the probability of the epsilon that we take for the random action chosen by the agent that maximizes the value function, will become clearer in the code part of this ‘TAXI PROBLEM’.


STEP 3 : OBSERVE THE REWARD

When the agent is in the process for exploration, the agent would observe the sort of rewards that it gets from executing a particular action (at) in that given state (St) for the next state to be initialized (St + 1).

STEP 4 : UPDATE THE VALUE FUNCTION

After the previous step of observing the reward function, this step focuses on updating the value function for the particular state and action pair using the given below formula, returning an updated Q-Table.

IN THIS FORMULA, WE HAVE THE LEARNING RATE AND DISCOUNT RATE AS TWO HYPER PARAMETERS :

The learning rate can be thought of as how it controls the weighting given to the current value compared to the new value

The discount rate can be defined as a tuning parameter to incentivize the agent to achieve the objective faster, as it ‘discounts’ the future value in state (St + 1)


￼





THE TAXI PROBLEM SPECIFICATION :

* Starting at a random state, our job is to get the taxi to the passenger’s location, pick up the passenger and drive to the destination, drop the customer, and then the episode ends.
* There are 4 designated locations in the grid indicated by Red — 0 , Green — 1, Yellow — 2, and Blue — 3, the blue letter correspond to pick up location and purple letter indicate the drop off location. The solid lines indicate walls that the taxi cannot pass, whereas the filled rectangle is the taxi, when it is yellow it is empty and when it is green it is carrying a passenger.

￼

* Each state is defined by a 4 entries tuple: （taxi_row, taxi_col, passenger_location, destination). For example, the image shows state (2,3,2,0), which means we are at position row index 2 (note that python index start at 0 so this means row 3), and column index 3, the passenger is at Yellow, encoded by 2 and our destination is red, encoded by 0.
* State Space: We can see that our state space consist of 500 possible states, with 25 possible taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations
* Action space: There are 6 discrete deterministic actions: 0 — move south, 1 — move north, 2 — move east, 3 — move west, 4 — pickup passenger , 5 — drop off passenger
* Rewards: Except for delivering the passenger with gets a reward of +20, each extra step has a penalty of R=-1, executing “pickup” and “drop-off” actions illegally results in R=-10

ANACONDA, 



THE CODE AS GIVEN BELOW :


import sys	
	from contextlib import closing
	from io import StringIO
	from gym import utils
	from gym.envs.toy_text import discrete
	import numpy as np
	
	MAP = [
	"+---------+",
	"|R: | : :G|",
	"| : | : : |",
	"| : : : : |",
	"| | : | : |",
	"|Y| : |B: |",
	"+---------+",
	]
	
	
	class TaxiEnv(discrete.DiscreteEnv):
	"""
	The Taxi Problem
	from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
	by Tom Dietterich
	
	Description:
	There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
	
	Observations:
	There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.
	
	Passenger locations:
	- 0: R(ed)
	- 1: G(reen)
	- 2: Y(ellow)
	- 3: B(lue)
	- 4: in taxi
	
	Destinations:
	- 0: R(ed)
	- 1: G(reen)
	- 2: Y(ellow)
	- 3: B(lue)
	
	Actions:
	There are 6 discrete deterministic actions:
	- 0: move south
	- 1: move north
	- 2: move east
	- 3: move west
	- 4: pickup passenger
	- 5: drop off passenger
	
	Rewards:
	There is a default per-step reward of -1,
	except for delivering the passenger, which is +20,
	or executing "pickup" and "drop-off" actions illegally, which is -10.
	
	Rendering:
	- blue: passenger
	- magenta: destination
	- yellow: empty taxi
	- green: full taxi
	- other letters (R, G, Y and B): locations for passengers and destinations
	
	state space is represented by:
	(taxi_row, taxi_col, passenger_location, destination)
	"""
	metadata = {'render.modes': ['human', 'ansi']}
	
	def __init__(self):
	self.desc = np.asarray(MAP, dtype='c')
	
	self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
	
	num_states = 500
	num_rows = 5
	num_columns = 5
	max_row = num_rows - 1
	max_col = num_columns - 1
	initial_state_distrib = np.zeros(num_states)
	num_actions = 6
	P = {state: {action: []
	for action in range(num_actions)} for state in range(num_states)}
	for row in range(num_rows):
	for col in range(num_columns):
	for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
	for dest_idx in range(len(locs)):
	state = self.encode(row, col, pass_idx, dest_idx)
	if pass_idx < 4 and pass_idx != dest_idx:
	initial_state_distrib[state] += 1
	for action in range(num_actions):
	# defaults
	new_row, new_col, new_pass_idx = row, col, pass_idx
	reward = -1  # default reward when there is no pickup/dropoff
	done = False
	taxi_loc = (row, col)
	
	if action == 0:
	new_row = min(row + 1, max_row)
	elif action == 1:
	new_row = max(row - 1, 0)
	if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
	new_col = min(col + 1, max_col)
	elif action == 3 and self.desc[1 + row, 2 * col] == b":":
	new_col = max(col - 1, 0)
	elif action == 4:  # pickup
	if (pass_idx < 4 and taxi_loc == locs[pass_idx]):
	new_pass_idx = 4
	else:  # passenger not at location
	reward = -10
	elif action == 5:  # dropoff
	if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
	new_pass_idx = dest_idx
	done = True
	reward = 20
	elif (taxi_loc in locs) and pass_idx == 4:
	new_pass_idx = locs.index(taxi_loc)
	else:  # dropoff at wrong location
	reward = -10
	new_state = self.encode(
	new_row, new_col, new_pass_idx, dest_idx)
	P[state][action].append(
	(1.0, new_state, reward, done))
	initial_state_distrib /= initial_state_distrib.sum()
	discrete.DiscreteEnv.__init__(
	self, num_states, num_actions, P, initial_state_distrib)
	
	def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
	# (5) 5, 5, 4
	i = taxi_row
	i *= 5
	i += taxi_col
	i *= 5
	i += pass_loc
	i *= 4
	i += dest_idx
	return i
	
	def decode(self, i):
	out = []
	out.append(i % 4)
	i = i // 4
	out.append(i % 5)
	i = i // 5
	out.append(i % 5)
	i = i // 5
	out.append(i)
	assert 0 <= i < 5
	return reversed(out)
	
	def render(self, mode='human'):
	outfile = StringIO() if mode == 'ansi' else sys.stdout
	
	out = self.desc.copy().tolist()
	out = [[c.decode('utf-8') for c in line] for line in out]
	taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
	
	def ul(x): return "_" if x == " " else x
	if pass_idx < 4:
	out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
	out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
	pi, pj = self.locs[pass_idx]
	out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
	else:  # passenger in taxi
	out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
	ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)
	
	di, dj = self.locs[dest_idx]
	out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
	outfile.write("\n".join(["".join(row) for row in out]) + "\n")
	if self.lastaction is not None:
	outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
	else:
	outfile.write("\n")
	
	# No need to return anything for human
	if mode != 'human':
	with closing(outfile):
	return outfile.getvalue()
  
  # ------------- X ---------------- X --------------------
  
  Information learned till now :

1. Good understanding of reinforcement learning -> Simulation in order to get the rewards, environment to get random states, if the action was taken by the model, what is the action, how to change the algorithm.
2. Environment : possible states repository. Each state for example, a chess board, multiple possibilities for a chess board. Has the ability to tell us if the certain action was taken, what was the reward. This is complex and hence, we have gym where we get some ready made environment.
3. For this, we have the MC simulation etc —> this is the environment.
4. We have to MAKE A MODEL TO BE ABLE TO TELL us the action to tell us the reward to be maximized.
5. In this example, the Deep Q algorithm is used with a Deep neural network. To tell the action, it has a standard architecture.
6. We create activation layers, have bash layers etc.
7. There are 3 convolutional neural networks, after this, we do a flattening layer and then apply a connected layer and for that, we use the number filters to be equal to the number of actions.
8. FUNDA : Has to be some mechanism by which the algorithm computes the deviation of the reward, optimize the loss(the deviation from the maximum reward)
9. First, we have the actual reward. If we have taken an action. USE THE FORMULA!!
10. NEXT : TO FINALIZE THE PROBLEM!!
11. Have a couple of ideas : We will have Bonsai : Firstly, get an azure subscription for Microsoft. Then we can create the Microsoft project bonsai : will have a user case. We have two elements. In bonsai, we will have the deep neural networks simplified. We just need to know the objective.


———> We need the environment, bonsai gives us a simulation, we need a problem to create the simulation environment. We can make it in python(sample codes available)
Next step : will take the python code to create the environment. WE WILL CREATE AN ENVIRONMENT -> LINK SHARED.

BONSAI : Create the instance and the environment to be made by python. Once we have this knowledge.

Lastly, use incling  simulation project bonsai

# MORE NOTES :

CREATING THE PYTHON REINFORCEMENT LEARNING ENVIRONMENT USING A SIMPLE BALL GAME :

Reinforcement learning is a branch of Machine learning where we have an agent and an environment. The environment is nothing but a task or simulation and the Agent is an AI algorithm that interacts with the environment and tries to solve it.
In the diagram below, the environment is the maze. The goal of the agent is to solve this maze by taking optimal actions.

￼

It is clear from the diagram that how agent and environment interact with each other. Agent sends action to the environment, and the environment sends the observation and reward to the agent after executing each action received from the agent. Observation is nothing but an internal state of the environment. And reward signifies how good the action was. It is going to get clearer as we proceed through the blog.
So we need 2 things in order to apply reinforcement learning.
* 		Agent: An AI algorithm.
* 		Environment: A task/simulation which needs to be solved by the Agent.
An environment interacts with the agent by sending its state and a reward. Thus following are the steps to create an environment.
* 		Create a Simulation.
* 		Add a State vector which represents the internal state of the Simulation.
* 		Add a Reward system into the Simulation.
Let’s start building the environment now.





CREATING THE ENVIRONMENT :


We will build a game in python and then add -> the STATE vector and the REWARD system in it. 

Through this we will create our first reinforcement environment :

THE game is going to be based on a paddle and a ball game -> In this game, the paddle will be moving and the ball will hit on the paddle in order to ket the game going on, otherwise, it will miss and the game will lose.

The game will use an inbuilt turtle module in python.

TURTLE : Provides an easy and inbuilt interface to build and moves different shapes based on the commands that are given in the code.

The code for creating the background given below as well as in the sublime text editor in the form of a python .py file.


mport turtle

win = turtle.Screen()    # Create a screen
win.title('Paddle')      # Set the title to paddle
win.bgcolor('black')     # Set the color to black
win.tracer(0)
win.setup(width=600, height=600)   # Set the width and height to 600

while True:
     win.update()        # Show the screen continuously 

paddle = turtle.Turtle()    # Create a turtle object
paddle.shape('square')      # Select a square shape
paddle.speed(0)             
paddle.shapesize(stretch_wid=1, stretch_len=5)   # Streach the length of square by 5 
paddle.penup()
paddle.color('white')       # Set the color to white
paddle.goto(0, -275)        # Place the shape on bottom of the screen

# Ball
ball = turtle.Turtle()      # Create a turtle object
ball.speed(0)
ball.shape('circle')        # Select a circle shape
ball.color('red')           # Set the color to red
ball.penup()
ball.goto(0, 100)           # Place the shape in middle


# Paddle Movement
def paddle_right():
    x = paddle.xcor()        # Get the x position of paddle
    if x < 225:
        paddle.setx(x+20)    # increment the x position by 20

def paddle_left():
    x = paddle.xcor()        # Get the x position of paddle
    if x > -225:
        paddle.setx(x-20)    # decrement the x position by 20

# Keyboard Control
win.listen()
win.onkey(paddle_right, 'Right')   # call paddle_right on right arrow key
win.onkey(paddle_left, 'Left')     # call paddle_left on right arrow key


ball.dx = 3   # ball's x-axis velocity 
ball.dy = -3  # ball's y-axis velocity 

while True:   # same loop

     win.update()

     ball.setx(ball.xcor() + ball.dx)  # update the ball's x-location using velocity
     ball.sety(ball.ycor() + ball.dy)  # update the ball's y-location using velocity


# Ball-Walls collision   
if self.ball.xcor() > 290:    # If ball touch the right wall
    self.ball.setx(290)
    self.ball.dx *= -1        # Reverse the x-axis velocity

if self.ball.xcor() < -290:   # If ball touch the left wall
    self.ball.setx(-290)
    self.ball.dx *= -1        # Reverse the x-axis velocity

if self.ball.ycor() > 290:    # If ball touch the upper wall
    self.ball.sety(290)
    self.ball.dy *= -1        # Reverse the y-axis velocity

# Ball-Paddle collision
if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
    self.ball.dy *= -1
        
# Ball-Ground collison            
if self.ball.ycor() < -290:   # If ball touch the ground 
    self.ball.goto(0, 100)    # Reset the ball position    


    hit, miss = 0, 0

# Scorecard
score = turtle.Turtle()   # Create a turtle object
score.speed(0)
score.color('white')      # Set the color to white
score.hideturtle()        # Hide the shape of the object
score.goto(0, 250)        # Set scorecard to upper middle of the screen
score.penup()
score.write("Hit: {}   Missed: {}".format(hit, miss), align='center', font=('Courier', 24, 'normal'))


def step(self, action):
   
   reward, done = 0, 0

   if action == 0:         # if action is 0, move paddle to left  
       paddle_left()
       reward -= .1        # reward of -0.1 for moving the paddle

   if action == 2:         # if action is 2, move paddle to right 
       paddle_right()
       reward -= .1        # reward of -0.1 for moving the paddle

   run_frame()   # run the game for one frame, reward is also updated inside this function
   
   # creating the state vector
   state = [paddle.xcor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy]
   
   return reward, state, done



import turtle	
	
	win = turtle.Screen()    # Create a screen
	win.title('Paddle')      # Set the title to paddle
	win.bgcolor('black')     # Set the color to black
	win.tracer(0)
	win.setup(width=600, height=600)   # Set the width and height to 600
	
	while True:
	win.update()        # Show the scree continuously


This given code above is the blank black screen where the game will take place which will act as an environment.

The code given below adds the paddle and the ball to the center on this environment created by us :


# Paddle	
	paddle = turtle.Turtle()    # Create a turtle object
	paddle.shape('square')      # Select a square shape
	paddle.speed(0)
	paddle.shapesize(stretch_wid=1, stretch_len=5)   # Streach the length of square by 5
	paddle.penup()
	paddle.color('white')       # Set the color to white
	paddle.goto(0, -275)        # Place the shape on bottom of the screen
	
	# Ball
	ball = turtle.Turtle()      # Create a turtle object
	ball.speed(0)
	ball.shape('circle')        # Select a circle shape
	ball.color('red')           # Set the color to red
	ball.penup()
	ball.goto(0, 100)           # Place the shape in middle

Through the code given in the sublime text editor, we will see how different aspects of the environment are functioned together and how it leads to the addition of the state and vector sections leading to a creation of a fully functioning reinforcement environment!



State vector and Reward system
We feed the state vector to our AI agent and agent choose an action based on that state. The state vector should contain valuable information. The goodness of the action taken by an agent depends on how informative the state vector is.
I created a state vector that contains the following information.
* 		Position of the paddle in the x-axis
* 		Position of the ball in the x and y-axis
* 		The velocity of the ball in x and y-axis
Following is the reward system I have implemented.
* 		Give a reward of +3 if the ball touches to the paddle
* 		Give a reward of -3 if the ball misses the paddle.
* 		Give a reward of -0.1 each time paddle moves, so that paddle does not move unnecessary.
We also have to implement an action space. The agent will choose one of the actions from the action space and send it to the environment. Following is the action space I implemented.
* 		0 - Move the paddle to left.
* 		1 - Do nothing.
* 		2 - Move the paddle to the right.
The agent will send one of these numbers to the environment, and the environment performs the action corresponding to that numbers.



Update :

1. Ran the environment for the python code given above adding in the state and the vector functions and getting their rewards based on the conditions given above and therefore, executing the paddle, ball game in the field of this project bonsai for the main goal of creating the reinforcement learning environment.
2. Inkling : Studying the language for this specific project Bonsai and going through the website simultaneously.

THE NOTES FOR INKLING ARE AS GIVEN BELOW :

Basics:

The Inkling code defines what and how and the elements used to be in order to teach your AI. The basic elements of Inkling are as given below :

1. Comments : User-defined annotations for example :

# Distances (meters)
const RadiusOfPlate = 0.1125

2. Identifiers : User-defined name for objects and types for example :
* start with an alphabetical character or underscore
* contain printable Unicode characters that are not reserved characters in Inkling.

const Velocity = 5.0
var _max_percent: number = 100
const Rotate90Degrees = true


Inkling provides the following escape characters for identifiers that must use unsupported characters:
* backticks (`) can be used to escape whitespace.
* backslash (\) can be used to escape backticks and backslashes.
For example:

const `1 Microsoft Way` = true
var `Service Name`: string = "Bonsai"
const `Don't \` do this` = "This is a legal, but very bad identifier"


INKLING TYPES :

This language is a statically typed language, which means that each of the variable, constant, and parameter must have a well-defined type.

The types are : number, string, Array, Structure.

FOR THIS WE USE THE STATEMENT TYPE TO CORRESPOND THE GIVEN VARIABLE OR CONSTANT OR GLOBAL INSTANCE TO THE TYPE OF THE PRIMITIVE DATE TYPE :

type Type1 number
type Type2 number

EXAMPLES OF USER DEFINED TYPES :

We use the “type” statement to create the custom types. Custom types can be used anywhere a type literal would be used.

For example :

type Height number

type Coordinate {X: number, Y: number}

type TicTacToeValue number<Blank=0, X=1, O=2>

type TicTacToeBoard TicTacToeValue[3][3]

Implicit casting :

The Inkling compiler always assumes the result of a mathematical operation is a number. As a result, the compiler cannot statically evaluate mathematical operations that use variables.

Assigning a number to an enumerated constraint like number<1, 2> is not allowed because the compiler cannot verify that the result belongs to the enumerated set.
Casting to a range constraint like number<1 .. 2> is allowed, but the compiler will warn that the result may be rounded to fit within the range.
For more information on range and enumeration constraints, see the Number type.

Examples of valid assignments
The following example works because the value of OneOrTwo is guaranteed to fit within the range assigned to ZeroTo100.
Inkling

Copy

var OneOrTwo: number<1, 2> = 1 + 1  # Enumerated constraint
var ZeroTo100: number<0..100> = 20  # Range constraint

ZeroTo100 = OneOrTwo
The following example also works, but generates a warning that the assignment may require rounding. The compiler cannot evaluate the value of 1 + OneOrTwo statically to ensure it results in a valid range value for ZeroTo2.
Inkling

Copy

var OneOrTwo: number<1, 2>  = 1 + 1
var ZeroTo2: number<0 .. 2> = 1 + OneOrTwo

Examples of invalid assignments
The following example generates a compiler error. The value of ZeroTo100 may exceed the permitted values of the range type of ZeroTo50.
Inkling

Copy

var ZeroTo50: number<0..50> = 50
var ZeroTo100: number<0..100> = 20

ZeroTo50 = ZeroTo100
For more details about casting compatibility, see the relevant type documentation.



ARRAYS IN INKLING :

Inkling denotes array literals with square brackets surrounding a comma-separated list of one or more values. All values in an array literal must be of the same type but can be specified as an expression. For example:


[Math.Pi, 50, (Math.Pi * (50 ** 2))]


Array types :


Array types must be initialized with their size and value types at creation, so Inkling does not provide an explicit array type name. Use the type token along with an array size and type definition to declare a user-defined array type. For example,


type ScreenGrid number[100][100] # A 2D array of numbers
type ValidStates string[5]       # A 1D array of strings


This example above shows above how the array either 1D or 2D will contain the number of values or the size of the array!


A type can be used in the form of an enumeration to check the value



# Creates an array type that is a 100 x 100 matrix of values between 1 and 1024
type ScreenGrid number<1 .. 1024>[100][100]

# Creates an array type that is a set of 5 values where the strings are all
# one of the enumberated values
type ValidStates string<"IN_PROGRESS", "FINISHED", "PENDING">[5]


Type equivalence and casting

Equivalence
An array type, X, is equivalent to array type Y if:
* 		the type stored in X are equivalent to the type stored in Y.
* 		the number of dimensions in X and Y are the same.
* 		the size of each array dimension in X is the same as the corresponding dimension in Y.

Equivalence examples
The following types are not equivalent, even though they contain the same types, because the arrays have a different number of dimensions:
Inkling

Copy

type SmallScreen number<1 .. 1024>[100][100]
Inkling

Copy

type MarqueeScreen number<1 .. 1024>[100]
The following types are not equivalent, even though they contain the same types and have the same number of dimensions, because the size of the dimensions are not the same:
Inkling

Copy

type SmallScreen number<1 .. 1024>[100][100]
Inkling

Copy

type LargeScreen number<1 .. 1024>[1000][1000]

Casting
An array type, X, is castable to array type Y if:
* 		the type stored in X is castable to the type stored in Y.
* 		the number of dimensions in X and Y are the same.
* 		the size of each array dimension in X is the same as the corresponding dimension in Y.

# CartPole example :

Cartpole-V1 Example :

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.


Explain the game!!

1. First we get all of our imports down -> We import gym for the reinforcement learning environment, then we import time, tensor flow, tflearn, NumPy as np, matplotlib, seaborn, inline
2. Next step : Make our environment. Gym has the ability to make a lot of different environments and one of them in Cartpole-v1, there is Minecraft, and other complex games as well.
3. Env = gym.make(“CartPole-v0”) this piece of code makes the particular environment for the cart pole example.
4. We use this to create the first environment -> This will be used -> obs = env.reset()
5. Obs -> Has four values, it has to do with where the cart is and where the pole is and which way it is going!
6. We render this then using env.render, and see our first frame in front of us.
7. We can make a while loop -> while true, time.sleep(0.005secs)
8. We are not telling it to take any of the actions but to advance the program, we take an action by:
9. Env.step(0) 0-> left, 1 is right.
10. We can also check when it is done. To check when it done, the step function returns the next observation, the next reward, positive if the pole stays up and negative if it falls.

11. Obs, r, d, _ = env.step(0)

If d = true we can break and then see that in the frame, as we say 0, the cart moves to the left and starts to fall. (20 degrees to lose)
We are going to keep running through this until the maximum of 200 frames because that is where we win.

In the while loop ->

While true;
For I in range(200);

-> We will have to reset the environment every time we start the game and see that it keeps losing!!

(We can print the obs to check the reward etc by doing -> print(obs).print(r), print(d) d-> d is whether or not it is done.


We can create several of these functions like the saving of the name of the model, the running of the model, to print the reward, next state, done or not, an also to test the function using the actions and checking later whether the game is done or not based on the amount of frames taken into consideration.


# PROJECT THEORY WORK (PROBLEM STATEMENT, CONCEPTS AND APPROACHES)

Phase 1 : Getting the features, how much movement, volume, essentially have the risk, with the cap stock, we take all of this and in the second phase, getting data, strengthening features, creating algorithm.

We give our environment a state, and we give a price to the stock, we have an algorithm capital and tell it to have to give it a trade and return the actions and at the end, the environment will tell us the reward. Overly complex is not there, if there is an automated environment on the internet!


CREATE A PROBLEM STATEMENT BASED ON THE PREVIOUS PROBLEMS ON THE INTERNET!!



FIRST APPROACH :

DEEP REINFORCEMENT LEARNING FOR AUTOMATED STOCK LEARNING :



-> This can be done using reinforcement learning to trade multiple stocks in order to minimize the risk and maximize the reward functions through the use of python and AI gym.

-> OVERVIEW :


1. One can hardly overestimate the crucial role of the stock trading strategies play in investment.
2. We need to take into consideration the fact that profitable automated stock trading strategy is something that is vital to the investment in big companies and funds and it is applied and used to optimize the capital allocation and maximize the investment performance, such as the expected return.
3. Return maximization can be done on the basis of the estimates of the potential return and risk ——> However, it is difficult to design a profitable strategy in a complex and a dynamic stock company market.
4. We need a winning strategy, in order to maximize the return by minimizing the risks —— AND WE CAN DO THIS BY :

Coming up with a deep reinforcement learning scheme that automatically learns the stock trading strategy and in order to maximize the investment return!

Q. What will we focus on and how will we do this?

1. SOLUTION : ENSEMBLE DEEP REINFORCEMENT LEARNING TRADING STRATEGY CAN BE USED.

Q1. What is Ensemble Deep Reinforcement trading strategy?

ANSWER : 

This strategy will use the combination of three essential actor-critic based algorithms : proximal policy optimization (PPO), Deep Deterministic Policy Gradient (DDPG), and Advantage Actor Critic (A2C).

These three will be used simultaneously in order to adapt to the market conditions and different market environments.


The performance and the reward strategy of the trading agent in the market environment with different reinforcement algorithms will be evaluated and calculated using the SHAPE RATIO AND THE COMPARED WITH BOTH OF THE ‘DOW JONES INDUSTRIAL AVERAGE INDEX’ AND the traditional variance portfolio allocation strategy.




￼




Questions and observations to consider before going on to the problem statement :

1. Why do we use DEEP REINFORCEMENT LEARNING FOR THE USE OF TRAINING THE STICK TRADING MACHINE?

We use this for several advantages :

* DRL doesn’t need large labeled training datasets. This is a significant advantage since the amount of data grows exponentially today, it becomes very time-and-labor-consuming to label a large dataset.
* DRL uses a reward function to optimize future rewards, in contrast to an ML regression/classification model that predicts the probability of future outcomes.
* The goal of stock trading is to maximize returns, while avoiding risks. DRL solves this optimization problem by maximizing the expected total reward from future actions over a time period.
* Stock trading is a continuous process of testing new ideas, getting feedback from the market, and trying to optimize the trading strategies over time. We can model stock trading process as Markov decision process which is the very foundation of Reinforcement Learning.
* Deep reinforcement learning algorithms can outperform human players in many challenging games. 
* Return maximization as trading goal: by defining the reward function as the change of the portfolio value, Deep Reinforcement Learning maximizes the portfolio value over time.
* The stock market provides sequential feedback. DRL can sequentially increase the model performance during the training process.
* The exploration-exploitation technique balances trying out different new things and taking advantage of what’s figured out. This is difference from other learning algorithms. Also, there is no requirement for a skilled human to provide training examples or labeled samples. Furthermore, during the exploration process, the agent is encouraged to explore the uncharted by human experts.
* Experience replay: is able to overcome the correlated samples issue, since learning from a batch of consecutive samples may experience high variances, hence is inefficient. Experience replay efficiently addresses this issue by randomly sampling mini-batches of transitions from a pre-saved replay memory.
* Multi-dimensional data: by using a continuous action space, DRL can handle large dimensional data.
* Computational power: Q-learning is a very important RL algorithm, however, it fails to handle large space. DRL, empowered by neural networks as efficient function approximator, is powerful to handle extremely large state space and action space.




THE THREE ACTOR-CRITIC ALGORITHMS WE WILL BE USING :

* A2C: A2C is a typical actor-critic algorithm. A2C uses copies of the same agent working in parallel to update gradients with different data samples. Each agent works independently to interact with the same environment.
* PPO: PPO is introduced to control the policy gradient update and ensure that the new policy will not be too different from the previous one.
* DDPG: DDPG combines the frameworks of both Q-learning and policy gradient, and uses neural networks as function approximators.




PROBLEM :

ENSEMBLE STRATEGY:

Ensemble learning : Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the (classification, prediction, function approximation, etc.)






1. A2C is a typical actor-critic algorithm which we use as a component in the ensemble method. A2C is introduced to improve the policy gradient updates. A2C utilizes an advantage function to reduce the variance of the policy gradient. Instead of only estimates the value function, the critic network estimates the advantage function. Thus, the evaluation of an action not only depends on how good the action is, but also considers how much better it can be. So that it reduces the high variance of the policy networks and makes the model more robust.A2C uses copies of the same agent working in parallel to update gradients with different data samples. Each agent works independently to interact with the same environment. After all of the parallel agents finish calculating their gradients, A2C uses a coordinator to pass the average gradients over all the agents to a global network. So that the global network can update the actor and the critic network. The presence of a global network increases the diversity of training data. The synchronized gradient update is more cost-effective, faster and works better with large batch sizes. A2C is a great model for stock trading because of its stability.

2. DDPG is an actor-critic based algorithm which we use as a component in the ensemble strategy to maximize the investment return. DDPG combines the frameworks of both Q-learning and policy gradient, and uses neural networks as function approximators. In contrast with DQN that learns indirectly through Q-values tables and suffers the curse of dimensionality problem, DDPG learns directly from the observations through policy gradient. It is proposed to deterministically map states to actions to better fit the continuous action space environment.

3. We explore and use PPO as a component in the ensemble method. PPO is introduced to control the policy gradient update and ensure that the new policy will not be too different from the older one. PPO tries to simplify the objective of Trust Region Policy Optimization (TRPO) by introducing a clipping term to the objective function.The objective function of PPO takes the minimum of the clipped and normal objective. PPO discourages large policy change move outside of the clipped interval. Therefore, PPO improves the stability of the policy networks training by restricting the policy update at each training step. We select PPO for stock trading because it is stable, fast, and simpler to implement and tune.



# PROJECT PROMPT :

1. How to use reinforcement strategy called Q-Learning to simultaneously predict prices for three stocks in a portfolio using several data points.
2. Stock and future trading are heavily used and these automated machines have earned billions.
3. Have been working for more than 2 decades.
4. AI in finance is the most mature application of AI in any industry.

How can we use the starting data set and the instructions to start the potentially generated problem :

1. We will deal with 3 csv files.
2. Each will contain stock prices for the past years from Microsoft, nse, Deloitte.
3. The data set can be created using api.
4. We should build an algorithm that can learn from this data.
5. We have historical data and access to real time data via the api.
6. Real time environment with time as a dimension.


We can frame our problem using a Markov decision process!

1. This will have states, agents and rewards. Our Agent will be able to execute an action in this environment and the action space will be a simple.


FOR ANY ACTION THERE WILL BE THREE ACTIONS TO CHOOSE FROM : BUY, SELL OR HOLD!

1. Buy : This will buy as much as stock as possible based on the current stock prices and the cash we have.
2. Sell : Will sell all the money from the stock and add the cash to the balance. If we are buying multiple stocks, we will equally distribute cash for each of those.
3. Hold : Will do nothing.

The number of actions will be : 3^n where n will be the number of stocks in our data or our portfolio.

At every time step, our agent will be at a state and an action will take place.

Problem framework creation in simple terms :


1. A: ACTION

(BUY, SELL, HOLD)
3^n, Where n = number of stocks present in our portfolio.

2. S: STATE

Number of stocks we own, taking into consideration out current stock price and the account balance.

STATE EXAMPLE : If we own 20 shares of Microsoft, 50 shares of google and 30 shares of facebook and then have 2000 dollars in our balance, 

We can represent our state as an array which contains the amount and prices of the stocks and the account balances.

REWARD : ?? TO CALCULATE.

1. We are dealing with a partially observable Markov Decision process (POMDP) and because of this, we do not know what the full state look like, and the reward looks like.

If we had the state functions and the reward functions in hand, we could use the dynamic approach of solving this problem but since we do not, we use another approach through which we will compute the optimal policy because we will learn the effect of taking an action in a space :


WE MUST USE A MODEL FREE TECHNIQUE :

Q-Learning :
￼
1. We first denote a function Q(of S and A)
2. This represented the maximum discounted future reward in state S
3. It is called the action value function since it measures the value of the action
4. This action can be the highest account balance we can have at the end of a training episode after A in state S.
5. Actions : BUY, SELL, HOLD. 



Q. How can we determine the Q-Value?

A. We can express the value of S AND A in terms of value for the next state
￼
USE THE BELLMAN’S EQUATION :

The equation says that the maximum future reward in this state and action is the immediate reward plus the future reward for the next state.

This lets us represent the value of the given state in terms of the values for the next state which makes it possible for us to get the q-function.


CODE :

CONTENTS OF THE CODE MUST BE :

* agent.py: a Deep Q learning agent
* envs.py: a simple 3-stock trading environment
* model.py: a multi-layer perceptron as the function approximator
* utils.py: some utility functions
* run.py: train/test logic
* requirement.txt: all dependencies
* data/: 3 csv files with IBM, MSFT, and QCOM stock prices from Jan 3rd, 2000 to Dec 27, 2017 (5629 days).


HENCE, THE ALGORITHM :


Substitute the values of the given variables into the given Q-Learning algorithm template.

























