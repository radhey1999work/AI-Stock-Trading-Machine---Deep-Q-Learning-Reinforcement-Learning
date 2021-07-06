import gym
from gym import spaces
# Represents initializing the random number generator
from gym.utils import seeding
import numpy as np
# Represents the ability to work on the iterators
import intertools

"""
This class represents the environment in which the trading of the 3 stocks from the portfolio take place. 
Represents a three stock trading environment from NSE, deloitte, and Axis Bank, India.
The trading environment contains two aspects -> 
a State, which involved the number of stocks from one of the companies owned, the current stock bank balance, and the amount of stock price at hand.
an Action, which involves three parts: Buy(0), Sell(1) and None(2)
"""

class StockPriceTradingEnv(gym.env):


"""
A Stock trading environment is an environment which has:
State: [current cash balance, current balance of the stock prices, # of stocks owned by that company]
	- In the case of evaluating the state, we will calculate the cash at hand at each of the step after each of the 
	action performed.
	- Array of the state with length of n_stock * 2 + 1
	- We use the close (Last price) for each of the stock to be evaluated
	- Price is updated every action step
Action: Buy(0), Sell(1) and None(2)
	- When the option is Buy, we buy as many shares or cash as permitted
	- When the option is Sell, we sell all of the shares
	- When in the case of buying multiple stock shares, the cash is to be equally distributed in order to 
	utilize the balance
	"""

def _initData_(self, givenData, investing_amount=1000000):
	# This function represents the data to be taken into consideration 
	SELF.trading_price_history = np.around(givenData)
	# This rounds the integer to reduce state space
	SELF.n_stock, SELF.n_step = SELF.trading_price_history.shape
	# This represents the number of stocks and steps which is equal to the sice of the trading history
	# data

	# Represents the current action space
	SELF.action_space = spaces.Discrete(3**SELF.n_stock)

	# Represents the instances in this trading environment
	SELF.number_of_stocked_step = NONE
	SELF.stock_price = NONE
	SELF.current_given_step = NONE
	SELF.current_cash_at_hand = NONE
	SELF.investing_amount = investing_amount

    # observation space: give estimates in order to sample and build scaler
    stock_maximum = SELF.stock_price_history.max(axis=1)
    stock_price_range = [[0, investing_amount * 2 // mx] for mx in stock_max_price]
    price_range = [[0, mx] for mx in stock_max_price]
    cash_in_hand_range = [[0, investing_amount * 2]]
    SELF.given_observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)



    # seed and start
    SELF._seed()
    SELF._reset()



    # Represents saving the state of the action in this function
    def _seed(self, seed=NONE):
    	SELF.np_random, seed = seeding.np_random(seed)
    	return [seed]


    # Represents all of the values that are resetted to the initial values
    def _reset(self):
    	SELF.stock_price = SELF.stock_price_history[;, SELF.current_given_step] # Updated price
    	SELF.current_given_step = 0;
    	SELF.current_stock_owned = [0] * SELF.n_stock
    	SELF.current_cash_at_hand = SELF.investing_amount
    	return SELF._return_observ()



    # Represents the function that takes into consideration the stepping through actions
    def _step_action_taken(self, action_taken):
    	# asserts the function if the condition is true for when the given action space contains the given action
    	# to be taken
    	assert SELF.action_space.contains(action_taken)
    	# Represents taking in the previous value and updating it
    	given_previous_amount = SELF._get_cur_val()
    	# Represents the incrementation of the current step by one evey time the action step is taken
    	SELF.current_given_step =+ 1
    	SELF._trade_it(action_taken)
    	# Gets the value of the current trading cash balance evaluated
    	current_value = SELF._get_cur_val()
    	returned_reward = current_value - given_previous_amount
    	displayed_description = {'current updated value': current_value}
    	return SELF._return_observ(), returned_reward, displayed_description




    # Represents the function which returns the observation 
    def _return_observ(self):
    	# Represents the empty array list to append to the append
    	given_observ = []
    	# Represents extending and adding the stock price to existing array list
    	given_observ.extend(SELF.stock_price)
    	# Represents extending and adding the current stock owned to existing array list
    	given_observ.extend(list(SELF.current_stock_owned))
    	# Represents extending and adding the current cash in our hands to the existing array list
    	given_observ.append(SELF.current_cash_at_hand)
    	# Represents returning the list of observations
    	return given_observ



    # Represents the function which calculates the value of the entire balance in our trading portfolio
    def _get_cur_val(self):
    	# Need to check and test to see if the formula given below works for calculating the
    	# current total value
    	return np.sum(SELF.current_stock_owned * SELF.stock_price) + SELF.current_cash_at_hand



    # Represents the final potential function for the environment which focusses on doing the Trading job containing
    # three of the actions (buy, sell, or hold) and takes into consideration the state and the updated traded price from the stock
    def _trade_it(self, action_taken):
    	"""
    	We will have empty lists of a buying and a selling array list of stock prices to our
    	portfolio
    	"""

    	# This function will contain the 3 options that an action can be taken that are
    	# Buy(0), Sell(1), or None(2)

    	# We can make the use of the map function in order to allow us to process
    	# and transform all of the items in an iterable without using a loop.

    	# Repeat the map from intertools.product([0, 1, 2], repeated=SELF.n_stock))
    	"""
    	^
    	|
    	|
    	|
    	|
    	We use the map function on each of the stock in the given n.stocks and then use 
    	the action (either 0, 1, or 2) to take the action after which, we can compute the
    	given reward and then train the machine accordingly.
    	"""
      ----------------- X -------------------- X --------------------- X -------------------- X -------------------- X -------------------
    	## TO COMPLETE
