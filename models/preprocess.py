import numpy as np

class Preprocess(Object):
	def __init__ (states, actions, rewards):
		raise NotImplementedError

	def get_input ():
		raise NotImplementedError

	def get_q_value_inputs ():
		raise NotImplementedError