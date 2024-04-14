from pypokerengine.api.game import setup_config, start_poker

from Player.random_player import RandomPlayer
from Player.rl_player_2 import EmulatorPlayer
# from my_model import MyModel if you need to reference it here as well.

# Game configuration
config = setup_config(max_round=10000000, initial_stack=10000000, small_blind_amount=10)

# Adding players
config.register_player(name='player1', algorithm=RandomPlayer())
config.register_player(name='Trevor', algorithm=EmulatorPlayer())

# Start the game
results = start_poker(config, verbose=1)  # Set verbose=1 for more game details

# Print the final results
print(results)