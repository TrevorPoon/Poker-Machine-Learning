from pypokerengine.api.game import setup_config, start_poker

from Player.random_player import RandomPlayer
from Player.rl_player import EmulatorPlayer
# from my_model import MyModel if you need to reference it here as well.

# Game configuration
config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)

# Adding players
config.register_player(name='player1', algorithm=RandomPlayer())
config.register_player(name='player2', algorithm=RandomPlayer())
config.register_player(name='player3', algorithm=EmulatorPlayer())

# Start the game
results = start_poker(config, verbose=1)  # Set verbose=1 for more game details

# Print the final results
print(results)