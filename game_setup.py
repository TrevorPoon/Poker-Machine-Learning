from pypokerengine.api.game import setup_config, start_poker

from Player.random_player import RandomPlayer
from Player.rl_player_2 import EmulatorPlayer
from Player.All_in_Player1 import All_in_Player_1
from Player.All_in_Player2 import All_in_Player_2
# from my_model import MyModel if you need to reference it here as well.



# Game configuration
config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=5)

# Adding players
config.register_player(name='player1', algorithm=All_in_Player_1())
config.register_player(name='player2', algorithm=All_in_Player_2())

# Start the game
results = start_poker(config, verbose=1)  # Set verbose=1 for more game details

# Print the final results
print(results)