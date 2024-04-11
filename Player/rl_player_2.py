from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck

from Models.RL_Model_1 import MyModel
from Models.RL_Model_2 import RLAgent

class EmulatorPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.my_model = MyModel()
        state_size = 10  # Represents number of features in game state representation
        action_size = 4  # Represents number of possible actions

        # Initialize the RL Agent
        self.rl_agent = RLAgent(state_size, action_size)
    # setup Emulator with passed game information
    
    def receive_game_start_message(self, game_info):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        state_representation = self._extract_features(round_state, hole_card)
        # Use the RL agent to select the best action
        action_index = self.rl_agent.select_action(state_representation)
        # Map the action_index to a valid action
        action, amount = self._map_to_valid_action(action_index, valid_actions)
        return action, amount
    
    def _extract_features(self, round_state, hole_card):
    # Example: assigning each street a unique integer
        street = round_state['street']
        street_encoded = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}[street]
        
        # Assuming there is a key in round_state for current stack and max stack
        cur_stack = round_state['current_player_stack']
        max_stack = round_state['max_player_stack']
        stack_normalized = cur_stack / max_stack
        
        # Assuming there is a 'call_amount' in round_state information
        to_call = round_state['call_amount'] / cur_stack
        
        # Assuming there is a 'is_dealer' boolean in round_state
        is_dealer = 1 if round_state['is_dealer'] else 0
        
        # Encode hole cards to numeric values
        # For this example, '2S' would be 2, '3C' would be 4, ... 'AS' would be 52
        card_encode = lambda card: (card.rank_index + 2) * (card.suit_index + 1)
        hole_encoded = [card_encode(card) for card in gen_cards(hole_card)]
        
        # Combine features into a single vector
        state_vector = [street_encoded, stack_normalized, to_call, is_dealer] + hole_encoded
        return state_vector

    def _map_to_valid_action(self, action_index, valid_actions):
        # Check if index is out of bounds for the valid_actions list
        if action_index >= len(valid_actions):
            action_index = action_index % len(valid_actions)
        
        # Retrieve the action information using the valid index
        action_info = valid_actions[action_index]
        
        # Depending on the chosen action, you might validate it further to make sure it's allowed
        # If the action requires a specific amount, make sure to retrieve it
        amount = action_info['amount']
        if action_info['action'] == 'raise':
            amount = amount['min']  # Assuming we always raise the minimum amount
        
        return action_info['action'], amount

    def _setup_game_state(self, round_state, my_hole_card):
        game_state = restore_game_state(round_state)
        game_state['table'].deck.shuffle()
        player_uuids = [player_info['uuid'] for player_info in round_state['seats']]
        for uuid in player_uuids:
            if uuid == self.uuid:
                game_state = attach_hole_card(game_state, uuid, gen_cards(my_hole_card))  # attach my holecard
            else:
                game_state = attach_hole_card_from_deck(game_state, uuid)  # attach opponents holecard at random
        return game_state

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
