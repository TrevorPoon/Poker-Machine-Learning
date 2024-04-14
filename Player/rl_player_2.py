from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck

from Models.RL_Model_Trial import RLAgent

class EmulatorPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        state_size = 10  # Represents number of features in game state representation
        action_size = 4  # Represents number of possible actions

        # Initialize the RL Agent
        # self.rl_agent = RLAgent(state_size, action_size)
        self.rl_agent = RLAgent()
    # setup Emulator with passed game information
    
    def receive_game_start_message(self, game_info):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        state_representation = self._extract_features(valid_actions, round_state, hole_card)
        # Use the RL agent to select the best action
        action_index = self.rl_agent.select_action(valid_actions, state_representation)
        # Map the action_index to a valid action
        # action, amount = self._map_to_valid_action(action_index, valid_actions)

        action, amount = self.rl_agent.select_action(valid_actions, state_representation)
        return action, amount
    
    def _extract_features(self, valid_actions, round_state, hole_card):
        # Example: assigning each street a unique integer
        street = round_state['street']
        street_encoded = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}[street]

        # Get the current player's stack from the round_state
        current_player_uuid = self.uuid
        player_stacks = {player['uuid']: player['stack'] for player in round_state['seats']}
        cur_stack = player_stacks[current_player_uuid]
        
        # Normalizing stacks relative to the big blind
        big_blind_amount = round_state['small_blind_amount'] * 2
        stack_normalized = cur_stack / big_blind_amount
        
        # Calculate amount to call
        main_pot = round_state['pot']['main']['amount'] / big_blind_amount

        to_call = valid_actions[1]['amount'] / big_blind_amount

        try:
            side_pot = round_state['pot']['side']['amount'] / big_blind_amount
        except:
            side_pot = 0
        
        # Determine if the current player is dealer
        position = round_state['next_player']

        # Extracting hole card ranks and suits
        hole_card_ranks = [self._encode_rank(card[1]) for card in hole_card]
        hole_card_suits = [self._encode_suit(card[0]) for card in hole_card]
        
        # Extracting community card ranks and suits
        community_cards = round_state['community_card']
        community_card_ranks = [self._encode_rank(card[1]) for card in community_cards]
        community_card_suits = [self._encode_suit(card[0]) for card in community_cards]
        
        # Add placeholder zeros for community cards that haven't been dealt
        total_community_cards = 5  # There are 5 community cards in total
        missing_cards_count = total_community_cards - len(community_cards)
        community_card_ranks += [0] * missing_cards_count
        community_card_suits += [0] * missing_cards_count

        # Combine the street, stacks, to_call, dealer status, hole card ranks and suits,
        # and community card ranks and suits into a single features vector
        state_vector = [
            street_encoded,
            stack_normalized,
            main_pot,
            to_call,
            side_pot,
            position
        ] + hole_card_ranks + hole_card_suits + community_card_ranks + community_card_suits

        print(state_vector)
        print(hole_card)
        print(round_state)

        return state_vector 

    def _encode_rank(self, rank):
        rank_map = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
        return rank_map.get(rank, 0)  # 0 if rank is not present (for placeholder)

    def _encode_suit(self, suit):
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
        return suit_map.get(suit, 0)  # 0 if suit is not present (for placeholder)


    def _map_to_valid_action(self, action_index, valid_actions):
        # Ensure that the action_index is within bounds of the valid_actions list
        action_index = action_index % len(valid_actions)

        # Retrieve the chosen action using the valid index
        action_info = valid_actions[action_index]

        # If the chosen action is raise, find the minimum raise amount
        if action_info['action'] == 'raise':
            action, amount = 'raise', action_info['amount']['min']
        else:
            # For fold and call, the amount is clearly defined
            action, amount = action_info['action'], action_info['amount']
        
        return action, amount

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
