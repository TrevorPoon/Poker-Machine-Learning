from pypokerengine.players import BasePokerPlayer

class All_in_Player_1(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [CheckAction, CallAction, FoldAction, RaiseAction]

        if round_state['pot']['main']['amount'] < 20:
            # All in
            call_action_info = valid_actions[2]
            action, amount = call_action_info["action"], call_action_info["amount"]['max']
        else:
            # Call
            call_action_info = valid_actions[1]
            action, amount = call_action_info["action"], call_action_info["amount"]

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
