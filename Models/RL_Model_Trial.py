from pypokerengine.players import BasePokerPlayer

class RLAgent(BasePokerPlayer):

    FOLD = 0
    CALL = 1
    MIN_RAISE = 2
    MAX_RAISE = 3

    def set_action(self, action):
        self.action = action

    def select_action(self, valid_actions, state):
        return valid_actions[1]['action'], valid_actions[1]['amount']


 