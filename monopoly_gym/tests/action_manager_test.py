import pytest

from monopoly_gym.state import State
from monopoly_gym.player import Player
from monopoly_gym.action import ActionManager, ActionSpaceType, EndTurnAction
from monopoly_gym.tile import Property

class SimplePlayer(Player):
    def decide_actions(self, game_state: State):
        return []

@pytest.fixture
def minimal_state():
    st = State()
    p1 = SimplePlayer(name="P1", mgn_code="P1")
    st.players = [p1]
    st.current_player_index = 0
    return st

def test_action_manager_no_endturn_if_unowned_property(minimal_state):
    """
    Ensures that when the current player is standing on an unowned property
    and no auction is active, the action mask for 'EndTurnAction' is false.
    """
    manager = ActionManager(action_space_type=ActionSpaceType.HIERARCHICAL)
    st = minimal_state
    st.players[0].position = 3
    tile = st.board.board[3]
    assert isinstance(tile, Property)
    assert tile.owner is None

    mask = manager.to_action_mask(st)  # hierarchical mask => {"action_type": [...], "parameters": {...}}
    action_classes = manager.action_classes
    endturn_idx = action_classes.index(EndTurnAction)
    assert mask["action_type"][endturn_idx] is False, (
        "EndTurn should not be valid if the current player is on an unowned property "
        "with no auction in progress."
    )
