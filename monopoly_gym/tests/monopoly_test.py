import pytest
from monopoly_gym.state import State, TradeOffer, AuctionState, AuctionBid, AuctionState

from monopoly_gym.player import Player
from monopoly_gym.tile import Property, Street

from monopoly_gym.action import (
    RollDiceAction,
    EndTurnAction,
    BuyAction,
    AuctionAction,
    AuctionBidAction,
    AuctionFoldAction,
    MortgageAction,
    UnmortgageAction,
    BuildAction,
    SellBuildingAction,
    UseJailCardAction,
    PayJailFineAction,
    RollJailAction,
    BankruptcyAction,
    ProposeTradeAction,
    AcceptTradeAction,
    RejectTradeAction,
    ActionManager,
    ActionSpaceType,
)



class SimplePlayer(Player):
    def decide_actions(self, game_state: State):
        return []


@pytest.fixture
def fresh_state():
    st = State(max_turns=100)
    p1 = SimplePlayer(name="P1", mgn_code="P1")
    p2 = SimplePlayer(name="P2", mgn_code="P2")
    st.players = [p1, p2]
    st.current_player_index = 0
    return st
def test_roll_dice_normal(fresh_state: State):

    st = fresh_state
    current_player = st.current_player()
    current_player.position = 7
    st.current_consecutive_doubles = 0
    st.rolled_this_turn = False

    action = RollDiceAction(current_player)
    action.dice_roll = (3, 4)
    action.rolled_doubles = False
    action.process(st)
    assert st.rolled_this_turn is True, "Should have ended the roll for a normal (non-doubles) roll."
    assert st.current_consecutive_doubles == 0


def test_roll_dice_doubles_once(fresh_state: State):
    """
    Scenario:
      - Current player at position 10.
      - Rolls (5,5) => doubles.
    Expected:
      - position should move by total (10) => new position = (10+10) mod 40 = 20.
      - rolled_this_turn = False (since they get another roll).
      - current_consecutive_doubles increments by 1.
    """
    st = fresh_state
    current_player = st.current_player()
    current_player.position = 10
    st.current_consecutive_doubles = 0
    st.rolled_this_turn = False

    action = RollDiceAction(current_player)
    action.dice_roll = (5, 5)
    action.rolled_doubles = True
    action.process(st)
    assert st.current_consecutive_doubles == 1, "Should have incremented doubles count."
    assert st.rolled_this_turn is True


def test_roll_dice_three_consecutive_doubles(fresh_state: State):
    """
    Scenario:
      - state.current_consecutive_doubles=2
      - Player at position 5, not in jail
      - Rolls doubles again => third consecutive
    Expected:
      - Goes to jail: position=10, in_jail=True
      - turn ends (rolled_this_turn=True), consecutive_doubles resets to 0
    """
    st = fresh_state
    current_player = st.current_player()
    current_player.position = 5
    st.current_consecutive_doubles = 2
    st.rolled_this_turn = False

    action = RollDiceAction(current_player)
    action.dice_roll = (3, 3)
    action.rolled_doubles = True
    action.process(st)

    assert current_player.in_jail, "After 3 consecutive doubles, player should be in jail."
    assert current_player.position == 10, "Should be teleported to Jail (index=10)."
    assert st.current_consecutive_doubles == 0, "Should reset consecutive doubles to 0."
    assert st.rolled_this_turn is True, "Turn ends immediately after going to jail."
def test_buy_unowned_property(fresh_state: State):
    """
    Scenario:
      - P1 lands on an unowned property, e.g. States Avenue (index=13, cost=140).
      - P1 has $500.
    Expected:
      - P1 pays 140 => new balance = 360
      - Property.owner = P1
      - Auction state remains None
    """
    st = fresh_state
    p1 = st.players[0]
    p1.balance = 500
    p1.position = 13
    prop = st.board.board[13]  # Should be "States Avenue" by default board
    assert isinstance(prop, Property)
    assert prop.owner is None

    action = BuyAction(p1, prop)
    action.process(st)

    assert p1.balance == 500 - 140, "Should have deducted 140 from buyer's balance."
    assert prop.owner == p1, "Should set the property owner."
    assert st.auction_state is None, "No auction should start if player simply buys."
def test_initiate_auction(fresh_state: State):
    """
    Scenario:
      - Player lands on unowned property but chooses Auction.
    Expected:
      - state.auction_state becomes an AuctionState for that property
      - property remains unowned
    """
    st = fresh_state
    p1 = st.players[0]
    p1.position = 1  # "Mediterranean Avenue" for example
    tile = st.board.board[1]
    assert tile.owner is None

    action = AuctionAction(p1, tile)
    action.process(st)

    assert st.auction_state is not None, "Auction should have started."
    assert st.auction_state.auction_item == tile
    assert tile.owner is None, "Property still unowned at auction start."


def test_auction_bid(fresh_state: State):
    """
    Scenario:
      - An AuctionState is active for a property.
      - Current auction participant is P1, highest bid so far is 120.
      - P1 has $600, bids 125.
    Expected:
      - Bids list updates
      - current_bidder_index rotates
    """
    st = fresh_state
    p1, p2 = st.players[0], st.players[1]
    tile = st.board.board[1]
    st.auction_state = AuctionState(
        aucition_item=tile,
        participants=[p1, p2],
        bids=[AuctionBid(p1, 120)],  # existing highest=120
        initial_bidding_index=0
    )
    p1.balance = 600

    action = AuctionBidAction(p1, bid_amount=125)
    action.process(st)
    all_bids = [b.bid_amount for b in st.auction_state.bids]
    assert 125 in all_bids, "Should contain new bid of 125."
    assert st.auction_state.current_bidder_index == 1, "Should rotate to next participant."


def test_auction_fold(fresh_state: State):
    """
    Scenario:
      - AuctionState with participants = [p1, p2]
      - Highest bid so far is by p1
      - p2 decides to fold
    Expected:
      - p2 removed from participants
      - Auction resolves => p1 wins the property
      - p1's balance decreases by the highest bid
    """
    st = fresh_state
    p1, p2 = st.players
    tile = st.board.board[5]  # Reading Railroad, say
    st.auction_state = AuctionState(
        aucition_item=tile,
        participants=[p1, p2],
        bids=[AuctionBid(p1, 200)],
        initial_bidding_index=1  # It's p2's turn
    )
    p1.balance = 1000
    p2.balance = 800

    action = AuctionFoldAction(p2)
    action.process(st)
    assert st.auction_state is None, "Auction should be cleared/resolved."
    assert tile.owner == p1, "p1 should now own the property after winning the auction."
    assert p1.balance == 1000 - 200, "Should have deducted the winning bid from p1."
def test_mortgage_property(fresh_state: State):
    """
    Scenario:
      - p1 owns Connecticut Avenue (index=9 cost=120, mortgage=60).
      - p1 has $20
      - Mortgage action => +60
    Expected:
      - property.is_mortgaged=True
      - p1.balance=80
    """
    st = fresh_state
    p1 = st.players[0]
    prop = st.board.board[9]  # "Connecticut Avenue"
    p1.properties.append(prop)
    prop.owner = p1
    p1.balance = 20

    action = MortgageAction(p1, prop)
    action.process(st)

    assert prop.is_mortgaged, "Should be mortgaged now."
    assert p1.balance == 80, "Should add mortgage value to balance."


def test_unmortgage_property(fresh_state: State):
    """
    Scenario:
      - p1 owns Electric Company (index=12)
      - It's mortgaged, unmortgage_price=~83
      - p1 has $500
    Expected:
      - p1 pays 83 => new balance=417
      - property.is_mortgaged=False
    """
    st = fresh_state
    p1 = st.players[0]
    prop = st.board.board[12]  # Electric Company
    p1.properties.append(prop)
    prop.owner = p1
    prop.is_mortgaged = True
    p1.balance = 500

    action = UnmortgageAction(p1, prop)
    action.process(st)

    assert not prop.is_mortgaged, "Should be unmortgaged now."
    assert p1.balance == 500 - prop.unmortgage_price, "Should deduct unmortgage cost."
def test_build_without_complete_color_set(fresh_state: State):
    """
    Scenario:
      - p1 owns only 'Oriental Avenue' (index=6) from Light Blue set
      - Has $1000, plenty houses in bank
      - tries BuildAction( quantity=1 )
    Expected:
      - Should not allow building, or raise error,
        because p1 doesn't own entire color set.
    """
    st = fresh_state
    p1 = st.players[0]
    oriental = st.board.board[6]  # "Oriental Avenue"
    p1.properties.append(oriental)
    oriental.owner = p1
    p1.balance = 1000

    action = BuildAction(p1, oriental, quantity=1)
    try:
        action.process(st)
    except Exception:
        return
    assert oriental.houses == 0, "Should not build if the color set isn't complete."


def test_build_three_houses(fresh_state: State):
    """
    Scenario:
      - p1 owns entire Orange set => indexes(16, 18, 19)
      - Each Orange house costs $100
      - p1 has $1500, bank has 32 houses
      - build 3 on 'New York Ave' (index=19)
    Expected:
      - newYorkAve.houses=3, p1.balance=1200, bank=29 houses
    """
    st = fresh_state
    p1 = st.players[0]
    st.houses_available = 32
    st_james = st.board.board[16]
    tennessee = st.board.board[18]
    new_york = st.board.board[19]

    for tile in [st_james, tennessee, new_york]:
        tile.owner = p1
        p1.properties.append(tile)

    p1.balance = 1500
    st_james.houses = 2
    tennessee.houses = 2
    action = BuildAction(p1, new_york, quantity=3)
    action.process(st)

    assert new_york.houses == 3, "Should have 3 houses now."
    assert p1.balance == 1500 - (3 * 100), "Should pay 300 total."
    assert st.houses_available == 32 - 3, "Should reduce available houses in bank by 3."


def test_build_hotel_from_4_houses(fresh_state: State):
    """
    Scenario:
      - 'Boardwalk' has 4 houses, cost of each house is $200 (Dark Blue).
      - p1 has $2000, bank has enough houses
      - build 1 more => becomes a hotel
    Expected:
      - boardwalk.houses=0, hotels=1, cost=200, p1.balance=1800
    """
    st = fresh_state
    p1 = st.players[0]
    boardwalk = st.board.board[39]
    assert isinstance(boardwalk, Street)
    boardwalk = st.board.board[39]
    park_place = st.board.board[37]
    boardwalk.owner = p1
    park_place.owner = p1
    p1.properties.extend([boardwalk, park_place])
    boardwalk.houses = 4
    park_place.houses = 4
    p1.balance = 2000

    action = BuildAction(p1, boardwalk, quantity=1)
    action.process(st)

    assert boardwalk.houses == 0 and boardwalk.hotels == 1, "Should convert to hotel."
    assert p1.balance == 2000 - 200, "Dark Blue house cost is 200 => 1 more = 200."


def test_build_beyond_hotel_raises(fresh_state: State):
    """
    Scenario:
      - 'Park Place' already has 1 hotel
      - Attempt building again => invalid
    """
    st = fresh_state
    p1 = st.players[0]
    park_place = st.board.board[37]
    boardwalk = st.board.board[39]
    park_place.owner = p1
    boardwalk.owner = p1
    p1.properties.extend([park_place, boardwalk])

    park_place.hotels = 1
    p1.balance = 2000

    action = BuildAction(p1, park_place, quantity=1)
    with pytest.raises(Exception, match="Already has a hotel"):
        action.process(st)
def test_sell_building_two_houses(fresh_state: State):
    """
    Scenario:
      - Kentucky Ave (index=21) has 3 houses
      - House cost is $150 each for Red
      - p1 sells 2 houses => gets 150 total (2*(150/2))
    Expected:
      - kentucky.houses=1
      - p1.balance increases by 150
      - bank houses += 2
    """
    st = fresh_state
    p1 = st.players[0]
    kentucky = st.board.board[21]
    kentucky.owner = p1
    p1.properties.append(kentucky)
    kentucky.houses = 3
    p1.balance = 200
    st.houses_available = 20

    action = SellBuildingAction(p1, kentucky, quantity=2)
    action.process(st)

    assert kentucky.houses == 1, "Should have sold 2 houses => left with 1."
    assert p1.balance == 350, "Starting at 200 plus 150 refund."
    assert st.houses_available == 22, "Should add 2 back to bank."
def test_landing_on_go_to_jail_tile(fresh_state: State):
    """
    Scenario:
      - p1 lands on index=30 => 'Go To Jail'
    Expected:
      - p1.position=10, p1.in_jail=True
    """
    st = fresh_state
    p1 = st.players[0]
    p1.position = 30
    st.handle_landing_on_tile(p1, dice_roll=(2, 3))
    assert p1.position == 10, "Should move to Jail"
    assert p1.in_jail, "Should be in jail now."


def test_use_jail_card(fresh_state: State):
    """
    Scenario:
      - p1 in jail, has 1 jail_free_cards
    Expected:
      - UseJailCard => p1.in_jail=False, jail_turns=0, jail_free_cards=0
    """
    st = fresh_state
    p1 = st.players[0]
    p1.in_jail = True
    p1.jail_turns = 1
    p1.jail_free_cards = 1

    action = UseJailCardAction(p1)
    action.process(st)

    assert not p1.in_jail, "Should free the player from jail."
    assert p1.jail_turns == 0, "Reset turns"
    assert p1.jail_free_cards == 0, "Used one card"


def test_pay_jail_fine(fresh_state: State):
    """
    Scenario:
      - p1 in jail, has $100, jail_turns=1
      - Pays $50 fine
    Expected:
      - p1 balance=50, in_jail=False, jail_turns=0
    """
    st = fresh_state
    p1 = st.players[0]
    p1.in_jail = True
    p1.jail_turns = 1
    p1.balance = 100

    action = PayJailFineAction(p1)
    action.process(st)

    assert p1.balance == 50
    assert not p1.in_jail
    assert p1.jail_turns == 0


def test_roll_jail_action_3rd_fail(fresh_state: State):
    """
    Scenario:
      - p1 in jail, jail_turns=2
      - This is their 3rd attempt to roll doubles
    Expected:
      - By standard rules, if they fail, they must pay or use card.
      - We'll just check that .process() increments jail_turns to 3
        but your code might not forcibly make them pay automatically.
    """
    st = fresh_state
    p1 = st.players[0]
    p1.in_jail = True
    p1.jail_turns = 2

    action = RollJailAction(p1)
    action.process(st)
    assert p1.jail_turns == 3, "Now they've tried 3 times."
def test_end_turn_normal(fresh_state: State):
    """
    Scenario:
      - p1 has already rolled, owes no money, no unowned property pending
    Expected:
      - end turn => rolled_this_turn=False
      - current_player_index moves from 0->1
    """
    st = fresh_state
    st.rolled_this_turn = True
    st.current_player_index = 0

    p1 = st.players[0]
    prop = st.board.board[p1.position]
    if isinstance(prop, Property):
        prop.owner = p1

    action = EndTurnAction(p1)
    action.process(st)

    assert st.rolled_this_turn is False, "Should reset for next turn."
    assert st.current_player_index == 1, "Should pass turn to next player."

def test_end_turn_negative_balance(fresh_state: State):
    """
    Scenario:
      - p1 has negative balance => can't end turn
    Expected:
      - End turn is disallowed or no-op
    """
    st = fresh_state
    p1 = st.players[0]
    p1.balance = -10

    action = EndTurnAction(p1)
    action.process(st)
    assert st.current_player_index == 0, "Should not have ended turn when negative balance."
def test_propose_trade(fresh_state: State):
    """
    Scenario:
      - p1 => p2, offering: $100 + no property in exchange for $200
      - no pending_trade
    Expected:
      - state.pending_trade is set to that trade
    """
    st = fresh_state
    p1, p2 = st.players
    assert st.pending_trade is None

    action = ProposeTradeAction(
        trade_offer=TradeOffer(
            proposer=p1,
            responder=p2,
            cash_offered=100,
            properties_offered=[],
            get_out_of_jail_cards_offered=0,
            cash_asking=200,
            properties_asking=[],
            get_out_of_jail_cards_asking=0
        )
    )
    action.process(st)

    assert st.pending_trade is not None, "Should create a pending trade."
    trade = st.pending_trade
    assert trade.cash_offered == 100
    assert trade.cash_asking == 200


def test_propose_trade_while_pending(fresh_state: State):
    """
    Scenario:
      - There's already a pending trade
      - p1 tries another ProposeTradeAction
    Expected:
      - Should be disallowed or raise warning
      - pending_trade remains the old one
    """
    st = fresh_state
    p1, p2 = st.players
    st.pending_trade = object()  # something not None

    old_trade = st.pending_trade
    action = ProposeTradeAction(
        trade_offer=TradeOffer(
          proposer=p1,
          responder=p2,
          cash_offered=10,
          properties_offered=[],
          get_out_of_jail_cards_offered=0,
          cash_asking=10,
          properties_asking=[],
          get_out_of_jail_cards_asking=0
        )
    )
    action.process(st)
    assert st.pending_trade == old_trade, "New trade shouldn't replace existing one."


def test_accept_trade(fresh_state: State):
    """
    Scenario:
      - pending_trade: p1 -> (cash=100, property=[3]) for p2 -> (cash=200)
      - p2 calls AcceptTrade
    Expected:
      - p1.balance => p1.balance -100 +200
      - p2.balance => p2.balance -200 +100
      - property index=3 changes ownership from p1 to p2
      - pending_trade=None
    """
    st = fresh_state
    p1, p2 = st.players
    p1.balance = 500
    p2.balance = 400
    baltic: Property = st.board.board[3]
    baltic.owner = p1
    p1.properties.append(baltic)

    st.pending_trade = type('FakeTrade', (), {})()
    st.pending_trade = TradeOffer(
        proposer=p1,
        responder=p2,
        cash_offered=100,
        properties_offered=[baltic],
        cash_asking=200,
        properties_asking=[],
        get_out_of_jail_cards_asking=0,
        get_out_of_jail_cards_offered=0
    )

    action = AcceptTradeAction(p2)
    action.process(st)

    assert st.pending_trade is None, "Trade is cleared after acceptance."
    assert p1.balance == 500 - 100 + 200 == 600, "p1 net +100"
    assert p2.balance == 400 - 200 + 100 == 300, "p2 net -100"
    assert baltic.owner == p2, "Baltic is transferred to p2."
    assert baltic in p2.properties and baltic not in p1.properties


def test_reject_trade(fresh_state: State):
    """
    Scenario:
      - pending_trade: p1->p2
      - p2 rejects
    Expected:
      - pending_trade cleared, no balances or ownership changes
    """
    st = fresh_state
    p1, p2 = st.players
    st.pending_trade = object()  # fake
    st.pending_trade = TradeOffer(
        proposer=p1,
        responder=p2,
        cash_asking=50,
        properties_asking=[],
        cash_offered=0,
        properties_offered=[],
        get_out_of_jail_cards_asking=0,
        get_out_of_jail_cards_offered=0
    )
    old_p1_balance = p1.balance
    old_p2_balance = p2.balance

    action = RejectTradeAction(p2)
    action.process(st)

    assert st.pending_trade is None, "Should remove the pending trade on reject."
    assert p1.balance == old_p1_balance, "No money changes hands."
    assert p2.balance == old_p2_balance, "No money changes hands."
def test_forced_bankruptcy(fresh_state: State):
    """
    Scenario:
      - p1 has negative balance => must declare bankruptcy
      - p1 has some properties
    Expected:
      - p1 is removed from state.players
      - p1's properties become unowned
      - turn passes to next
    """
    st = fresh_state
    p1, p2 = st.players
    p1.balance = -100
    some_tile = st.board.board[3]  # e.g. Baltic
    some_tile.owner = p1
    p1.properties.append(some_tile)

    action = BankruptcyAction(p1)
    action.process(st)

    assert p1 not in st.players, "Bankrupt player removed from the game."
    assert some_tile.owner is None, "Properties returned to bank."
    assert st.current_player_index == 0, "Now only 1 player remains, index=0 might remain p2 if the code re-lists players."


def test_voluntary_bankruptcy_positive_balance(fresh_state: State):
    """
    Scenario:
      - If VOLUNTARY_BANKRUPTCY=True, a player might bankrupt themselves
        even with positive cash. This is not standard Monopoly, but we check the code path.
    Expected:
      - They are removed from the game, properties unowned
    """
    st = fresh_state
    p1, p2 = st.players
    p1.balance = 100
    tile = st.board.board[1]
    tile.owner = p1
    p1.properties.append(tile)

    action = BankruptcyAction(p1)
    action.process(st)
    assert p1 not in st.players
    assert tile.owner is None


def test_action_manager_no_end_turn_if_unowned_property_hierarchical(fresh_state):

    st = fresh_state
    p1 = st.players[0]
    p1.position = 3  # unowned
    st.auction_state = None

    manager = ActionManager(action_space_type=ActionSpaceType.HIERARCHICAL)
    mask_dict = manager.to_action_mask(st)  # This returns a dict with 'action_type' and 'parameters'
    endturn_idx = manager.action_classes.index(EndTurnAction)
    valid_for_endturn = mask_dict["action_type"][endturn_idx]
    assert not valid_for_endturn, "EndTurn should be masked out in hierarchical mode, too."

def test_auction_no_bids_resolves_nobody_buys(fresh_state: State):
    """
    Scenario:
      - There's an AuctionState for an unowned property, participants=[p1,p2]
      - Neither player bids (or they fold), resulting in no bids at all.
    Expected:
      - Auction ends with property unowned
      - state.auction_state becomes None
    """
    st = fresh_state
    p1, p2 = st.players
    tile = st.board.board[5]  # Reading Railroad
    st.auction_state = AuctionState(
        aucition_item=tile,
        participants=[p1, p2],
        bids=[]
    )
    st.auction_state.resolve(st)

    assert st.auction_state is None, "Auction should end with no bids."
    assert tile.owner is None, "Property remains unowned."

def test_mortgage_property_twice_noop(fresh_state: State):
    """
    Scenario:
      - p1 mortgages a property successfully
      - attempts to mortgage again
    Expected:
      - Second mortgage action doesn't change anything
    """
    st = fresh_state
    p1 = st.players[0]
    prop = st.board.board[9]  # Connecticut Avenue
    prop.owner = p1
    p1.properties.append(prop)
    p1.balance = 10
    action1 = MortgageAction(p1, prop)
    action1.process(st)
    assert prop.is_mortgaged
    first_balance = p1.balance
    action2 = MortgageAction(p1, prop)
    action2.process(st)

    assert p1.balance == first_balance, "No additional money after second mortgage."
    assert prop.is_mortgaged, "Still mortgaged, no change."

def test_pay_rent_full_color_set_no_houses(fresh_state: State):
    """
    Scenario:
      - p1 owns all PINK properties (St. Charles, States Ave, Virginia).
      - p2 lands on 'States Avenue' (rent=20 if color set complete, 10 if partial).
      - No houses built.
    Expected:
      - p2 pays the 'color_set' rent => e.g. 2x the no_color_set rent
      - p1's balance increases by that rent
    """
    st = fresh_state
    p1, p2 = st.players
    p1.position = 11  # St. Charles
    p2.position = 13  # States Ave, which is about to be landed on
    pink_indices = [11, 13, 14]  # St. Charles, States Ave, Virginia
    for idx in pink_indices:
        tile = st.board.board[idx]
        tile.owner = p1
        p1.properties.append(tile)
    old_balance_p2 = p2.balance
    old_balance_p1 = p1.balance
    st.handle_landing_on_tile(player=p2, dice_roll=(2, 4))
    expected_rent = 20
    assert p2.balance == (old_balance_p2 - expected_rent), "p2 pays rent of 20"
    assert p1.balance == (old_balance_p1 + expected_rent), "p1 receives rent of 20"


def test_build_on_mortgaged_property_disallowed(fresh_state: State):
    """
    Scenario:
      - p1 owns entire Light Blue set, but 'Oriental Avenue' is mortgaged.
      - p1 attempts to BuildAction( ... ) on Oriental.
    Expected:
      - Build fails or no-op (because the property is mortgaged).
      - No change in houses or player's balance.
    """
    st = fresh_state
    p1 = st.players[0]
    tiles = [st.board.board[6], st.board.board[8], st.board.board[9]]  # Light Blue
    for t in tiles:
        t.owner = p1
        p1.properties.append(t)
    tiles[0].is_mortgaged = True

    old_balance = p1.balance = 800
    st.houses_available = 20

    action = BuildAction(p1, tiles[0], quantity=1)  # tries building on mortgaged
      
    with pytest.raises(Exception, match="Cannot build on Oriental Avenue, it is mortgaged."):
      action.process(st)
    assert tiles[0].houses == 0, "Should not build on mortgaged property"
    assert p1.balance == old_balance, "No money spent"
    assert st.houses_available == 20, "Bank houses remain the same"


def test_build_no_houses_left_in_bank(fresh_state: State):
    """
    Scenario:
      - st.houses_available=0 => bank has no houses left
      - p1 owns entire Red set, tries to build on 'Illinois Avenue'
    Expected:
      - Build fails or no-op, because st.houses_available=0
      - No changes to p1 balance or street houses
    """
    st = fresh_state
    p1 = st.players[0]
    for idx in [21, 23, 24]:
        prop = st.board.board[idx]
        prop.owner = p1
        p1.properties.append(prop)

    st.houses_available = 0
    illinois = st.board.board[24]
    old_balance = p1.balance = 1000

    action = BuildAction(p1, illinois, quantity=1)
    action.process(st)

    assert illinois.houses == 0, "No houses should be added because bank=0"
    assert p1.balance == old_balance, "Should not charge if no houses available."

def test_build_evenly_across_color_set(fresh_state: State):
    """
    Scenario:
      - By standard Monopoly rules, you must build houses evenly.
      - p1 owns entire Green set => Pacific(31), N. Carolina(32), Pennsylvania(34).
      - p1 tries to build 2 houses on North Carolina while the others have 0.
    Expected (if you enforce even-building):
      - Raise Exception or skip building if it’s not even.
    Expected (if you do NOT enforce even-building):
      - Successfully builds 2 houses on N. Carolina.
    We'll just show the check for an *Exception*, but you can adapt for no-ops if you want.
    """
    st = fresh_state
    p1 = st.players[0]
    green_idx = [31, 32, 34]  # Pacific, North Carolina, Pennsylvania
    for idx in green_idx:
        tile = st.board.board[idx]
        tile.owner = p1
        p1.properties.append(tile)
    nc = st.board.board[32]
    old_balance = p1.balance = 1500
    st.houses_available = 20
    action = BuildAction(p1, nc, quantity=2)
    with pytest.raises(Exception, match="Cannot build due to even-build rule on Pacific Avenue"):
        action.process(st)

def test_bankruptcy_with_mortgaged_properties(fresh_state: State):
    """
    Scenario:
      - p1 has negative balance, owns 2 properties (1 is mortgaged, 1 is not).
      - p1 declares bankruptcy
    Expected:
      - p1 removed from game
      - Both properties become unowned (mortgage status doesn't matter after they're returned to bank)
    """
    st = fresh_state
    p1, p2 = st.players
    p1.balance = -50
    baltic = st.board.board[3]
    states_ave = st.board.board[13]
    baltic.owner = p1
    states_ave.owner = p1
    states_ave.is_mortgaged = True
    p1.properties.extend([baltic, states_ave])

    action = BankruptcyAction(p1)
    action.process(st)

    assert p1 not in st.players, "p1 removed"
    assert baltic.owner is None, "baltic unowned after bankruptcy"
    assert states_ave.owner is None, "states_ave unowned even if mortgaged"

def test_use_jail_card_none_available(fresh_state: State):
    """
    Scenario:
      - p1 in jail with 0 jail_free_cards
      - tries UseJailCardAction
    Expected:
      - no-op, still in jail, still 0 cards
    """
    st = fresh_state
    p1 = st.players[0]
    p1.in_jail = True
    p1.jail_turns = 1
    p1.jail_free_cards = 0

    old_balance = p1.balance
    action = UseJailCardAction(p1)
    action.process(st)

    assert p1.in_jail, "Should remain in jail"
    assert p1.jail_free_cards == 0, "No card to use"
    assert p1.balance == old_balance, "No cost changes"
