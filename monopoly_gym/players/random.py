import time
from typing import List, Optional, Set, Tuple
import random
import logging

from monopoly_gym.player import Player
from monopoly_gym.state import State, AuctionBid
from monopoly_gym.tile import ColorSet, Property, Street
from monopoly_gym.action import Action, EndTurnAction
logger = logging.getLogger("RandomPlayer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


class RandomPlayer(Player):
    """
    A player that makes random decisions:
    - Randomly buys available properties.
    - Randomly builds houses/hotels on complete color sets.
    - Randomly sells houses/hotels on owned properties.
    - Randomly participates in auctions.
    - Randomly mortgages properties when funds are low.
    - Randomly unmortgages properties.
    - Handles jail situations randomly.
    """

    def __init__(self, name, mgn_code: str, starting_balance: int = 1500, decision_delay_seconds: float = 1.0):
        super().__init__(name=name, mgn_code=mgn_code, starting_balance=starting_balance)
        self.decision_delay_seconds = decision_delay_seconds

    def decide_actions(self, game_state: State) -> List[Action]:
        time.sleep(self.decision_delay_seconds)
        actions: List[Action] = []

        temp_balance = self.balance
        if game_state.auction_state and self in game_state.auction_state.participants:
            auction_actions = self._decide_auction(game_state)
            actions.extend(auction_actions)
            return actions
        if self.in_jail:
            jail_actions, temp_balance = self._decide_jail(game_state=game_state, temp_balance=temp_balance)
            actions.extend(jail_actions)
            return actions

        bankruptcy_actions = self._decide_declare_bankruptcy(game_state=game_state)
        if len(bankruptcy_actions) > 0:
            return bankruptcy_actions
        current_tile = game_state.board.board[self.position]
        if isinstance(current_tile, Property) and current_tile.owner is None:
            buy_actions, temp_balance = self._decide_buy_property(game_state, current_tile, temp_balance)
            actions.extend(buy_actions)
        build_actions, temp_balance = self._decide_build(game_state=game_state, temp_balance=temp_balance)
        actions.extend(build_actions)
        mortgage_actions, temp_balance = self._decide_mortgage(game_state=game_state, temp_balance=temp_balance)
        actions.extend(mortgage_actions)
        unmortgage_actions, temp_balance = self._decide_unmortgage(game_state=game_state, temp_balance=temp_balance)
        actions.extend(unmortgage_actions)
        actions.append(EndTurnAction(player=self))
        return actions

    def _decide_sell(self, game_state: State, temp_balance: int) -> Tuple[List[Action], int]:

        actions: List[Action] = []
        SELL_THRESHOLD = 150 

        if self.balance >= SELL_THRESHOLD:
            return actions, temp_balance
        sellable_properties = [
            prop for prop in self.properties
            if isinstance(prop, Street) and (prop.houses > 0 or prop.hotels > 0)
        ]

        random.shuffle(sellable_properties) 

        for street in sellable_properties:
            if self.balance >= SELL_THRESHOLD:
                break

            if street.hotels > 0:
                actions.append(
                    SellBuildingAction(
                        player=self,
                        street=street,
                        quantity=1 
                    )
                )
                logger.info(
                    f"{self.name} decides to sell 1 hotel on {street.name} for ${street.color_set.hotel_cost * 0.5}."
                )
                temp_balance += street.color_set.hotel_cost * 0.5
                street.sell(quantity=1)
            elif street.houses > 0:
                actions.append(
                    SellBuildingAction(
                        player=self,
                        street=street,
                        quantity=1 
                    )
                )
                logger.info(
                    f"{self.name} decides to sell 1 house on {street.name} for ${street.color_set.house_cost * 0.5}."
                )
                temp_balance += street.color_set.house_cost * 0.5
                street.sell(quantity=1)

        return actions, temp_balance

    def _decide_auction(self, game_state: State) -> List[Action]:
        actions: List[Action] = []
        current_auction = game_state.auction_state
        highest_bid = current_auction.highest_bid()
        current_bid_amount = highest_bid.bid_amount if highest_bid else 0
        bid_decision = random.choice([True, False])

        if bid_decision and self.balance > current_bid_amount + 50:
            max_possible_bid = min(self.balance, current_bid_amount + 200)
            min_bid = current_bid_amount + 50
            bid_amount = random.randint(min_bid, max_possible_bid)
            bid = AuctionBid(bidder=self, bid_amount=bid_amount)
            actions.append(AuctionBidAction(player=self, auction_bid=bid))
            logger.info(
                f"{self.name} decides to bid ${bid_amount} on {current_auction.property.name}."
            )
        else:
            actions.append(AuctionFoldAction(player=self))
            logger.info(
                f"{self.name} decides to fold from the auction for {current_auction.property.name}."
            )

        return actions

    def _decide_buy_property(
        self, game_state: State, property_tile: Property, temp_balance: int
    ) -> Tuple[List[Action], int]:
        actions: List[Action] = []
        buy_decision = random.random() < 0.6

        if buy_decision and self.balance >= property_tile.purchase_cost:
            actions.append(
                BuyAction(
                    player=self,
                    property=property_tile,
                    price=property_tile.purchase_cost,
                )
            )
            logger.info(
                f"{self.name} decides to buy {property_tile.name} for ${property_tile.purchase_cost}."
            )
            temp_balance -= property_tile.purchase_cost
            logger.info(
                f"{self.name} now has a balance of ${self.balance} and will have ${temp_balance}."
            )
        else:
            actions.append(AuctionAction(player=self, property=property_tile))
            logger.info(
                f"{self.name} decides not to buy {property_tile.name}. Initiating auction."
            )

        return actions, temp_balance

    def _decide_build(self, game_state: State, temp_balance: int) -> Tuple[List[Action], int]:
        actions: List[Action] = []
        complete_color_sets = self._get_complete_color_sets(game_state=game_state)

        for color_set in complete_color_sets:
            streets = [
                prop
                for prop in game_state.board.board
                if isinstance(prop, Street) and prop.color_set == color_set
            ]
            min_houses = min(street.houses for street in streets)
            buildable_streets = [
                street
                for street in streets
                if street.houses == min_houses and street.houses < 5
            ]

            for street in buildable_streets:
                build_decision = random.choice([True, False])
                if build_decision and game_state.houses_available > 0 and temp_balance >= street.color_set.house_cost:
                    if street.houses < 4:
                        actions.append(
                            BuildAction(
                                player=self, street=street, quantity=1
                            )
                        )
                        logger.info(
                            f"{self.name} decides to build 1 house on {street.name}."
                        )
                        temp_balance -= street.color_set.house_cost

                    elif street.houses == 4 and game_state.houses_available > 0:
                        actions.append(
                            BuildAction(
                                player=self, street=street, quantity=1
                            )
                        )
                        logger.info(
                            f"{self.name} decides to build a hotel on {street.name}."
                        )
                        temp_balance -= street.color_set.hotel_cost
        return actions, temp_balance

    def _decide_mortgage(self, game_state: State, temp_balance :int) -> Tuple[List[Action], int]:
        actions: List[Action] = []
        MORTGAGE_THRESHOLD = 100

        if self.balance >= MORTGAGE_THRESHOLD:
            return actions, temp_balance
        unmortgaged_properties = [
            prop for prop in self.properties if not prop.is_mortgaged
        ]
        random.shuffle(unmortgaged_properties)

        for prop in unmortgaged_properties:
            if self.balance >= MORTGAGE_THRESHOLD:
                break
            actions.append(MortgageAction(player=self, property=prop))
            logger.info(
                f"{self.name} decides to mortgage {prop.name} for ${prop.mortgage_price}."
            )
            logger.info(f"{self.name} now has ${temp_balance} left")
            temp_balance += prop.mortgage_price

        return actions, temp_balance

    def _decide_unmortgage(self, game_state: State, temp_balance: int) -> Tuple[List[Action], int]:
        actions: List[Action] = []
        mortgaged_properties = [
            prop for prop in self.properties if prop.is_mortgaged
        ]
        UNMORTGAGE_THRESHOLD = 100

        random.shuffle(mortgaged_properties)
        for prop in mortgaged_properties:
            if not prop.is_mortgaged:
                continue
            unmortgage_decision = random.random() < 0.3
            if (
                unmortgage_decision and temp_balance - prop.unmortgage_price > UNMORTGAGE_THRESHOLD
            ):
                actions.append(
                    UnmortgageAction(
                        player=self, property=prop
                    )
                )
                logger.info(
                    f"{self.name} decides to unmortgage {prop.name} for ${prop.unmortgage_price}."
                )

                temp_balance -= prop.unmortgage_price

        return actions, temp_balance

    def _decide_jail(self, game_state: State, temp_balance: int) -> Tuple[List[Action], int]:
        actions: List[Action] = []
        if self.jail_free_cards > 0:
            use_card = random.choice([True, False])
            if use_card:
                actions.append(JailAction(player=self, use_card=True))
                logger.info(f"{self.name} decides to use a 'Get Out of Jail Free' card.")
                return actions, temp_balance
        if self.balance >= 50:
            pay_fine = random.random() < 0.7
            if pay_fine:
                actions.append(JailAction(player=self, pay_fine=True))
                logger.info(f"{self.name} decides to pay $50 to get out of jail.")
                return actions, temp_balance - 50
        logger.info(f"{self.name} decides to attempt rolling doubles to get out of jail.")
        return actions, temp_balance

    def _decide_trade(self, game_state: State) -> List[Action]:
        actions: List[Action] = []
        if random.random() < 0.2 and len(game_state.players) > 1:
            other_players = [p for p in game_state.players if p != self]
            responder = random.choice(other_players)
            trade_type = random.choice(['cash', 'property', 'both'])
            offer = {}
            if trade_type in ['cash', 'both']:
                cash_offer = random.randint(10, min(200, self.balance))
                offer['give'] = [f"${cash_offer}"]
            if trade_type in ['property', 'both']:
                if self.properties:
                    property_offer = random.choice(self.properties)
                    offer['give'] = offer.get('give', []) + [f"@{property_offer.index}"]
            response_type = random.choice(['cash', 'property', 'both', 'none'])
            if response_type in ['cash', 'both']:
                cash_receive = random.randint(10, min(200, responder.balance))
                offer['receive'] = [f"${cash_receive}"]
            if response_type in ['property', 'both']:
                if responder.properties:
                    property_receive = random.choice(responder.properties)
                    offer['receive'] = offer.get('receive', []) + [f"@{property_receive.index}"]
            trade_action = TradeAction(
                proposer=self,
                responder=responder,
                offer=offer,
                response='A'  # Automatically accept for simplicity
            )
            actions.append(trade_action)
            logger.info(f"{self.name} initiates a trade with {responder.name}.")
        return actions

    def _get_complete_color_sets(self, game_state: State) -> List[str]:
        """
        Returns a list of color sets for which the player owns all properties.
        """
        color_sets: Set[ColorSet] = set()
        for prop in self.properties:
            if isinstance(prop, Street):
                color_sets.add(prop.color_set)

        complete_color_sets = []
        for color_set in color_sets:
            set_properties = [
                p
                for p in game_state.board.board
                if isinstance(p, Street) and p.color_set == color_set
            ]
            if all(p.owner == self for p in set_properties):
                complete_color_sets.append(color_set)

        return complete_color_sets

    def _decide_declare_bankruptcy(self, game_state: State) -> List[Action]:

        net_worth = sum([p.max_sale_value() for p in self.properties])
        if net_worth + self.balance < 0:
            logger.info(f"{self.name} decides to declare bankruptcy!")
            return [BankruptcyAction(player=self, creditor=None)]
        return []