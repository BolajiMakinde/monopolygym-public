import logging
import numpy as np
from typing import List, Optional, Set, Tuple

from monopoly_gym.player import Player
from monopoly_gym.state import State
from monopoly_gym.action import (
    Action, ActionSpaceType, ActionManager, EndTurnAction, RollDiceAction, BuyAction,
    AuctionAction, MortgageAction, UnmortgageAction, BuildAction, SellBuildingAction,
    UseJailCardAction, PayJailFineAction, RollJailAction, BankruptcyAction,
    AuctionBidAction, AuctionFoldAction, RejectTradeAction
)
from monopoly_gym.tile import Property, Street, Railroad, Utility

logger = logging.getLogger(__name__)

class FixedPolicyThreePlayer(Player):

    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(ActionSpaceType.HIERARCHICAL)
        self.policy_name = "FixedPolicyThree (Railroad Baron)"
        self.safety_buffer_general = 100
        self.min_cash_after_buy_via_mortgage = 0

    def _get_property_obj_from_game_state(self, game_state: State, property_tile_index: int) -> Optional[Property]:
        if 0 <= property_tile_index < len(game_state.board.tiles):
            tile = game_state.board.tiles[property_tile_index]
            if isinstance(tile, Property):
                return tile
        return None

    def _get_owned_property_by_tile_index(self, tile_index: int) -> Optional[Property]:
        for prop in self.properties:
            if prop.index == tile_index:
                return prop
        return None
        
    def _get_all_streets_in_color_set(self, game_state: State, color_set_name: str) -> List[Street]:
        if not color_set_name: return []
        return [
            p for p in game_state.board.tiles
            if isinstance(p, Street) and p.color_set == color_set_name
        ]

    def _owns_full_color_set(self, game_state: State, street_obj: Street) -> bool:
        if not street_obj.color_set: return False
        all_props_in_set = self._get_all_streets_in_color_set(game_state, street_obj.color_set)
        if not all_props_in_set: return False
        for prop_in_set in all_props_in_set:
            if prop_in_set.owner != self:
                return False
        return True
    
    def _get_potential_mortgage_value(self, property_to_mortgage: Property, game_state: State) -> int:
        return property_to_mortgage.mortgage_value

    def _count_owned_railroads(self, game_state: State) -> int:
        count = 0
        for prop in self.properties:
            if isinstance(prop, Railroad):
                count += 1
        return count

    def _count_owned_utilities(self, game_state: State) -> int:
        count = 0
        for prop in self.properties:
            if isinstance(prop, Utility):
                count += 1
        return count

    def decide_actions(self, game_state: State) -> List[Action]:
        player = self
        action_mask_full = self.action_manager.to_action_mask(game_state)
        action_types_mask = action_mask_full['action_type']
        am = self.action_manager

        def is_action_type_valid(action_name_str: str) -> bool:
            if action_name_str not in am.action_name_to_idx:
                return False
            return bool(action_types_mask[am.action_name_to_idx[action_name_str]])

        if game_state.auction_state and game_state.auction_state.player_to_act == player:
            prop_on_auction = game_state.auction_state.property
            min_bid = game_state.auction_state.current_bid + (game_state.auction_state.bid_increment or 1)
            max_bid_allowed = player.balance

            bid_limit = 0
            if isinstance(prop_on_auction, Railroad):
                bid_limit = prop_on_auction.purchase_cost * 1.5
            elif isinstance(prop_on_auction, Utility):
                bid_limit = prop_on_auction.purchase_cost * 1.1
            else:
                bid_limit = prop_on_auction.purchase_cost * 0.6 

            actual_bid = min(min_bid, int(bid_limit), max_bid_allowed)

            if actual_bid >= min_bid:
                 if is_action_type_valid('AuctionBidAction'):
                    logger.debug(f"{self.policy_name} ({player.name}): Bidding ${actual_bid} on {prop_on_auction.name}.")
                    return [AuctionBidAction(player=player, bid_amount=actual_bid)]
            
            if is_action_type_valid('AuctionFoldAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Folding auction for {prop_on_auction.name}.")
                return [AuctionFoldAction(player=player)]
        if game_state.pending_trade and game_state.pending_trade.responder == player:
            if is_action_type_valid('RejectTradeAction'):
                return [RejectTradeAction(player=player)]
        if player.in_jail:
            if player.jail_free_cards > 0 and is_action_type_valid('UseJailCardAction'):
                return [UseJailCardAction(player=player)]
            if player.balance >= 50 + self.safety_buffer_general and is_action_type_valid('PayJailFineAction'):
                return [PayJailFineAction(player=player)]
            if is_action_type_valid('RollJailAction'):
                return [RollJailAction(player=player)]
        if player.balance < 0:
            if is_action_type_valid('SellBuildingAction'):
                owned_streets_with_buildings = [p for p in player.properties if isinstance(p, Street) and (p.houses > 0 or p.hotels > 0)]
                owned_streets_with_buildings.sort(key=lambda s: (s.hotels, s.houses, s.purchase_cost), reverse=True)
                for street_to_sell_from in owned_streets_with_buildings:
                    return [SellBuildingAction(player=player, street=street_to_sell_from, quantity=1)]
            if is_action_type_valid('MortgageAction'):
                unmortgaged_props = [p for p in player.properties if not p.is_mortgaged]
                
                props_to_consider_mortgaging = sorted(
                    unmortgaged_props,
                    key=lambda p_obj: (
                        isinstance(p_obj, Railroad),
                        isinstance(p_obj, Utility), 
                        isinstance(p_obj, Street) and self._owns_full_color_set(game_state, p_obj), 
                        p_obj.purchase_cost
                    )
                )
                for prop_to_mortgage in props_to_consider_mortgaging:
                    return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if player.balance < 0 and is_action_type_valid('BankruptcyAction'):
                return [BankruptcyAction(player=player)]
        if is_action_type_valid('UnmortgageAction'):
            if player.balance > 750:
                mortgaged_rrs_utils = [p for p in player.properties if p.is_mortgaged and (isinstance(p, Railroad) or isinstance(p, Utility))]
                mortgaged_rrs_utils.sort(key=lambda p_obj: -p_obj.purchase_cost) 
                for prop_to_unmortgage in mortgaged_rrs_utils:
                    if player.balance >= prop_to_unmortgage.unmortgage_cost + 600:
                        return [UnmortgageAction(player=player, property_obj=prop_to_unmortgage)]
        num_owned_rrs = self._count_owned_railroads(game_state)
        num_owned_utils = self._count_owned_utilities(game_state)
        if is_action_type_valid('BuildAction') and num_owned_rrs >= 4 and num_owned_utils >=2 and player.balance > 1000:
            distinct_owned_color_sets: Set[str] = set()
            for prop in player.properties:
                if isinstance(prop, Street) and prop.color_set and self._owns_full_color_set(game_state, prop):
                    all_in_set_unmortgaged = True
                    for s_in_set in self._get_all_streets_in_color_set(game_state, prop.color_set):
                        if s_in_set.is_mortgaged:
                            all_in_set_unmortgaged = False; break
                    if all_in_set_unmortgaged:
                        distinct_owned_color_sets.add(prop.color_set)
            
            sorted_color_sets = sorted(list(distinct_owned_color_sets), 
                                       key=lambda cs_name: self._get_all_streets_in_color_set(game_state, cs_name)[0].house_cost)

            for cs_name in sorted_color_sets:
                streets_in_this_set = [p for p in player.properties if isinstance(p, Street) and p.color_set == cs_name]
                streets_in_this_set.sort(key=lambda s: (s.hotels, s.houses))
                
                prop_to_build_on = streets_in_this_set[0]
                if prop_to_build_on.hotels == 1: continue 

                cost_to_build_one_house = prop_to_build_on.house_cost
                if player.balance >= cost_to_build_one_house + 800: 
                    return [BuildAction(player=player, street=prop_to_build_on, quantity=1)]
        current_tile_obj = self._get_property_obj_from_game_state(game_state, player.position)

        if is_action_type_valid('RollDiceAction'):
            return [RollDiceAction(player=player)]

        if isinstance(current_tile_obj, Property) and current_tile_obj.owner is None:
            can_afford_directly = player.balance >= current_tile_obj.purchase_cost + self.safety_buffer_general
            is_priority_target = isinstance(current_tile_obj, (Railroad, Utility))

            if is_action_type_valid('BuyAction'):
                if is_priority_target:
                    if player.balance >= current_tile_obj.purchase_cost:
                        return [BuyAction(player=player, property_obj=current_tile_obj)]
                    elif is_action_type_valid('MortgageAction'):
                        props_to_mortgage_for_buy = [
                            p for p in player.properties 
                            if not p.is_mortgaged and not isinstance(p, (Railroad, Utility))
                        ]
                        props_to_mortgage_for_buy.sort(key=lambda p_obj: ((isinstance(p_obj,Street) and self._owns_full_color_set(game_state,p_obj)), p_obj.purchase_cost) )
                        
                        for prop_to_mortgage in props_to_mortgage_for_buy:
                            if player.balance + self._get_potential_mortgage_value(prop_to_mortgage, game_state) >= current_tile_obj.purchase_cost + self.min_cash_after_buy_via_mortgage:
                                return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
                elif can_afford_directly and current_tile_obj.purchase_cost < 150 :
                     return [BuyAction(player=player, property_obj=current_tile_obj)]
            
            if is_action_type_valid('AuctionAction'):
                return [AuctionAction(player=player, property_obj=current_tile_obj)]
        if is_action_type_valid('EndTurnAction'):
            return [EndTurnAction(player=player)]
        logger.warning(f"{self.policy_name} ({player.name}): Reached fallback. Action Mask: {action_types_mask}")
        if is_action_type_valid('EndTurnAction'): return [EndTurnAction(player=player)]
        if is_action_type_valid('BankruptcyAction') and player.balance < 0 : return [BankruptcyAction(player=player)]
        return []