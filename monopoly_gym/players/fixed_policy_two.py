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
from monopoly_gym.tile import Property, Street

logger = logging.getLogger(__name__)

class FixedPolicyOnePlayer(Player):
    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(ActionSpaceType.HIERARCHICAL)
        self.policy_name = "FixedPolicyOne (Conservative Developer)"
        self.safety_buffer_general = 200
        self.safety_buffer_building = 500
        self.safety_buffer_unmortgage = 500

    def _get_property_obj_from_game_state(self, game_state: State, property_tile_index: int) -> Optional[Property]:
        if 0 <= property_tile_index < len(game_state.board.tiles):
            tile = game_state.board.tiles[property_tile_index]
            if isinstance(tile, Property):
                return tile
        logger.warning(f"{self.policy_name}: Could not find property with tile index {property_tile_index}")
        return None

    def _get_owned_property_by_tile_index(self, tile_index: int) -> Optional[Property]:
        for prop in self.properties:
            if prop.index == tile_index:
                return prop
        return None
        
    def _get_all_streets_in_color_set(self, game_state: State, color_set_name: str) -> List[Street]:
        if not color_set_name: return []
        return [
            p for p in game_state.board.tiles # Iterate all board tiles
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
            
            completes_set = False
            if isinstance(prop_on_auction, Street) and prop_on_auction.color_set:
                all_in_set = self._get_all_streets_in_color_set(game_state, prop_on_auction.color_set)
                owned_others_count = 0
                for p_in_set in all_in_set:
                    if p_in_set.index == prop_on_auction.index: continue
                    if p_in_set.owner == player:
                        owned_others_count +=1
                if len(all_in_set) > 0 and owned_others_count == len(all_in_set) - 1:
                    completes_set = True
            
            if completes_set and player.balance >= min_bid and min_bid <= prop_on_auction.purchase_cost:
                if is_action_type_valid('AuctionBidAction'):
                    logger.debug(f"{self.policy_name} ({player.name}): Bidding ${min_bid} on {prop_on_auction.name}.")
                    return [AuctionBidAction(player=player, bid_amount=min_bid)]
            
            if is_action_type_valid('AuctionFoldAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Folding auction for {prop_on_auction.name}.")
                return [AuctionFoldAction(player=player)]
        if game_state.pending_trade and game_state.pending_trade.responder == player:
            if is_action_type_valid('RejectTradeAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Declining trade from {game_state.pending_trade.proposer.name}.")
                return [RejectTradeAction(player=player)]
        if player.in_jail:
            if player.jail_free_cards > 0 and is_action_type_valid('UseJailCardAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Using Get Out of Jail Free card.")
                return [UseJailCardAction(player=player)]
            if player.balance >= 50 + self.safety_buffer_general and is_action_type_valid('PayJailFineAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Paying jail fine.")
                return [PayJailFineAction(player=player)]
            if is_action_type_valid('RollJailAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Rolling to get out of jail.")
                return [RollJailAction(player=player)]
        if player.balance < 0:
            logger.debug(f"{self.policy_name} ({player.name}): Negative balance (${player.balance}). Attempting to resolve.")
            if is_action_type_valid('SellBuildingAction'):
                owned_streets_with_buildings = []
                for prop in player.properties: # player.properties contains Property objects player owns
                    if isinstance(prop, Street) and (prop.houses > 0 or prop.hotels > 0):
                        owned_streets_with_buildings.append(prop)
                owned_streets_with_buildings.sort(key=lambda s: (s.hotels, s.houses, s.purchase_cost), reverse=True)
                
                for street_to_sell_from in owned_streets_with_buildings:
                    logger.debug(f"{self.policy_name} ({player.name}): Trying to sell building on {street_to_sell_from.name}.")
                    return [SellBuildingAction(player=player, street=street_to_sell_from, quantity=1)]
            if is_action_type_valid('MortgageAction'):
                unmortgaged_props = [p for p in player.properties if not p.is_mortgaged]
                props_to_consider_mortgaging = sorted(
                    unmortgaged_props,
                    key=lambda p_obj: (isinstance(p_obj, Street) and self._owns_full_color_set(game_state, p_obj), p_obj.purchase_cost)
                )
                for prop_to_mortgage in props_to_consider_mortgaging:
                    logger.debug(f"{self.policy_name} ({player.name}): Trying to mortgage {prop_to_mortgage.name}.")
                    return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if player.balance < 0 and is_action_type_valid('BankruptcyAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Declaring bankruptcy.")
                return [BankruptcyAction(player=player)]
        if is_action_type_valid('UnmortgageAction'):
            mortgaged_owned_props = [p for p in player.properties if p.is_mortgaged]
            mortgaged_owned_props.sort(
                key=lambda p_obj: (not (isinstance(p_obj, Street) and self._owns_full_color_set(game_state, p_obj)), -p_obj.purchase_cost)
            )
            for prop_to_unmortgage in mortgaged_owned_props:
                if player.balance >= prop_to_unmortgage.unmortgage_cost + self.safety_buffer_unmortgage:
                    logger.debug(f"{self.policy_name} ({player.name}): Unmortgaging {prop_to_unmortgage.name}.")
                    return [UnmortgageAction(player=player, property_obj=prop_to_unmortgage)]
        if is_action_type_valid('BuildAction'):
            buildable_monopolies_streets = [] # List of Street objects to build on
            
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
                if player.balance >= cost_to_build_one_house + self.safety_buffer_building:
                    logger.debug(f"{self.policy_name} ({player.name}): Building 1 house on {prop_to_build_on.name}.")
                    return [BuildAction(player=player, street=prop_to_build_on, quantity=1)]
        current_tile_obj = self._get_property_obj_from_game_state(game_state, player.position)

        if is_action_type_valid('RollDiceAction'):
            logger.debug(f"{self.policy_name} ({player.name}): Rolling dice.")
            return [RollDiceAction(player=player)]
        if isinstance(current_tile_obj, Property) and current_tile_obj.owner is None:
            if is_action_type_valid('BuyAction'):
                if player.balance >= current_tile_obj.purchase_cost + self.safety_buffer_general:
                    logger.debug(f"{self.policy_name} ({player.name}): Buying {current_tile_obj.name}.")
                    return [BuyAction(player=player, property_obj=current_tile_obj)]
                elif player.balance >= current_tile_obj.purchase_cost:
                    logger.debug(f"{self.policy_name} ({player.name}): Buying {current_tile_obj.name} (tight on cash).")
                    return [BuyAction(player=player, property_obj=current_tile_obj)]
            if is_action_type_valid('AuctionAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Auctioning {current_tile_obj.name}.")
                return [AuctionAction(player=player, property_obj=current_tile_obj)]
        if is_action_type_valid('EndTurnAction'):
            logger.debug(f"{self.policy_name} ({player.name}): Ending turn.")
            return [EndTurnAction(player=player)]
        logger.warning(f"{self.policy_name} ({player.name}): Reached fallback, no specific action chosen. Action Mask: {action_types_mask}")
        valid_action_indices = np.where(action_types_mask)[0]
        if valid_action_indices.size > 0:
            if is_action_type_valid('EndTurnAction'): return [EndTurnAction(player=player)]
            if is_action_type_valid('BankruptcyAction') and player.balance < 0 : return [BankruptcyAction(player=player)]
            action_cls_idx = valid_action_indices[0]
            action_cls = am.action_classes[action_cls_idx]
            logger.error(f"{self.policy_name} ({player.name}): CRITICAL FALLBACK. No clear default action. First valid action in mask is {action_cls.__name__}. Returning empty list, game might error.")
            return [] 
        
        logger.error(f"{self.policy_name} ({player.name}): No valid actions available at all according to mask. This is highly unusual.")
        return []