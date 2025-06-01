"""
monopoly_gym/players/llm_hf.py
"""

from __future__ import annotations

import os
import json
import logging
from enum import Enum
import sys
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

from monopoly_gym.player import Player
from monopoly_gym.state import State, AuctionBid
from monopoly_gym.tile import Property, Street
from monopoly_gym.action import HIERARCHICAL_ACTION_CLASSES

logger = logging.getLogger(__name__)


class SupportedModel(Enum):
    """
    Enumerates the supported Hugging Face models.
    Extend this Enum if you add more HF models.
    """
    GEMMA_2_2B_IT = "google/gemma-2-2b-it"
    GEMMA_7B = "google/gemma-7b"
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
    LLAMA_3_2_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B"
    MISTRAL_7B_INSTRUCT_V0_3 = "mistralai/Mistral-7B-Instruct-v0.3"
    MISTRAL_7B_INSTRUCT_V0_2 = "mistralai/Mistral-7B-Instruct-v0.2"
    MISTRAL_7B_INSTRUCT_V0_1 = "mistralai/Mistral-7B-Instruct-v0.1"
    MISTRAL_NEMO_INSTRUCT_2407 = "mistralai/Mistral-Nemo-Instruct-2407"
    GPT2 = "openai-community/gpt2"
    GPT2_MEDIUM = "openai-community/gpt2-medium"
    GPTJ_6B = "EleutherAI/gpt-j-6b"
    BLOOM = "bigscience/bloom"
    FALCON_40B = "tiiuae/falcon-40b"
    FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"
    PHI_2 = "microsoft/phi-2"
    PHI_3_MINI_4K_INSTRUCT = "microsoft/phi-3-mini-4k-instruct"
    STARCODER = "bigcode/starcoder"
    DOLLY_V2_12B = "databricks/dolly-v2-12b"
    GROK_1 = "xai-org/grok-1"
    HERMES_2_PRO_MISTRAL_7B = "NousResearch/Hermes-2-Pro-Mistral-7B"
    OLMO_2_1124_13B_INSTRUCT = "allenai/OLMo-2-1124-13B-Instruct"
    QWEN2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
    QWEN2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"
    QWEN2_5_32B_INSTRUCT = "Qwen/Qwen2.5-32B-Instruct"
    QWEN2_5_3B_INSTRUCT = "Qwen/Qwen2.5-3B-Instruct"
    QWEN2_5_0_5B = "Qwen/Qwen2.5-0.5B"

class LLMAction(BaseModel):
    name: str = Field(
        ...,
        description="The name of the action. Must be one of: "
                    "BuyAction, AuctionAction, AuctionBidAction, AuctionFoldAction, "
                    "MortgageAction, UnmortgageAction, BuildAction, SellBuildingAction, "
                    "JailAction, TradeAction, BankruptcyAction"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value parameters for the action."
    )


class LLMActions(BaseModel):

    actions: List[LLMAction] = Field(
        default_factory=list,
        description="List of actions to perform sequentially."
    )

class LanguageModelHuggingFace(Player):
    """
    A Monopoly bot that delegates decisions to a Hugging Face model. It uses
    the 'decide_actions' method to gather the current action space mask,
    builds a prompt with game state + allowed actions, calls the HF Inference
    API, parses the structured JSON, and returns the corresponding Action
    instances.

    If the response from the primary model is invalid or fails,
    we fall back to a second model if defined in the environment.
    """

    def __init__(
        self,
        name: str,
        mgn_code: str,
        model_type: SupportedModel,
        starting_balance: int = 1500
    ) -> None:
        super().__init__(name=name, mgn_code=mgn_code, starting_balance=starting_balance)
        self.model_type = model_type.value

        # Primary HF token
        primary_token = os.environ.get("HF_TOKEN", None)
        if not primary_token:
            raise ValueError("HF_TOKEN env var not found. Set your primary Hugging Face token.")
        self.client = InferenceClient(api_key=primary_token, timeout=60)

        # Optional fallback model
        fallback_model_name = os.environ.get("FALLBACK_MODEL")  # e.g. "tiiuae/falcon-7b-instruct"
        fallback_token = os.environ.get("FALLBACK_HF_TOKEN") or primary_token
        self.fallback_client = None
        self.fallback_model_name = None
        if fallback_model_name:
            self.fallback_model_name = fallback_model_name
            # We can reuse the same token or use FALLBACK_HF_TOKEN if set
            self.fallback_client = InferenceClient(api_key=fallback_token)

    def decide_actions(self, game_state: State) -> List[Action]:
        """
        Main entrypoint that decides which actions to take.
        1) Build prompt.
        2) Call the first model.
        3) If invalid JSON or fail, call the fallback model if available.
        4) Convert JSON to Action objects.
        5) Return the final list of valid actions.
        """
        action_space_masks = self._compute_action_space_mask(game_state)
        system_instructions = self._build_system_instructions()
        user_prompt = self._build_user_prompt(game_state, action_space_masks)
        final_prompt = system_instructions + "\n\n" + user_prompt

        logger.debug(f"Full prompt for {self.name} (model={self.model_type}):\n{final_prompt}")

        # 1) Attempt the first model
        result = self._call_model(
            prompt=final_prompt,
            model_name=self.model_type,
            client=self.client,
            state=game_state
        )
        if result is not None:
            return result

        # 2) If we get here => first model failed or invalid JSON
        logger.warning(f"{self.name} - primary model {self.model_type} failed. Checking fallback.")
        if self.fallback_client and self.fallback_model_name:
            logger.info(f"Trying fallback model: {self.fallback_model_name} ...")
            fallback_result = self._call_model(
                prompt=final_prompt,
                model_name=self.fallback_model_name,
                client=self.fallback_client,
                state=game_state
            )
            if fallback_result is not None:
                return fallback_result

        # 3) If we get here => fallback also failed or not defined
        logger.error(f"{self.name} - no valid response from LLM. Returning no actions.")
        return []

    def _call_model(
        self,
        prompt: str,
        model_name: str,
        client: InferenceClient,
        state: State
    ) -> Optional[List[Action]]:
        """
        Calls the HF InferenceClient with grammar if supported. If success,
        parse JSON -> actions -> return. If fail or invalid, return None.
        """
        # Attempt inference
        try:
            logger.info(f"[{self.name}] Sending prompt to {model_name} ...")
            # Some open-source models may ignore grammar param. We'll try anyway.
            # If they reject it, we just continue gracefully.
            try:
                completion = client.text_generation(
                    prompt=prompt,
                    model=model_name,
                    max_new_tokens=300,
                    temperature=0.2,
                    return_full_text=False,
                    grammar={"type": "json", "value": LLMActions.schema_json()},  # might get ignored
                    details=True,
                )
            except TypeError:
                # If the 'grammar' argument triggers a TypeError, remove it
                logger.debug("Model likely doesn't support 'grammar' parameter. Retrying without grammar.")
                completion = client.text_generation(
                    prompt=prompt,
                    model=model_name,
                    max_new_tokens=300,
                    temperature=0.2,
                    return_full_text=False,
                    details=True
                )
            except Exception as ex:
                logger.error(f"Moedel recevied ex={str(ex)}")
                logger.info("Exiting ")
                sys.exit()

            # Extract text
            if hasattr(completion, 'generated_text'):
                raw_text = completion.generated_text
            elif isinstance(completion, str):
                raw_text = completion
            else:
                logger.error(f"Unexpected response type from {model_name}: {type(completion)}")
                return None

            logger.info(f"[{self.name}] Raw response from {model_name}:\n{raw_text}")

            # Parse
            try:
                parsed = LLMActions.parse_raw(raw_text)
                logger.debug(f"[{self.name}] Successfully parsed JSON from {model_name}.")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to parse JSON from {model_name}: {e}")
                return None

            # Convert LLMAction -> Action
            return self._convert_llm_actions_to_game_actions(state=state, parsed_actions=parsed)

        except Exception as e:
            logger.error(f"[{self.name}] HF API call to {model_name} failed: {e}")
            return None

    def _convert_llm_actions_to_game_actions(self, state: State, parsed_actions: LLMActions) -> List[Action]:
        """
        Turn the Pydantic list of LLMAction into a list of Monopoly Action objects.
        """        

        if not state:
            logger.error("No current environment state found to interpret actions.")
            return []

        final_actions: List[Action] = []
        for llm_action in parsed_actions.actions:
            action_obj = self._create_action_from_llm_action(llm_action, state)
            if action_obj is not None:
                final_actions.append(action_obj)
            else:
                logger.info(
                    f"LLM requested invalid or un-constructible action: {llm_action.name} "
                    f"params={llm_action.parameters}"
                )
        return final_actions

    # ------------------------------------------------------------------------
    # Internals: computing the action space mask
    # ------------------------------------------------------------------------
    def _compute_action_space_mask(self, state: State) -> Dict[str, Dict[str, Any]]:
        """
        For each action class, retrieve or construct a mask describing valid param ranges.
        """
        action_class_list = HIERARCHICAL_ACTION_CLASSES

        out = {}
        for cls in action_class_list:
            try:
                if cls == BuildAction:
                    # Custom build logic
                    mask = self._compute_build_action_mask(state)
                else:
                    # Generic
                    mask_space = cls.to_action_space_mask(state)  # type: ignore
                    if hasattr(mask_space, "spaces") and mask_space.spaces:
                        mask = {}
                        for k, v in mask_space.spaces.items():
                            desc = {"type": v.__class__.__name__}
                            if hasattr(v, "n"):
                                desc["n"] = v.n
                            if hasattr(v, "low"):
                                desc["low"] = (
                                    v.low.tolist() if hasattr(v.low, "tolist") else v.low
                                )
                            if hasattr(v, "high"):
                                desc["high"] = (
                                    v.high.tolist() if hasattr(v.high, "tolist") else v.high
                                )
                            mask[k] = desc
                    else:
                        mask = {}
                out[cls.__name__] = mask
            except Exception as e:
                logger.warning(f"Failed to compute mask for {cls.__name__}: {e}")
                out[cls.__name__] = {}
        return out

    def _compute_build_action_mask(self, state: State) -> Dict[str, Any]:
        """
        Example: player must own all streets in the color set to build
        """
        current_player = state.players[state.current_player_index]
        buildable_streets = []
        # Attempt to gather all owned streets where entire color_set is owned
        for st in current_player.properties:
            if not isinstance(st, Street):
                continue
            color_set = st.color_set  # Each Street has color_set
            # This requires Board.get_properties_by_color(...) to exist
            all_owned = all(
                (prop.owner == current_player)
                for prop in state.board.get_properties_by_color(color_set)
            )
            if all_owned:
                buildable_streets.append(st.index)

        if not buildable_streets:
            return {}
        max_houses = 5  # up to 4 houses + 1 hotel
        return {
            "street_index": {"type": "Discrete", "n": len(buildable_streets)},
            "quantity": {"type": "Discrete", "n": max_houses + 1},
            "street_indices": buildable_streets
        }

    # ------------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------------
    def _build_system_instructions(self) -> str:
        """
        Returns a system message with strict instructions + a short example.
        """
        example_json = r'''
{
  "actions": [
    {
      "name": "BuyAction",
      "parameters": { "buy": 1 }
    }
  ]
}
        '''.strip()

        return (
            "You are a helpful assistant that outputs valid JSON with the schema:\n"
            "{\"actions\":[{\"name\":\"...\",\"parameters\":{...}}, ...]}.\n"
            "Do not include extra keys or text outside the JSON.\n"
            "Here is a minimal example:\n"
            f"{example_json}\n"
            "Ensure you produce well-formed JSON that Pydantic can parse.\n"
        )

    def _build_user_prompt(
        self,
        game_state: State,
        mask: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Summarize the state, the mask, etc. 
        """
        current_player = game_state.players[game_state.current_player_index]
        balance = current_player.balance
        position = current_player.position
        tile = game_state.board.board[position]
        tile_name = tile.name

        # Gather info about the player's owned streets
        properties_info = []
        for prop in current_player.properties:
            if isinstance(prop, Street):
                properties_info.append({
                    "name": prop.name,
                    "owner": prop.owner.name if prop.owner else None,
                    "houses": prop.houses,
                    "hotels": prop.hotels
                })

        prompt = (
            f"Current Player: {current_player.name} ({current_player.mgn_code}).\n"
            f"Balance: ${balance}\n"
            f"Current Position: {tile_name}\n"
            f"Properties Owned: {json.dumps(properties_info)}\n\n"
            "Available Actions Mask (as JSON):\n"
            f"{json.dumps(mask, indent=2)}\n\n"
            "Decide zero or more valid actions. Output them in the final JSON.\n"
            "No extra text. No partial code. Just valid JSON matching the schema.\n"
        )
        return prompt

    # ------------------------------------------------------------------------
    # Translate LLMAction -> actual Monopoly Action
    # ------------------------------------------------------------------------
    def _create_action_from_llm_action(
        self,
        llm_action: LLMAction,
        state: State
    ) -> Optional[Action]:
        """
        Attempt to create a Monopoly Action from the LLMAction. If invalid,
        return None.
        """
        current_player = state.players[state.current_player_index]
        name_lower = llm_action.name.lower()
        params = llm_action.parameters

        # We do it big if-elif style. 
        try:
            if name_lower == "buyaction":
                buy_decision = params.get("buy", 0)
                if buy_decision != 1:
                    return None
                tile = state.board.board[current_player.position]
                if not isinstance(tile, Property) or tile.owner is not None:
                    return None
                if current_player.balance < tile.purchase_cost:
                    return None
                return BuyAction(player=current_player, property=tile, price=tile.purchase_cost)

            elif name_lower == "auctionaction":
                # param: {"auction": 1}
                if params.get("auction", 0) != 1:
                    return None
                tile = state.board.board[current_player.position]
                if not isinstance(tile, Property) or tile.owner is not None:
                    return None
                return AuctionAction(player=current_player, property=tile)

            elif name_lower == "auctionbidaction":
                # param: {"bid_amount": int}
                bid_amount = params.get("bid_amount")
                if not isinstance(bid_amount, int):
                    return None
                return AuctionBidAction(
                    player=current_player,
                    auction_bid=AuctionBid(bidder=current_player, bid_amount=bid_amount)
                )

            elif name_lower == "auctionfoldaction":
                # param: {"fold": 1}
                if params.get("fold", 0) != 1:
                    return None
                return AuctionFoldAction(player=current_player)

            elif name_lower == "mortgageaction":
                # param: {"property_index": int}
                idx = params.get("property_index")
                if not isinstance(idx, int):
                    return None
                real_prop = next(
                    (p for p in current_player.properties if p.index == idx and not p.is_mortgaged),
                    None
                )
                if real_prop is None:
                    return None
                return MortgageAction(player=current_player, property=real_prop)

            elif name_lower == "unmortgageaction":
                idx = params.get("property_index")
                if not isinstance(idx, int):
                    return None
                real_prop = next(
                    (p for p in current_player.properties if p.index == idx and p.is_mortgaged),
                    None
                )
                if real_prop is None:
                    return None
                return UnmortgageAction(player=current_player, property=real_prop)

            elif name_lower == "buildaction":
                # param: {"street_index": int, "quantity": int}
                st_idx = params.get("street_index")
                qty = params.get("quantity")
                if not isinstance(st_idx, int) or not isinstance(qty, int):
                    return None
                # Check if this street is in buildable_streets
                build_mask = self._compute_build_action_mask(state)
                buildable = build_mask.get("street_indices", [])
                if st_idx not in buildable:
                    return None
                real_street = next(
                    (s for s in current_player.properties if isinstance(s, Street) and s.index == st_idx),
                    None
                )
                if real_street is None:
                    return None
                return BuildAction(player=current_player, street=real_street, quantity=qty)

            elif name_lower == "sellbuildingaction":
                # param: {"street_index": int, "quantity": int}
                st_idx = params.get("street_index")
                qty = params.get("quantity")
                if not isinstance(st_idx, int) or not isinstance(qty, int):
                    return None
                real_street = next(
                    (
                        s for s in current_player.properties
                        if isinstance(s, Street) and s.index == st_idx and (s.houses > 0 or s.hotels > 0)
                    ),
                    None
                )
                if real_street is None:
                    return None
                return SellBuildingAction(player=current_player, street=real_street, quantity=qty)

            elif name_lower == "jailaction":
                # param: {"use_card": bool-int, "pay_fine": bool-int}
                use_card = bool(params.get("use_card", 0))
                pay_fine = bool(params.get("pay_fine", 0))
                return JailAction(player=current_player, use_card=use_card, pay_fine=pay_fine)

            elif name_lower == "tradeaction":
                # param: {"responder_id": int, "offer": {...}, "accept_trade": 1/0}
                r_id = params.get("responder_id")
                offer = params.get("offer", {})
                acc = int(params.get("accept_trade", 0))

                if not isinstance(r_id, int):
                    return None
                if r_id < 0 or r_id >= len(state.players):
                    return None
                responder = state.players[r_id]
                response_str = "A" if acc == 1 else "R"

                # parse the simple 'give' / 'receive'
                give = []
                if "give_cash" in offer and isinstance(offer["give_cash"], int):
                    cval = offer["give_cash"]
                    if cval > 0:
                        give.append(f"${cval}")
                if "give_properties" in offer and isinstance(offer["give_properties"], int):
                    pidx = offer["give_properties"]
                    if any(p.index == pidx for p in current_player.properties):
                        give.append(f"@{pidx}")

                receive = []
                if "receive_cash" in offer and isinstance(offer["receive_cash"], int):
                    cval = offer["receive_cash"]
                    if cval > 0:
                        receive.append(f"${cval}")
                if "receive_properties" in offer and isinstance(offer["receive_properties"], int):
                    pidx = offer["receive_properties"]
                    if any(p.index == pidx for p in responder.properties):
                        receive.append(f"@{pidx}")

                trade_offer = {"give": give, "receive": receive}
                return TradeAction(
                    proposer=current_player,
                    responder=responder,
                    offer=trade_offer,
                    response=response_str
                )

            elif name_lower == "bankruptcyaction":
                # param: {"declare_bankruptcy": 1}
                db = params.get("declare_bankruptcy", 0)
                if db != 1:
                    return None
                return BankruptcyAction(player=current_player, creditor=None)

            else:
                # unrecognized
                logger.debug(f"Unrecognized action name: {llm_action.name}")
                return None

        except Exception as e:
            logger.error(
                f"[{self.name}] Error constructing action from LLM: {llm_action.name} "
                f"params={llm_action.parameters}. Ex={e}"
            )
            return None

        return None
