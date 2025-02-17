import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

# --------------------------
# 1. Opponent Range Estimation Model (Placeholder for later use)
# --------------------------
class OpponentRangeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OpponentRangeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Using sigmoid to produce outputs between 0 and 1 (a dummy "range strength")
        x = torch.sigmoid(self.fc2(x))
        return x

# --------------------------
# Helper: Determine the current stage of the game
# --------------------------
def determine_stage(community_cards):
    """Determine the current stage of the game based on the number of community cards."""
    if len(community_cards) == 0:
        return "Pre-Flop"
    elif len(community_cards) == 3:
        return "Flop"
    elif len(community_cards) == 4:
        return "Turn"
    elif len(community_cards) == 5:
        return "River"
    else:
        return "Unknown Stage"

# --------------------------
# Helper: Infer player action from bet amounts
# --------------------------
def infer_actions(betting_order, bets):
    """
    Infers each player's action based on their bet.
    If a player bets -1, that's a fold.
    Otherwise, if the bet equals the current call amount, it is a check (if 0) or call.
    If the bet is higher than the current call amount, it's a raise (and updates the call amount).
    """
    current_call = 0.0
    actions = {}
    for pos in betting_order:
        bet = bets[pos]
        if bet == -1:
            actions[pos] = "fold"
        elif bet == current_call:
            if current_call == 0:
                actions[pos] = "check"
            else:
                actions[pos] = "call"
        elif bet > current_call:
            actions[pos] = "raise"
            current_call = bet
        else:
            actions[pos] = "call"
    return actions

# --------------------------
# Main Loop: Run rounds continuously
# --------------------------
def main():
    print("Welcome to the AI Poker Win Rate Estimator!")
    print("Note: Use -1 as the bet amount to indicate a fold.")
    print("Positions: Dealer is player 0, Small Blind is player 1, Big Blind is player 2.\n")

    # Game Setup: Get number of players and user position (0-indexed)
    num_players = int(input("Enter the number of players on the table: "))
    user_position = int(input("Enter your position (0-indexed): "))

    # Validate minimum number of players (at least 3 are needed for Texas Hold'em)
    if num_players < 3:
        print("Error: Texas Hold'em requires at least 3 players.")
        return

    # Get the user's cards once (memorized for subsequent rounds)
    user_cards_input = input("Enter your cards separated by a comma (e.g., AH, KD): ")
    user_cards = [card.strip().upper() for card in user_cards_input.split(",") if card.strip()]

    round_counter = 1
    # Initialize active players for the hand: all players start active
    active_players = list(range(num_players))

    while True:
        print("\n=== Round {} ===".format(round_counter))
        # If only one player remains, the hand is over.
        if len(active_players) == 1:
            print(f"Hand over! Only Player {active_players[0]} remains active.")
            # Option to start a new hand (reset active players)
            new_hand = input("Start a new hand? (y/n): ")
            if new_hand.lower() != 'y':
                print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
                break
            else:
                # Reset active players for a new hand.
                active_players = list(range(num_players))
                round_counter = 1
                # In a new hand, community cards are cleared.
                continue

        # Get community cards (if any) for the current round
        community_cards_input = input("Enter community cards separated by a comma (if none, leave blank): ")
        community_cards = [card.strip().upper() for card in community_cards_input.split(",") if card.strip()]

        # Determine current stage based on community cards
        stage = determine_stage(community_cards)

        # Determine betting order (full order based on stage) then filter to active players.
        if stage == "Pre-Flop":
            full_order = list(range(3, num_players)) + [0, 1, 2]
        else:
            full_order = list(range(1, num_players)) + [0]
        betting_order = [pos for pos in full_order if pos in active_players]

        # Get bets from players in the proper order (only for active players)
        bets = {}
        print("\n--- Enter Bets ---")
        for pos in betting_order:
            if pos == user_position:
                bet_input = input(f"Player {pos} (You) - Enter your bet amount: ")
            else:
                bet_input = input(f"Player {pos} - Enter bet amount: ")
            try:
                bet = float(bet_input)
            except:
                bet = 0.0
            bets[pos] = bet

        # Infer actions based on the bets and betting order
        actions = infer_actions(betting_order, bets)

        # Update active players: remove any player who just folded
        for pos in betting_order:
            if actions[pos] == "fold" and pos in active_players:
                active_players.remove(pos)

        # Compute the current total cumulated bet for this round (ignoring folds)
        cumulated_bet = sum(bet for bet in bets.values() if bet != -1)

        # Build players_info only for active players
        players_info = {}
        for pos in active_players:
            # Use current round info if available, otherwise mark as N/A
            players_info[pos] = {
                "bet": bets.get(pos, "N/A"),
                "action": actions.get(pos, "N/A")
            }

        # Placeholder values for computed outputs
        placeholder_win_rate = 50.0  # Dummy win rate (50%)
        opponents_predictions = {}
        for pos in active_players:
            if pos != user_position:
                opponents_predictions[pos] = ["??", "??"]

        # Print Game Summary for the round
        print("\n--- Game Summary ---")
        print(f"Current Stage: {stage}")
        print(f"User Position: {user_position}")
        print(f"User Cards: {user_cards}")
        print("Community Cards:", community_cards if community_cards else "None")
        print(f"Estimated Win Rate: {placeholder_win_rate:.2f}%")
        print(f"Total Cumulated Bet (Pot): {cumulated_bet}\n")

        print("Active Players Info:")
        for pos in active_players:
            if pos == user_position:
                print(f"  Position {pos} (You) - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}")
            else:
                pred = opponents_predictions.get(pos, ["??", "??"])
                print(f"  Position {pos} - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}, Predicted Cards: {pred}")

        # Ask if the user wants to continue to the next betting round of the current hand.
        cont = input("\nContinue to next betting round? (y/n): ")
        if cont.lower() != 'y':
            print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
            break

        round_counter += 1

if __name__ == "__main__":
    main()
