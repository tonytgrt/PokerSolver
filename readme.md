# AI Poker Win Rate Estimator

## Overview

This project is an AI-based tool designed to assist with poker decision-making by estimating the player's win rate based on the current game state. The tool accepts inputs such as the player's hole cards, community cards, number of players, and betting actions. It uses a Monte Carlo simulation to estimate the win rate. Additionally, it includes a neural network model called **OpponentCardPredictor** that is intended to predict opponents' hole cards based on their betting behavior. Currently, the opponent card prediction model remains a placeholder and has not yet been trained on real data, but the win rate simulation is fully functional.

## Features

- **Win Rate Simulation:**  
  Uses Monte Carlo simulations to complete the board, evaluate hands, and determine the probability that the player's hand will win against opponents.

- **Opponent Card Prediction (Placeholder):**  
  Implements a PyTorch neural network that outputs a probability distribution over 1,326 unique two-card combinations based on encoded betting features. This model is designed with GTO (Game Theory Optimal) principles in mind but is currently untrained and serves as a placeholder.

- **Game State Management:**  
  Manages multiple rounds of betting, tracks active players, and infers player actions (fold, check, call, raise) based on bet amounts.

- **Hand Evaluation:**  
  Evaluates poker hands and produces descriptive rankings such as "Ace High Card", "Pair of 10s", and "Jack High Diamond Flush".

## How to Run

1. Ensure you have Python installed along with the required libraries (e.g., PyTorch, NumPy, Pandas).
2. Run the main script (`main.py`) to launch the AI Poker Win Rate Estimator.
3. Follow the on-screen prompts to enter:
   - The number of players on the table.
   - Your position (0-indexed).
   - Your hole cards.
   - Community cards (if any).
   - Betting amounts for each active player.
4. The program will display a game summary that includes:
   - The current game stage.
   - Your best hand evaluation.
   - The estimated win rate.
   - Predicted opponents' hole cards (from the placeholder model).

## Models Used

- **OpponentCardPredictor:**  
  A PyTorch neural network that consists of three fully connected layers with ReLU activations and a softmax output. It takes a 10-dimensional feature vector (which encodes an opponent’s betting behavior) as input and outputs a probability distribution over 1,326 possible two-card combinations. Note that this model is currently a placeholder and has not yet been trained on real poker data.

- **Monte Carlo Win Rate Simulator:**  
  A simulation module that randomly completes the board (if necessary), evaluates the player's hand against randomly generated opponents' hands, and computes the win rate over multiple simulations.

## Current Status

- The win rate simulation is fully implemented and operational.
- The opponent card prediction model is implemented as a neural network but is currently a placeholder and has yet to be trained on real data.

## Future Work

- Train the **OpponentCardPredictor** model using a comprehensive dataset of historical poker hands and betting actions based on GTO principles.
- Enhance the feature encoding to include additional context such as player position, pot size, and betting history.
- Refine the hand evaluation logic for more accurate hand strength assessments.

## License

This project is licensed under the MIT License.
