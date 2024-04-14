# Poker Machine Learning README

Welcome to the Poker Machine Learning project repository! This advanced machine learning endeavor is tailored to unearth optimal strategies for playing heads-up (one-on-one) poker, focusing on Texas Hold'em – the most popular variation of poker. This project has been meticulously crafted to bridge the gap between theoretical machine learning algorithms and their practical application in game theory and decision-making under uncertainty.

## Project Overview

The aim of the project is to develop a robust AI poker player capable of competing against human players and other AI contestants. Through a combination of hand-crafted feature extraction, state-of-the-art machine learning models, and rigorous training/testing procedures, this project delves into the complex and intriguing world of poker strategy.

The repository is organized into several vital components:

- `Models/`: Contains the trained machine learning models.
- `Player/`: Houses the bot logic and decision-making algorithms.
- `Utils/`: Includes various utilities for data processing and analytics.
- `Game_setup.py`: Configuration and setup script for initializing game parameters.

## Features

- **Advanced Feature Extraction**: A sophisticated system for processing the game state, distilling vital information such as normalized stack sizes, hole card strengths, and pot sizes to inform AI decisions.
- **Machine Learning Integration**: Machine learning models developed with PyTorch are integrated to determine the best course of action given the current game situation.
- **Optimized Strategy**: Through rigorous training, the AI players adjust their strategies to maximize expected value and adopt the most effective tactics against a wide range of opponents.
- **Simulated Testing Environment**: Comprehensive testing frameworks to evaluate AI performance against various playstyles and scenarios.
  
## Models Documentation

The machine learning models are the core of this project. They have been trained on exhaustive datasets that encapsulate numerous poker scenarios and strategic considerations, thus encapsulating the essence of human-like intuition and decision-making in a poker game.

## Getting Started

Refer to `game_setup.py` for basic configurations to set up the initial game parameters.


## Contributions

This project welcomes contributions from the community! Whether it's improving the models, refining feature extraction techniques, or proposing entirely new algorithms – your insights can be a catalyst for progress.

## Further Work

The field of AI in poker is continuously evolving. Future directions of this project may include the implementation of reinforcement learning techniques, analysis of multi-player scenarios, and real-time adaptation algorithms.

## Authors

- **Trevor Poon** - _Initial work_

## Acknowledgments

A heartfelt thank you to the AI research community, whose open-source contributions and insightful research papers have been the guiding light for this project.

---

For any inquiries or suggestions, please open an issue in this repository or reach out directly. We hope you find this work both insightful and inspiring as you embark on your AI journey in the realm of strategic gaming! Enjoy the challenge, and may the odds be in your favor.