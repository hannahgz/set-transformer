# SET Transformer
Explore relational composition using toy models of the pattern-matching card game SET. 

Understanding if and how neural networks have the ability to reason through problems is a fundamental open question that will help us govern and evaluate the future of large language models (LLMs). In this repository, we provide code that can be used to explore the concept of binding—a crucial aspect of reasoning that examines how models relate and connect ideas to one another. 

The thesis uses toy transformers trained on a custom task based on the **pattern-matching card game SET** to evaluate different mechanisms for relational composition, a more specific notion of binding tailored to neural networks concerned with the relationship between feature vectors and if they are bound together to represent more complicated ideas.

Code developed as part of a senior thesis in Computer Science at Harvard College with the Insight and Interaction Lab. Advised by Professor Martin Wattenberg. Thank you to Andrew Lee for his advice on the transformer model.

Contact Hannah Zhou hannah.g.zhou@gmail.com with any questions.

This repository can be broken into three main components:
1. `src_clean`: Main transformer model trained to play 5-card SET variant. 
    1. `new_datasets`: Different seeds of the transformer model above.
2. `flask_plot`: Visualizer web tool used to visualize attention patterns generated from transformer model.
3. `mlp`: Feed-forward neural network basic example of SET card game.

<!-- Summary/Abstract (from thesis):  -->
>
## SET Transformer Model

We outline the development of a toy model of the card game SET that is tailored
to investigate binding. We structure our SET task so the model must learn to accurately make
predictions following the rules of the game while also having to perform binding in multiple different instances. 


## Visualizer Web Application

This folder contains a web application for visualizing how a transformer model identifies "sets" in a card game. The application allows users to select cards with different attributes, visualize attention patterns in the model, and observe how the model's predictions change with different card combinations.

### Overview

The application uses a Flask backend with a transformer model implemented in PyTorch to analyze sets of cards. A "set" consists of three cards where for each attribute (shape, color, number, shading), all cards have either all the same value or all different values.

### Files

- **app.py**: Flask application that handles routing and API endpoints. Contains functions for processing card data, generating attention visualizations, and interfacing with the model.

- **model.py**: Contains the transformer model implementation (GPT architecture) with various configurations for different layer and head counts.

- **tokenizer.py**: Simple tokenizer for encoding/decoding sequences of card attributes.

- **index.html**: Main web interface with card selectors and visualization components.

- **requirements.txt**: Dependencies for the application.

- **runtime.txt**: Specifies Python version for deployment.

### Model Architecture

The model uses a GPT-style transformer architecture with several configurable parameters:
- 2-4 transformer layers
- 4 attention heads per layer
- 64-dimensional embeddings
- Token and position embeddings

### Features

1. **Card Selection**: Users can create and manipulate two groups of cards with attributes:
  - Shape (oval, squiggle, diamond)
  - Color (green, blue, pink) 
  - Number (one, two, three)
  - Shading (solid, striped, open)

2. **Attention Visualization**: Interactive visualization of attention patterns across layers and heads.

3. **Set Detection**: The model tries to identify whether the selected cards form valid "sets".

4. **Comparative Analysis**: Visualize the difference in attention patterns between two different card combinations.

5. **Configuration Options**:
  - Adjust visualization threshold
  - Change random seed for shuffling input tokens
  - Select number of model layers
  - Choose between different model variants
  - Toggle value weighting for attention visualization

### Setup and Running

1. Install dependencies: `pip install -r requirements.txt`

2. Run the application: `python app.py`

3. Access the web interface at http://localhost:8000

### Usage

1. Select card attributes using dropdown menus in the left panel
2. Use buttons to save selected cards, duplicate between groups, or randomize
3. Adjust visualization parameters as needed
4. Observe the model's predictions and attention patterns in the display area

The application helps visualize how the transformer model processes information to identify valid sets in the card game, demonstrating attention mechanisms in practice.

## MLP

As a motivating example for using SET as a basis for this thesis, we present a simple feed-forward
network trained to play a simulated version of the SET game.
