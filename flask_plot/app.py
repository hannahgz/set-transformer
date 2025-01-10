from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass
import torch
import os
from model import GPT, GPTConfig44_BalancedSets
from tokenizer import load_tokenizer

app = Flask(__name__)

PATH_PREFIX = "/Users/I835284/Desktop/thesis/set-transformer/src"

# Card attributes
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

# Store the saved cards (in a real application, you'd want to use a database)
saved_card_mappings = []

card_vectors = ["A", "B", "C", "D", "E"]

def is_set(card1, card2, card3):
    return all((a + b + c) % 3 == 0 for a, b, c in zip(card1, card2, card3))

def find_sets_with_cards(combination: Tuple) -> List[Tuple]:
    sets = []
    n_cards = len(combination)
    # print("combination: ", combination)
    for i in range(n_cards - 2):
        for j in range(i + 1, n_cards - 1):
            for k in range(j + 1, n_cards):
                if is_set(combination[i], combination[j], combination[k]):
                    # print("appending: ", (card_vectors[i], card_vectors[j], card_vectors[k]))
                    sets.extend(
                        [card_vectors[i], card_vectors[j], card_vectors[k]])

    if len(sets) == 6:
        sets.insert(3, "/")

    if len(sets) == 0:
        sets.append("*")

    return sets

def get_target_seq(combination, target_size, pad_symbol):
    target_seq = find_sets_with_cards(combination)

    target_seq.append(".")

    for _ in range(target_size - len(target_seq)):
        target_seq.append(pad_symbol)

    return target_seq

def map_card_to_letter(card_num):
    return chr(65 + card_num)  # 0->A, 1->B, 2->C, etc.

def get_attention_weights():
    return [np.random.rand(10, 10) for _ in range(2)]

def get_labels():
    return ['Label' + str(i) for i in range(10)], ['Label' + str(i) for i in range(10, 20)]

def generate_input(card_groups, group):
    input = []
    combination = []
    for i, card in enumerate(card_groups.get(group, [])):
        letter = map_card_to_letter(i)  # Assuming letters are assigned sequentially

        # ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 1))
        for _, value in card.items():
            input.append(letter)
            input.append(value)

        combination.append((shapes.index(card['shape']), colors.index(card['color']), numbers.index(card['number']), shadings.index(card['shading'])))
    
    combination = tuple(combination)
    input.append(">")
    input.extend(
        get_target_seq(combination, target_size=8, pad_symbol="_")
    )
    return input


def attention_weights_from_sequence(
        config,
        input,
        tokenizer_path=f"{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl",
        get_prediction=False,
        filename_prefix=""):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # Restore the model state dict
    checkpoint = torch.load(os.path.join(
        PATH_PREFIX, config.filename), weights_only=False, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model"])
    print("Loaded model")

    tokenizer = load_tokenizer(tokenizer_path)
    input = torch.tensor(tokenizer.encode(input))
    sequences = input.unsqueeze(0)

    if get_prediction:
        inputs = sequences[:, : config.input_size].to(device)
        targets = sequences[:, config.input_size:].to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=config.target_size)

        predictions = outputs[:, config.input_size:]

        mask = targets != config.padding_token  # Create a mask to ignore padding
        matches = ((predictions == targets) | ~mask).all(dim=1)
        print("correct: ", matches.sum().item())

        print("full output: ", tokenizer.decode(outputs[0].tolist()))
        print("predictions: ", tokenizer.decode(predictions[0].tolist()))
        print("target: ", tokenizer.decode(targets[0].tolist()))

    _, _, attention_weights, _ = model(
        sequences.to(device), False)
    
    all_att_weights_np = []
    for layer in range(4):
        for head in range(4):
            all_att_weights_np.append(attention_weights[layer][0][head].detach().cpu().numpy())
    
    return all_att_weights_np


@app.route('/')
def index():
    # Get data for the heatmap
    attention_weights = get_attention_weights()
    labels1, labels2 = get_labels()
    n_layers, n_heads = 4, 4
    
    fig = make_subplots(rows=n_layers, cols=n_heads, 
                        subplot_titles=[f"Layer {l+1} Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
                        specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)])
    
    for layer in range(n_layers):
        for head in range(n_heads):
            attention_weights_diff = attention_weights[0] - attention_weights[1]
            
            for i in range(len(labels1)):
                for j in range(len(labels1)):
                    weight = attention_weights_diff[i, j]
                    fig.add_trace(
                        go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
                                   opacity=abs(weight), showlegend=False, hoverinfo='x+y'),
                        row=layer+1, col=head+1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
                                   opacity=0, showlegend=False, hoverinfo='x+y'),
                        row=layer+1, col=head+1,
                        secondary_y=True,
                    )
            
            fig.update_xaxes(ticktext=["Query", "Key"], tickvals=[0, 1], row=layer+1, col=head+1)
            
            fig.update_yaxes(
                ticktext=labels1, 
                tickvals=list(range(len(labels1))), 
                secondary_y=False,
                row=layer+1, 
                col=head+1
            )
            
            fig.update_yaxes(
                ticktext=labels2,
                tickvals=list(range(len(labels2))),
                secondary_y=True,
                row=layer+1, 
                col=head+1,
            )

    fig.update_layout(
        height=n_layers*400, 
        width=n_heads*300, 
        title_text="Attention Line Pattern Differences",
    )
    
    plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    return render_template('index.html', 
                            shapes=shapes,
                            colors=colors,
                            numbers=numbers,
                            shadings=shadings,
                            plot_json=plot_json,
                            saved_mappings=saved_card_mappings,
                            chr=chr)
@app.route('/save_cards', methods=['POST'])
def save_cards():
    card_groups = request.json  # Expecting data in the format {'group1': [...], 'group2': [...]}

    # Initialize the mappings for both groups
    mapping = {"group1": [], "group2": []}
    
    # Process Group 1
    for i, card in enumerate(card_groups.get('group1', [])):
        letter = map_card_to_letter(i)  # Assuming letters are assigned sequentially
        card_info = {
            'letter': letter,
            'attributes': card
        }
        mapping['group1'].append(card_info)

    # Process Group 2
    for i, card in enumerate(card_groups.get('group2', [])):
        letter = map_card_to_letter(i)  # Offset letters for group 2
        card_info = {
            'letter': letter,
            'attributes': card
        }
        mapping['group2'].append(card_info)
    
    # Store the mappings (in a real application, you'd want to use a database)
    global saved_card_mappings
    saved_card_mappings = mapping
    
    sequence1 = generate_input(card_groups, "group1")
    sequence2 = generate_input(card_groups, "group2")

    attention_weights1 = attention_weights_from_sequence(
        GPTConfig44_BalancedSets, sequence1, get_prediction=True)
    
    return jsonify({
        "status": "success",
        "cards": mapping,
        "input1": sequence1,
        "input2": sequence2,
        "attention_weights1": attention_weights1
    })


if __name__ == '__main__':
    app.run(debug=True)