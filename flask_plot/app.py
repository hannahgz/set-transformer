from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass
import torch
import os
from model import GPT, GPTConfig44_BalancedSets, GPTConfig44_Final
from tokenizer import load_tokenizer
import random

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
    # return [[[np.random.rand(10, 10) for _ in range(4)] for _ in range(4)] for _ in range(2)]
    # return np.random.rand(2, 4, 4, 49, 49)
    return np.zeros((2, 4, 4, 49, 49))

def get_labels():
    return ['Label' + str(i) for i in range(49)], ['Label' + str(i) for i in range(49)]

def shuffle_input(input):
    # Step 1: Group elements into pairs
    pairs = [(input[i], input[i + 1]) for i in range(0, len(input), 2)]

    # Step 2: Shuffle the pairs
    random.shuffle(pairs)

    # Step 3: Flatten the shuffled pairs back into a sequence
    shuffled_input = [item for pair in pairs for item in pair]
    return shuffled_input

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
    # input = shuffle_input(input)
    input.append(">")
    input.extend(
        get_target_seq(combination, target_size=8, pad_symbol="_")
    )
    return input


def attention_weights_from_sequence(
        config,
        input,
        tokenizer_path=f"larger_balanced_set_dataset_random_tokenizer.pkl",
        get_prediction=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # # Restore the model state dict
    # checkpoint = torch.load(os.path.join(
    #     PATH_PREFIX, config.filename), weights_only=False, map_location=torch.device('cpu'))
    
    checkpoint = torch.load(config.filename, weights_only=False, map_location=torch.device('cpu'))

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
        is_correct = bool(matches.sum().item())

        decoded_predictions = tokenizer.decode(predictions[0].tolist())
        decoded_targets = tokenizer.decode(targets[0].tolist())
        print("correct: ", is_correct)

        # print("full output: ", tokenizer.decode(outputs[0].tolist()))
        print("predictions: ", decoded_predictions)
        print("target: ", decoded_targets)

    _, _, attention_weights, _ = model(
        sequences.to(device), False)
    
    all_att_weights_np = []
    for layer in range(4):
        layer_weights = []  # Create a sublist for each layer
        for head in range(4):
            # Append the attention weights for the current head to the layer sublist
            layer_weights.append(attention_weights[layer][0][head].detach().cpu().numpy())
        all_att_weights_np.append(layer_weights)  # Append the layer sublist to the main list

    if get_prediction:
        return all_att_weights_np, is_correct, decoded_predictions, decoded_targets
    
    return all_att_weights_np

# def create_attention_weight_fig(attention_weights, labels1, labels2, n_layers=4, n_heads=4, threshold=0.01):
#     fig = make_subplots(rows=n_layers, cols=n_heads, 
#                         subplot_titles=[f"Layer {l+1} Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
#                         specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)],
#                         vertical_spacing=0.03)
    
#     # Pre-calculate x coordinates
#     x_coords = [0, 1]

#     # Set range for both axes
#     y_min = -0.5  # Add some padding
#     y_max = len(labels1) - 0.5  # Add some padding
    

#     reversed_labels = labels1[::-1]

#     highlight_indices = {len(labels1) - i - 1: 'green' for i, (label1, label2) in enumerate(zip(labels1, labels2)) if label1 != label2}

#     custom_ticktext = []
#     for i, label in enumerate(reversed_labels):
#         if i in highlight_indices and i > 8:
#             # Use HTML color tag
#             custom_ticktext.append(f'<span style="color: {highlight_indices[i]}">{label}</span>')
#         else:
#             custom_ticktext.append(label)
    
#     for layer in range(n_layers):
#         for head in range(n_heads):
#             # Get the attention weights for this layer and head
#             weights_diff = attention_weights[0][layer][head] - attention_weights[1][layer][head]
            
#             # Create coordinate arrays for all possible connections
#             n_points = len(labels1)
#             all_connections = []
            
#             # Generate all possible connections
#             for i in range(n_points):
#                 for j in range(n_points):
#                     weight = abs(weights_diff[i, j])
#                     if weight > threshold:
#                         # Flip the y-coordinates to match natural reading direction (top-to-bottom)
#                         all_connections.append({
#                             'x': x_coords,
#                             'y': [n_points - 1 - i, n_points - 1 - j],  # Flip the y coordinates
#                             'weight': weight,
#                             'from_idx': i,  # Store original indices for hover text
#                             'to_idx': j
#                         })
            
#             # Sort connections by weight to draw strongest connections last
#             # all_connections.sort(key=lambda x: x['weight'])
            
#             # Add traces for each significant connection
#             for conn in all_connections:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=conn['x'],
#                         y=conn['y'],
#                         mode='lines',
#                         line=dict(
#                             color='blue',
#                             width=max(1.0, conn['weight'] * 5)
#                         ),
#                         opacity=conn['weight'],
#                         showlegend=False,
#                         hovertemplate=f"Weight: {conn['weight']:.3f}<br>From: {labels1[conn['from_idx']]}<br>To: {labels1[conn['to_idx']]}",
#                         name="",
#                         hoveron='points+fills',
#                     ),
#                     row=layer+1, col=head+1,
#                     secondary_y=False
#                 )
        
#             if len(all_connections) == 0:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=[None],
#                         y=[None],
#                         mode='lines',
#                         line=dict(color='blue', width=1),
#                         opacity=0,
#                         showlegend=False,
#                         hoverinfo='skip'
#                     ),
#                     row=layer+1, col=head+1,
#                     secondary_y=False
#                 )

#             # Add dummy trace for secondary y-axis to ensure labels are shown
#             fig.add_trace(
#                 go.Scatter(
#                     x=[None],
#                     y=[None],
#                     mode='lines',
#                     line=dict(color='blue', width=1),
#                     opacity=0,
#                     showlegend=False,
#                     hoverinfo='skip'
#                 ),
#                 row=layer+1, col=head+1,
#                 secondary_y=True
#             )
            
#             # Update axes (doing this only once per subplot)
#             fig.update_xaxes(
#                 ticktext=["Query", "Key"],
#                 tickvals=[0, 1],
#                 row=layer+1,
#                 col=head+1,
#                 range=[-0.1, 1.1],
#                 showline=False,
#                 showgrid=False  # Disable grid to reduce rendering overhead
#             )

#             # for secondary_y in [False, True]:
#             #     fig.update_yaxes(
#             #         # ticktext=reversed_labels,
#             #         ticktext=custom_ticktext,
#             #         tickvals=list(range(len(labels1))),
#             #         range=[y_min, y_max],
#             #         secondary_y=secondary_y,
#             #         row=layer+1,
#             #         col=head+1,
#             #         showline=False,
#             #         showgrid=False,
#             #         side='left' if not secondary_y else 'right'
#             #     )

#             for secondary_y in [False, True]:
#                 fig.update_yaxes(
#                     ticktext=custom_ticktext,
#                     tickvals=list(range(len(labels1))),
#                     range=[y_min, y_max],
#                     secondary_y=secondary_y,
#                     row=layer+1,
#                     col=head+1,
#                     showline=False,
#                     showgrid=False,
#                     side='left' if not secondary_y else 'right',
#     )

#     # Optimize layout
#     fig.update_layout(
#         height=n_layers*300,
#         width=n_heads*250,
#         # title_text="Attention Line Pattern Differences (Group 1 minus Group 2)",
#         showlegend=False,
#     )

#     return fig

def create_attention_weight_fig(attention_weights, labels1, labels2, n_layers=4, n_heads=4, threshold=0.01):
    # Calculate a more compact height based on screen size
    height_per_layer = 400  # Reduced from 300
    total_height = n_layers * height_per_layer
    
    fig = make_subplots(
        rows=n_layers, 
        cols=n_heads,
        subplot_titles=[f"L{l+1}H{h+1}" for l in range(n_layers) for h in range(n_heads)],  # Shortened titles
        specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)],
        vertical_spacing=0.02,  # Reduced spacing between subplots
        horizontal_spacing=0.02  # Reduced horizontal spacing
    )
    
    x_coords = [0, 1]
    y_min = -0.5
    y_max = len(labels1) - 0.5
    
    reversed_labels = labels1[::-1]
    highlight_indices = {len(labels1) - i - 1: 'green' for i, (label1, label2) in enumerate(zip(labels1, labels2)) if label1 != label2}

    # Make the font size smaller
    custom_ticktext = []
    for i, label in enumerate(reversed_labels):
        if i in highlight_indices and i > 8:
            custom_ticktext.append(f'<span style="color: {highlight_indices[i]};font-size:10px">{label}</span>')
        else:
            custom_ticktext.append(f'<span style="font-size:10px">{label}</span>')
    
    for layer in range(n_layers):
        for head in range(n_heads):
            weights_diff = attention_weights[0][layer][head] - attention_weights[1][layer][head]
            n_points = len(labels1)
            all_connections = []
            
            for i in range(n_points):
                for j in range(n_points):
                    weight = abs(weights_diff[i, j])
                    if weight > threshold:
                        all_connections.append({
                            'x': x_coords,
                            'y': [n_points - 1 - i, n_points - 1 - j],
                            'weight': weight,
                            'from_idx': i,
                            'to_idx': j
                        })
            
            for conn in all_connections:
                fig.add_trace(
                    go.Scatter(
                        x=conn['x'],
                        y=conn['y'],
                        mode='lines',
                        line=dict(
                            color='blue',
                            width=max(0.5, conn['weight'] * 3)  # Reduced line width
                        ),
                        opacity=conn['weight'],
                        showlegend=False,
                        hovertemplate=f"Weight: {conn['weight']:.3f}<br>From: {labels1[conn['from_idx']]}<br>To: {labels1[conn['to_idx']]}",
                        name="",
                        hoveron='points+fills',
                    ),
                    row=layer+1, col=head+1,
                    secondary_y=False
                )
        
            if len(all_connections) == 0:
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode='lines', 
                             line=dict(color='blue', width=1), opacity=0, 
                             showlegend=False, hoverinfo='skip'),
                    row=layer+1, col=head+1, secondary_y=False
                )

            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode='lines',
                          line=dict(color='blue', width=1), opacity=0,
                          showlegend=False, hoverinfo='skip'),
                row=layer+1, col=head+1, secondary_y=True
            )
            
            # Update x-axes with smaller font and ticks
            fig.update_xaxes(
                ticktext=["Q", "K"],  # Shortened labels
                tickvals=[0, 1],
                row=layer+1,
                col=head+1,
                range=[-0.1, 1.1],
                showline=False,
                showgrid=False,
                tickfont=dict(size=4)  # Smaller font size
            )

            # Update y-axes with smaller font and optimized spacing
            for secondary_y in [False, True]:
                fig.update_yaxes(
                    ticktext=custom_ticktext,
                    tickvals=list(range(len(labels1))),
                    range=[y_min, y_max],
                    secondary_y=secondary_y,
                    row=layer+1,
                    col=head+1,
                    showline=False,
                    showgrid=False,
                    side='left' if not secondary_y else 'right',
                    tickfont=dict(size=2),  # Smaller font size
                )

    # Optimize layout
    fig.update_layout(
        height=total_height,
        width=n_heads*250,  # Reduced width
        showlegend=False,
        margin=dict(l=50, r=50, t=10, b=20),  # Optimized margins
        title_x=0.5,
        # font=dict(size=4)  # Global font size reduction
    )

    return fig

@app.route('/')
def index():
    # Get data for the heatmap
    labels1, labels2 = get_labels()
    fig = create_attention_weight_fig(
        attention_weights=get_attention_weights(),
        labels1=labels1,
        labels2=labels2
    )

    card_labels = [chr(65 + i) for i in range(5)]  # ['A', 'B', 'C', 'D', 'E']

    plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    return render_template('index.html', 
                            shapes=shapes,
                            colors=colors,
                            numbers=numbers,
                            shadings=shadings,
                            plot_json=plot_json,
                            saved_mappings=saved_card_mappings,
                            card_labels=card_labels,
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
    # sequence1 = [
    #     "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "A", "B", "C", ".", "_", "_", "_", "_"
    # ]
    sequence2 = generate_input(card_groups, "group2")

    # attention_weights1, is_correct1, decoded_predictions1, decoded_targets1 = attention_weights_from_sequence(
    #     GPTConfig44_BalancedSets, sequence1, get_prediction=True)
    
    attention_weights1, is_correct1, decoded_predictions1, decoded_targets1 = attention_weights_from_sequence(
        GPTConfig44_Final, sequence1, tokenizer_path="final_causal_balanced_tokenizer.pkl", get_prediction=True)
    print("Got attention weights 1")

    # attention_weights2, is_correct2, decoded_predictions2, decoded_targets2 = attention_weights_from_sequence(
    #     GPTConfig44_BalancedSets, sequence2, get_prediction=True)
    attention_weights2, is_correct2, decoded_predictions2, decoded_targets2 = attention_weights_from_sequence(
        GPTConfig44_Final, sequence2, tokenizer_path="final_causal_balanced_tokenizer.pkl", get_prediction=True)
    print("Got attention weights 2")

    prediction_results = {
        1: (sequence1, is_correct1, decoded_predictions1, decoded_targets1),
        2: (sequence2, is_correct2, decoded_predictions2, decoded_targets2)
    }

    fig = create_attention_weight_fig(
        attention_weights=[attention_weights1, attention_weights2],
        labels1=sequence1,
        labels2=sequence2,
        threshold=card_groups.get('threshold', 0.1)
    )
    print("Created attention weight fig")

    plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    print(("Constructed plot JSON"))

    print("returning sucess json")
    return jsonify({
        "status": "success",
        "cards": mapping,
        "prediction_results": prediction_results,
        "plot_json": plot_json
    })



if __name__ == '__main__':
    app.run(debug=True)