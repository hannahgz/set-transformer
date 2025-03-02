from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass
import torch
import os
from model import GPT, GPTConfig44_Complete, GPTConfig34_Complete, GPTConfig44_SeededOrigDataset
from tokenizer import load_tokenizer
import random

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

PATH_PREFIX = "/Users/I835284/Desktop/thesis/set-transformer/flask_plot/models"

# Card attributes
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

# Store the saved cards (in a real application, you'd want to use a database)
saved_card_mappings = []

card_vectors = ["A", "B", "C", "D", "E"]

config = GPTConfig34_Complete()

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
    return [str(i) for i in range(49)], [str(i) for i in range(49)]


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
        # Assuming letters are assigned sequentially
        letter = map_card_to_letter(i)

        # ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 1))
        for _, value in card.items():
            input.append(letter)
            input.append(value)

        combination.append((shapes.index(card['shape']), colors.index(
            card['color']), numbers.index(card['number']), shadings.index(card['shading'])))

    combination = tuple(combination)
    # input = shuffle_input(input)
    input.append(">")
    input.extend(
        get_target_seq(combination, target_size=8, pad_symbol="_")
    )
    return input


def check_sequence_equality(prediction, target):
    """
    Check if two sequences are equivalent, accounting for:
    1. Different orderings of groups (for two triplet cases)
    2. Different orderings within each triplet
    3. Single triplet cases with padding
    4. Different padding tokens ('.' vs '_')
    5. No triplet cases (just "*" and padding)

    Args:
        prediction: List of strings, e.g. ["A", "B", "C", ".", ".", ".", ".", "."]
                   or ["A", "B", "C", "/", "D", "E", "F", "."]
                   or ["*", ".", ".", ".", ".", ".", ".", "."]
        target: List of strings in same format

    Returns:
        bool: True if sequences are equivalent, False otherwise
    """
    def get_triplets(sequence):
        # First check if this is a "no triplet" case
        if "*" in sequence:
            return []

        # Find the separator index if it exists
        try:
            sep_idx = sequence.index("/")
            # Split into two triplets
            first_triplet = set(sequence[:sep_idx])
            second_triplet = set(sequence[sep_idx + 1:])
            # Remove padding tokens
            first_triplet = {x for x in first_triplet if x not in [".", "_"]}
            second_triplet = {x for x in second_triplet if x not in [".", "_"]}
            return [first_triplet, second_triplet]
        except ValueError:
            # No separator found - single triplet case
            # Get all non-padding tokens as one triplet
            triplet = set(x for x in sequence if x not in [".", "_", "/"])
            return [triplet]

    # Get triplets from both sequences
    pred_triplets = get_triplets(prediction)
    target_triplets = get_triplets(target)

    # If both have no triplets, they're equal
    if not pred_triplets and not target_triplets:
        return True

    # Check if same number of non-empty triplets
    if len(pred_triplets) != len(target_triplets):
        return False

    # For single triplet case, just compare the sets directly
    if len(pred_triplets) == 1:
        return pred_triplets[0] == target_triplets[0]

    # For two triplet case, try matching triplets in either order
    if len(pred_triplets) == 2:
        return (pred_triplets[0] == target_triplets[0] and pred_triplets[1] == target_triplets[1]) or \
               (pred_triplets[0] == target_triplets[1]
                and pred_triplets[1] == target_triplets[0])

    return False

def normalize_attention_impact(attention_weights, value_vectors):
    model_impacts = []
    all_impacts = []
    max_location = {'layer': -1, 'head': -1, 'alpha_idx': (-1,-1), 'value_norm': -1}
    max_impact_global = 0
    
    # First pass to find max and its location
    for layer in range(len(attention_weights)):
        alpha = attention_weights[layer]
        v = value_vectors[layer]
        
        for head in range(alpha.shape[1]):
            v_norms = torch.norm(v[0,head], dim=1)  # [49]
            att_values = alpha[0,head]  # [49,49]
            
            # For each position pair, compute impact
            for i in range(att_values.shape[0]):
                for j in range(att_values.shape[1]):
                    impact = float(att_values[i,j] * v_norms[j])
                    all_impacts.append(impact)
                    
                    if impact > max_impact_global:
                        max_impact_global = impact
                        max_location = {
                            'layer': layer,
                            'head': head,
                            'alpha_idx': (i,j),
                            'value_norm': float(v_norms[j])
                        }
    
    print(f"Maximum impact found in:")
    print(f"Layer: {max_location['layer']}")
    print(f"Head: {max_location['head']}")
    print(f"Attention position (i,j): {max_location['alpha_idx']}")
    print(f"Value vector norm at position j: {max_location['value_norm']}")
    print(f"Attention weight at this position: {float(attention_weights[max_location['layer']][0,max_location['head']][max_location['alpha_idx']])}")
    print(f"Maximum impact: {max_impact_global}")
    
    impact_95th = np.percentile(all_impacts, 99)
    print(f"95th percentile impact: {impact_95th}")

    # print("max_impact_global: ", max_impact_global)
    # print("impact_95th: ", impact_95th)

    # Normalize all impacts by global max
    for layer in range(len(attention_weights)):
        layer_impacts = []
        alpha = attention_weights[layer]
        v = value_vectors[layer]
        
        print("alpha shape: ", alpha.shape)
        for head in range(alpha.shape[1]):
            v_norms = torch.norm(v[0,head], dim=1)  # [49] 
            print("v_norms shape: ", v_norms.shape)
            print("v_norms: ", v_norms)
            weighted_impact = alpha[0,head] * v_norms  # [49,49]
            weighted_impact = weighted_impact.detach().cpu().numpy()
            print("weighted impact:", weighted_impact)
            # normalized_impact = (weighted_impact / max_impact_global).detach().cpu().numpy()
            # normalized_impact = (weighted_impact / impact_95th).detach().cpu().numpy()
            # normalized_impact = (weighted_impact / impact_95th).detach().cpu().numpy()
            # layer_impacts.append(normalized_impact)
            layer_impacts.append(weighted_impact)
            
        model_impacts.append(layer_impacts)
        
    return model_impacts

def attention_weights_from_sequence(
        config,
        input,
        tokenizer_path=f"all_cards_tokenizer.pkl",
        get_prediction=False,
        value_weighting=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # # Restore the model state dict
    # checkpoint = torch.load(os.path.join(
    #     PATH_PREFIX, config.filename), weights_only=False, map_location=torch.device('cpu'))

    checkpoint = torch.load(
        config.filename, weights_only=False, map_location=torch.device('cpu'))

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

        # no_set_token = tokenizer.token_to_id["*"]

        # is_correct = check_sequence_equality(
        #     prediciton = predictions[0],
        #     target = targets[0],
        #     no_set_token = no_set_token
        # )
        # # check_sequence_equality()
        # mask = targets != config.padding_token  # Create a mask to ignore padding
        # matches = ((predictions == targets) | ~mask).all(dim=1)
        # is_correct = bool(matches.sum().item())

        # # breakpoint()
        # print("predictions[0]: ", predictions[0])
        # print("targets[0]: ", targets[0])

        decoded_predictions = tokenizer.decode(predictions[0].tolist())
        decoded_targets = tokenizer.decode(targets[0].tolist())

        is_correct = check_sequence_equality(
            decoded_predictions,
            decoded_targets
        )
        # print("correct: ", is_correct)

        # print("full output: ", tokenizer.decode(outputs[0].tolist()))
        print("predictions: ", decoded_predictions)
        print("target: ", decoded_targets)

    _, _, attention_weights, _, value_vectors = model(
        sequences.to(device), False)
    # print("Value vectors: ", value_vectors)
    # print("len(value_vectors): ", len(value_vectors))
    # print("value vectors 0 shape: ", value_vectors[0].shape)

    # print("Attention weights: ", attention_weights)
    # print("len(attention_weights): ", len(attention_weights))
    # print("attention weights 0 shape: ", attention_weights[0].shape)

    if value_weighting:
        all_att_weights_np = normalize_attention_impact(attention_weights, value_vectors)
    else:
        all_att_weights_np = []
        for layer in range(config.n_layer):
            layer_weights = []  # Create a sublist for each layer
            for head in range(config.n_head):
                # Append the attention weights for the current head to the layer sublist
                layer_weights.append(
                    attention_weights[layer][0][head].detach().cpu().numpy())
            # Append the layer sublist to the main list
            all_att_weights_np.append(layer_weights)

    if get_prediction:
        return all_att_weights_np, is_correct, decoded_predictions, decoded_targets

    return all_att_weights_np

# Add new Python function to app.py:


def create_single_attention_weight_fig(attention_weights, labels, n_layers=4, n_heads=4, threshold=0.01, seed=0):
    height_per_layer = 400
    total_height = n_layers * height_per_layer

    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[
            f"L{l+1}H{h+1}" for l in range(n_layers) for h in range(n_heads)],
        specs=[[{"secondary_y": True}
                for _ in range(n_heads)] for _ in range(n_layers)],
        vertical_spacing=0.02,
        # horizontal_spacing=0.02
    )

    x_coords = [0, 1]
    y_min = -0.5
    y_max = len(labels) - 0.5

    reversed_labels = labels[::-1]
    custom_ticktext = [
        f'<span style="font-size:10px">{label}</span>' for label in reversed_labels]

    for layer in range(n_layers):
        for head in range(n_heads):
            weights = attention_weights[layer][head]
            n_points = len(labels)
            all_connections = []

            for i in range(n_points):
                for j in range(n_points):
                    weight = weights[i, j]
                    if weight > threshold:
                        all_connections.append({
                            'x': x_coords,
                            'y': [n_points - 1 - i, n_points - 1 - j],
                            'weight': weight,
                            'from_idx': i,
                            'to_idx': j
                        })

            for conn in all_connections:
                # Create more points along the line for better hover
                num_points = 20  # Number of points to interpolate
                x_interp = np.linspace(conn['x'][0], conn['x'][1], num_points)
                y_interp = np.linspace(conn['y'][0], conn['y'][1], num_points)

                fig.add_trace(
                    go.Scatter(
                        x=x_interp,
                        y=y_interp,
                        mode='lines+markers',  # Add markers for hover points
                        line=dict(
                            color='blue',
                            width=max(0.5, conn['weight'] * 3)
                        ),
                        marker=dict(
                            size=0.1,  # Very small markers
                            color='blue'
                        ),
                        opacity=min(conn['weight'],1),
                        showlegend=False,
                        hovertemplate=f"Weight: {conn['weight']:.3f}<br>From: {labels[conn['from_idx']]}<br>To: {labels[conn['to_idx']]}",
                        name="",
                        hoveron='points',
                        hoverinfo='all',
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

            fig.update_xaxes(
                ticktext=["Q", "K"],
                tickvals=[0, 1],
                row=layer+1,
                col=head+1,
                range=[-0.1, 1.1],
                showline=False,
                showgrid=False,
                tickfont=dict(size=4)
            )

            for secondary_y in [False, True]:
                fig.update_yaxes(
                    ticktext=custom_ticktext,
                    tickvals=list(range(len(labels))),
                    range=[y_min, y_max],
                    secondary_y=secondary_y,
                    row=layer+1,
                    col=head+1,
                    showline=False,
                    showgrid=False,
                    side='left' if not secondary_y else 'right',
                    tickfont=dict(size=2),
                )

    # Optimize layout
    fig.update_layout(
        height=total_height,
        width=n_heads*220,  # Reduced width
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),  # Optimized margins
        title_x=0.5,
        hovermode='closest',
    )

    return fig


def create_attention_weight_fig(attention_weights, labels1, labels2, n_layers=4, n_heads=4, threshold=0.01, seed=0):
    if seed != 0:
        random.seed(seed)
        pairs = [(sequence[i], sequence[i+1])
                 for i in range(0, len(sequence), 2)]
        random.shuffle(pairs)
        sequence = [item for pair in pairs for item in pair]

    height_per_layer = 400
    total_height = n_layers * height_per_layer

    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[
            f"L{l+1}H{h+1}" for l in range(n_layers) for h in range(n_heads)],
        specs=[[{"secondary_y": True}
                for _ in range(n_heads)] for _ in range(n_layers)],
        vertical_spacing=0.02,
    )

    x_coords = [0, 1]
    y_min = -0.5
    y_max = len(labels1) - 0.5

    reversed_labels = labels1[::-1]
    highlight_indices = {len(labels1) - i - 1: 'green' for i, (label1, label2)
                         in enumerate(zip(labels1, labels2)) if label1 != label2}

    custom_ticktext = []
    for i, label in enumerate(reversed_labels):
        if i in highlight_indices and i > 8:
            custom_ticktext.append(
                f'<span style="color: {highlight_indices[i]};font-size:10px">{label}</span>')
        else:
            custom_ticktext.append(
                f'<span style="font-size:10px">{label}</span>')

    for layer in range(n_layers):
        for head in range(n_heads):
            weights_diff = attention_weights[0][layer][head] - \
                attention_weights[1][layer][head]
            n_points = len(labels1)
            all_connections = []

            for i in range(n_points):
                for j in range(n_points):
                    weight = weights_diff[i, j]
                    if abs(weight) > threshold:
                        all_connections.append({
                            'x': x_coords,
                            'y': [n_points - 1 - i, n_points - 1 - j],
                            'weight': weight,
                            'from_idx': i,
                            'to_idx': j
                        })

            for conn in all_connections:
                # Create more points along the line for better hover
                num_points = 20  # Number of points to interpolate
                x_interp = np.linspace(conn['x'][0], conn['x'][1], num_points)
                y_interp = np.linspace(conn['y'][0], conn['y'][1], num_points)

                # Create smooth color gradient from red (-1) to white (0) to blue (1)
                # weight = max(-1, min(1, conn['weight']))  # Clamp weight between -1 and 1
                weight = conn["weight"]
                if weight < 0:
                    if weight < -1:
                        weight = -1
                    # Dark red (negative) to purple
                    r = 220
                    g = 20
                    b = int(20 + 235 * (1 + weight))
                else:
                    if weight > 1:
                        weight = 1
                    # Purple to light blue (positive)
                    r = int(220 - 180 * weight)
                    g = int(20 + 180 * weight)
                    b = 255
                color = f'rgb({r},{g},{b})'

                fig.add_trace(
                    go.Scatter(
                        x=x_interp,
                        y=y_interp,
                        mode='lines+markers',  # Add markers for hover points
                        line=dict(
                            color=color,
                            width=abs(conn['weight']) * 5
                        ),
                        marker=dict(
                            size=0.1,  # Very small markers
                            color=color
                        ),
                        showlegend=False,
                        hovertemplate=f"Weight: {conn['weight']:.3f}<br>From: {labels1[conn['from_idx']]}<br>To: {labels1[conn['to_idx']]}",
                        name="",
                        hoveron='points',
                        hoverinfo='all',
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

            fig.update_xaxes(
                ticktext=["Q", "K"],
                tickvals=[0, 1],
                row=layer+1,
                col=head+1,
                range=[-0.1, 1.1],
                showline=False,
                showgrid=False,
                tickfont=dict(size=4)
            )

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
                    tickfont=dict(size=2),
                )

    # Add colorscale
    colorscale_trace = go.Scatter(
        x=np.linspace(-1, 1, 100),
        y=[0]*100,
        mode='markers',
        marker=dict(
            size=10,
            color=np.linspace(-1, 1, 100),
            colorscale=[
                [0, 'rgb(40,200,255)'],      # Light blue for 1 (bottom)
                [0.5, 'rgb(140,100,255)'],   # Purple for 0 (middle)
                [1, 'rgb(220,20,20)']        # Dark red for -1 (top)
            ],
            showscale=True,
            colorbar=dict(
                title='Weight Difference',
                titleside='right',
                thickness=10,
                len=0.3,
                x=1.02,
                y=0.5,
                ticktext=['-1', '0', '1'],
                # Maps -1 to top (1.0), 0 to middle (0.5), 1 to bottom (0.0)
                tickvals=[1.0, 0, -1],
                tickmode='array'
            )
        ),
        showlegend=False
    )

    fig.add_trace(colorscale_trace)

    fig.update_layout(
        height=total_height,
        width=n_heads*220,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        title_x=0.5,
    )

    return fig


def shuffle_input_sequence(sequence, seed):
    if seed == 0:
        return sequence

    random.seed(seed)
    pairs = [(sequence[i], sequence[i+1])
             for i in range(0, len(sequence) - 9, 2)]
    random.shuffle(pairs)
    sequence_start = [item for pair in pairs for item in pair]
    sequence = sequence_start + sequence[-9:]
    return sequence


@app.route('/')
def index():
    # Get data for the heatmap
    labels1, labels2 = get_labels()
    fig = create_attention_weight_fig(
        attention_weights=get_attention_weights(),
        labels1=labels1,
        labels2=labels2,
        n_layers=config.n_layer,
        n_heads=config.n_head,
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

# Add new endpoint in app.py:


@app.route('/save_single_cards', methods=['POST'])
def save_single_cards():
    card_groups = request.json
    mapping = {"group1": []}

    for i, card in enumerate(card_groups.get('group1', [])):
        letter = map_card_to_letter(i)
        mapping['group1'].append({
            'letter': letter,
            'attributes': card
        })

    sequence1 = generate_input(card_groups, "group1")
    print("sequence1: ", sequence1)
    seed = card_groups.get('seed', 0)
    sequence1 = shuffle_input_sequence(sequence1, seed)
    value_weighting = card_groups.get("valueWeighting", False)

    global config
    model_num = card_groups.get('model_num', 1)
    if model_num == 100:
        config = GPTConfig44_SeededOrigDataset(seed=100)
    else:
        config = GPTConfig44_Complete()

    attention_weights1, is_correct1, decoded_predictions1, decoded_targets1 = attention_weights_from_sequence(
        config, sequence1, tokenizer_path="all_tokenizer.pkl", get_prediction=True, value_weighting=value_weighting)

    if is_correct1:
        sequence1 = sequence1[:-8] + decoded_predictions1

    fig = create_single_attention_weight_fig(
        attention_weights=attention_weights1,
        labels=sequence1,
        threshold=card_groups.get('threshold', 0.1),
        n_layers=config.n_layer,
        n_heads=config.n_head,
    )

    return jsonify({
        "status": "success",
        "cards": mapping,
        "prediction_results": {1: (sequence1, is_correct1, decoded_predictions1, decoded_targets1)},
        "plot_json": plotly.utils.PlotlyJSONEncoder().encode(fig)
    })


@app.route('/save_difference_cards', methods=['POST'])
def save_difference_cards():
    # Expecting data in the format {'group1': [...], 'group2': [...]}
    card_groups = request.json

    # Initialize the mappings for both groups
    mapping = {"group1": [], "group2": []}

    # Process Group 1
    for i, card in enumerate(card_groups.get('group1', [])):
        # Assuming letters are assigned sequentially
        letter = map_card_to_letter(i)
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
    # breakpoint()
    print("sequence1: ", sequence1)
    # print("sequence1: ", sequence1)
    seed = card_groups.get('seed', 0)
    value_weighting = card_groups.get("valueWeighting", False)

    sequence1 = shuffle_input_sequence(sequence1, seed)
    sequence2 = shuffle_input_sequence(sequence2, seed)

    global config
    model_num = card_groups.get('model_num', 1)
    if model_num == 100:
        config = GPTConfig44_SeededOrigDataset(seed=100)
    else:
        config = GPTConfig44_Complete()
  
    attention_weights1, is_correct1, decoded_predictions1, decoded_targets1 = attention_weights_from_sequence(
        config, sequence1, tokenizer_path="all_tokenizer.pkl", get_prediction=True, value_weighting=value_weighting)
    print("Got attention weights 1")

    # Update the labels to reflect the actual predicted sequence if correct, just different order
    if is_correct1:
        sequence1 = sequence1[:-8] + decoded_predictions1

    # attention_weights2, is_correct2, decoded_predictions2, decoded_targets2 = attention_weights_from_sequence(
    #     GPTConfig44_BalancedSets, sequence2, get_prediction=True)
    attention_weights2, is_correct2, decoded_predictions2, decoded_targets2 = attention_weights_from_sequence(
        config, sequence2, tokenizer_path="all_tokenizer.pkl", get_prediction=True, value_weighting=value_weighting)
    print("Got attention weights 2")

    if is_correct2:
        sequence2 = sequence2[:-8] + decoded_predictions2

    prediction_results = {
        1: (sequence1, is_correct1, decoded_predictions1, decoded_targets1),
        2: (sequence2, is_correct2, decoded_predictions2, decoded_targets2)
    }

    fig = create_attention_weight_fig(
        attention_weights=[attention_weights1, attention_weights2],
        labels1=sequence1,
        labels2=sequence2,
        threshold=card_groups.get('threshold', 0.1),
        n_layers=config.n_layer,
        n_heads=config.n_head,
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

@app.route('/change_config', methods=['POST'])
def change_config():
    global config
    layers = request.json.get('layers')
    if layers == 3:
        config = GPTConfig34_Complete()
    else:
        config = GPTConfig44_Complete()
    return jsonify({"status": "success"})


# @app.route('/change_model_config', methods=['POST'])
# def change_model_config():
#     global config
#     model_num = request.json.get('model_num', 1)
#     if model_num == 1:
#         config = GPTConfig44_Complete()
#     elif model_num == 100:
#         config = GPTConfig44_SeededOrigDataset(seed=100)
#     return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
