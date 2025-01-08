from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = Flask(__name__)

# Card attributes
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

# Store the saved cards (in a real application, you'd want to use a database)
saved_card_mappings = []

def map_card_to_letter(card_num):
    return chr(65 + card_num)  # 0->A, 1->B, 2->C, etc.

def get_attention_weights():
    return [np.random.rand(10, 10) for _ in range(2)]

def get_labels():
    return ['Label' + str(i) for i in range(10)], ['Label' + str(i) for i in range(10, 20)]

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
    cards = request.json
    
    # Create mapping for each card
    mapping = []
    for i, card in enumerate(cards):
        letter = map_card_to_letter(i)
        card_info = {
            'letter': letter,
            'attributes': card
        }
        mapping.append(card_info)
    
    # Store the mapping (in a real application, you'd want to use a database)
    global saved_card_mappings
    saved_card_mappings = mapping
    
    return jsonify({
        "status": "success", 
        "cards": mapping
    })

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import plotly
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# app = Flask(__name__)

# # Card attributes
# shapes = ["oval", "squiggle", "diamond"]
# colors = ["green", "blue", "pink"]
# numbers = ["one", "two", "three"]
# shadings = ["solid", "striped", "open"]

# def get_attention_weights():
#     # Generate random attention weights for demonstration
#     return [np.random.rand(10, 10) for _ in range(2)]

# def get_labels():
#     # Generate random labels for demonstration
#     return ['Label' + str(i) for i in range(10)], ['Label' + str(i) for i in range(10, 20)]

# @app.route('/')
# def index():
#     # Get data for the heatmap
#     attention_weights = get_attention_weights()
#     labels1, labels2 = get_labels()
#     n_layers, n_heads = 4, 4
    
#     fig = make_subplots(rows=n_layers, cols=n_heads, 
#                         subplot_titles=[f"Layer {l+1} Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
#                         specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)])
    
#     for layer in range(n_layers):
#         for head in range(n_heads):
#             attention_weights_diff = attention_weights[0] - attention_weights[1]
            
#             for i in range(len(labels1)):
#                 for j in range(len(labels1)):
#                     weight = attention_weights_diff[i, j]
#                     fig.add_trace(
#                         go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
#                                    opacity=abs(weight), showlegend=False, hoverinfo='x+y'),
#                         row=layer+1, col=head+1,
#                         secondary_y=False,
#                     )
#                     fig.add_trace(
#                         go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
#                                    opacity=0, showlegend=False, hoverinfo='x+y'),
#                         row=layer+1, col=head+1,
#                         secondary_y=True,
#                     )
            
#             fig.update_xaxes(ticktext=["Query", "Key"], tickvals=[0, 1], row=layer+1, col=head+1)
            
#             fig.update_yaxes(
#                 ticktext=labels1, 
#                 tickvals=list(range(len(labels1))), 
#                 secondary_y=False,
#                 row=layer+1, 
#                 col=head+1
#             )
            
#             fig.update_yaxes(
#                 ticktext=labels2,
#                 tickvals=list(range(len(labels2))),
#                 secondary_y=True,
#                 row=layer+1, 
#                 col=head+1,
#             )

#     fig.update_layout(
#         height=n_layers*400, 
#         width=n_heads*300, 
#         title_text="Attention Line Pattern Differences",
#     )
    
#     plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
#     return render_template('index.html', 
#                          shapes=shapes,
#                          colors=colors,
#                          numbers=numbers,
#                          shadings=shadings,
#                          plot_json=plot_json)

# @app.route('/save_cards', methods=['POST'])
# def save_cards():
#     cards = request.json
#     # Here you can process the cards data as needed
#     return jsonify({"status": "success", "cards": cards})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.utils

# app = Flask(__name__)

# def get_attention_weights():
#     # Generate random attention weights for demonstration
#     return [np.random.rand(10, 10) for _ in range(2)]

# def get_labels():
#     # Generate random labels for demonstration
#     return ['Label' + str(i) for i in range(10)], ['Label' + str(i) for i in range(10, 20)]

# @app.route('/')
# def lineplot():
#     attention_weights = get_attention_weights()
#     labels1, labels2 = get_labels()
#     n_layers, n_heads = 4, 4
    
#     fig = make_subplots(rows=n_layers, cols=n_heads, 
#                         subplot_titles=[f"Layer {l+1} Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
#                         specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)])
    
#     for layer in range(n_layers):
#         for head in range(n_heads):
#             attention_weights_diff = attention_weights[0] - attention_weights[1]
            
#             for i in range(len(labels1)):
#                 for j in range(len(labels1)):
#                     weight = attention_weights_diff[i, j]
#                     fig.add_trace(
#                         go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
#                                    opacity=abs(weight), showlegend=False, hoverinfo='x+y'),
#                         row=layer+1, col=head+1,
#                         secondary_y=False,
#                     )
#                     fig.add_trace(
#                         go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
#                                    opacity=0, showlegend=False, hoverinfo='x+y'),
#                         row=layer+1, col=head+1,
#                         secondary_y=True,
#                     )
            
#             fig.update_xaxes(ticktext=["Query", "Key"], tickvals=[0, 1], row=layer+1, col=head+1)
            
#             # Update left y-axis
#             fig.update_yaxes(
#                 ticktext=labels1, 
#                 tickvals=list(range(len(labels1))), 
#                 secondary_y=False,
#                 row=layer+1, 
#                 col=head+1
#             )
            
#             # Add right y-axis
#             fig.update_yaxes(
#                 ticktext=labels2,
#                 tickvals=list(range(len(labels2))),
#                 secondary_y=True,
#                 row=layer+1, 
#                 col=head+1,
#             )

#     fig.update_layout(
#         height=n_layers*400, 
#         width=n_heads*300, 
#         title_text="Attention Line Pattern Differences",
#         # yaxis=dict(side='left'),
#         # yaxis2=dict(side='right', overlaying='y', showgrid=False)
#     )
    
#     plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
#     return render_template('lineplot.html', plot_json=plot_json)


# if __name__ == '__main__':
#     app.run(debug=True)


# OLD


# from flask import Flask, render_template
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.utils

# app = Flask(__name__)

# def get_attention_weights():
#     # Generate random attention weights for demonstration
#     return [np.random.rand(10, 10) for _ in range(2)]

# def get_labels():
#     # Generate random labels for demonstration
#     return ['Label' + str(i) for i in range(10)], ['Label' + str(i) for i in range(10, 20)]

# @app.route('/')
# def lineplot():
#     attention_weights = get_attention_weights()
#     labels1, labels2 = get_labels()
#     n_layers, n_heads = 4, 4
    
#     fig = make_subplots(rows=n_layers, cols=n_heads, 
#                         subplot_titles=[f"Layer {l+1} Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
#                         specs=[[{"secondary_y": True} for _ in range(n_heads)] for _ in range(n_layers)])
    
#     for layer in range(n_layers):
#         for head in range(n_heads):
#             attention_weights_diff = attention_weights[0] - attention_weights[1]
            
#             for i in range(len(labels1)):
#                 for j in range(len(labels1)):
#                     weight = attention_weights_diff[i, j]
#                     fig.add_trace(
#                         go.Scatter(x=[0, 1], y=[i, j], mode='lines', line=dict(color='blue', width=1), 
#                                    opacity=abs(weight), showlegend=False),
#                         row=layer+1, col=head+1
#                     )
            
#             fig.update_xaxes(ticktext=["Query", "Key"], tickvals=[0, 1], row=layer+1, col=head+1)
#             # fig.update_yaxes(ticktext=labels1, tickvals=list(range(len(labels1))), showticklabels=True, row=layer+1, col=head+1)
#             # Adding the same y-axis labels on the right side
#             fig.update_yaxes(
#                 ticktext=labels1, 
#                 tickvals=list(range(len(labels1))), 
#                 side='left',  # Set the y-axis to the left side
#                 showticklabels=True,  # Ensure the labels are visible
#                 row=layer+1, 
#                 col=head+1
#             )

#             fig.update_yaxes(
#                 ticktext=labels1, 
#                 tickvals=list(range(len(labels1))), 
#                 side='right',  # Set the y-axis to the right side
#                 showticklabels=True,  # Ensure the labels are visible
#                 overlaying='y',
#                 row=layer+1, 
#                 col=head+1
#             )

            
#             # fig.update_yaxes(
#             #     ticktext=labels1,
#             #     tickvals=list(range(len(labels1))),
#             #     side='left',
#             #     secondary_y=False,
#             #     showticklabels=True,
#             #     row=layer+1,
#             #     col=head+1
#             # )
#             # fig.update_yaxes(
#             #     ticktext=labels2,
#             #     tickvals=list(range(len(labels2))),
#             #     side='right',
#             #     # secondary_y=True,
#             #     # showticklabels=True,
#             #     row=layer+1,
#             #     col=head+1
#             # )
    
#     # fig.update_layout(
#     #     yaxis_showticklabels=True,
#     #     yaxis2_showticklabels=True
#     # )

#     fig.update_layout(height=n_layers*400, width=n_heads*300, 
#                       title_text="Attention Line Pattern Differences",
#                       margin=dict(t=50, b=50, l=50, r=150))  # Increased right margin (r=150))
    
#     plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
#     return render_template('lineplot.html', plot_json=plot_json)

# if __name__ == '__main__':
#     app.run(debug=True)


# # from flask import Flask, render_template
# # import numpy as np
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots
# # import plotly.utils

# # app = Flask(__name__)

# # def get_attention_weights():
# #     # Generate 16 random 10x10 matrices
# #     return [np.random.rand(10, 10) for _ in range(16)]

# # @app.route('/')
# # def heatmap():
# #     attention_weights = get_attention_weights()
    
# #     fig = make_subplots(rows=4, cols=4, subplot_titles=[f"Layer {i+1}" for i in range(16)])
    
# #     for i, weights in enumerate(attention_weights):
# #         row = i // 4 + 1
# #         col = i % 4 + 1
# #         fig.add_trace(
# #             go.Heatmap(z=weights, colorscale='Viridis', showscale=False),
# #             row=row, col=col
# #         )

# #     fig.update_layout(
# #         title_text="4x4 Grid of Attention Weights Heatmaps",
# #         height=1000,
# #         width=1000
# #     )
    
# #     plot_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
    
# #     return render_template('heatmap.html', plot_json=plot_json)

# # if __name__ == '__main__':
# #     app.run(debug=True)
