import * as d3 from 'd3';
import { prettyPrintInput } from './data_utils.js';
import { GPT } from './model.js';
import { loadTokenizer } from './tokenizer.js';

const PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp';

export function plotAttentionHeatmap(attWeights, labels, title = "Attention Pattern", savefig = null) {
    const attWeightsArray = attWeights; // Assuming attWeights is already a JavaScript array

    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const width = 800 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    const svg = d3.select("body").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
        .range([0, width])
        .domain(labels)
        .padding(0.01);

    const y = d3.scaleBand()
        .range([height, 0])
        .domain(labels)
        .padding(0.01);

    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end");

    svg.append("g")
        .call(d3.axisLeft(y));

    const colorScale = d3.scaleSequential(d3.interpolateReds)
        .domain([0, d3.max(attWeightsArray, d => d3.max(d))]);

    svg.selectAll()
        .data(attWeightsArray)
        .enter()
        .append("rect")
        .attr("x", (d, i) => x(labels[i % labels.length]))
        .attr("y", (d, i) => y(labels[Math.floor(i / labels.length)]))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", d => colorScale(d));

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", -margin.top / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text(title);

    if (savefig) {
        // Code to save the figure
    }
}

export function plotAttentionHeadsLayerHorizontal(attentionWeights, labels, layer, nHeads, titlePrefix = "Attention Pattern", savefig = null) {
    // ...existing code...
}

export function plotAttentionPatternAll(attentionWeights, labels, nLayers, nHeads, titlePrefix = "Attention Pattern", savefig = null) {
    // ...existing code...
}

export function plotAttentionPatternLines(attentionWeights, labels, nLayers, nHeads, titlePrefix = "Attention Line Pattern", savefig = null, threshold = null) {
    // ...existing code...
}

export function generateLineplot(config, datasetIndices, datasetPath, tokenizerPath, useLabels = false, threshold = 0.05, getPrediction = false) {
    // ...existing code...
}

export function makePredictionGivenInput(config, input, tokenizer, model, getPrediction = false) {
    // ...existing code...
}

export function lineplotSpecific(config, input, tokenizerPath = `${PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl`, threshold = 0.1, getPrediction = false, filenamePrefix = "") {
    // ...existing code...
}

export function plotAttentionPatternLinesComparison(attentionWeights1, attentionWeights2, labels1, labels2, nLayers = 4, nHeads = 4, titlePrefix = "Attention Line Pattern Differences", savefig = null, threshold = null) {
    // ...existing code...
}

export function lineplotDifferenceInputs(config, input1, input2, tokenizerPath = `${PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl`, threshold = 0.1, getPrediction = false, filenamePrefix = "") {
    // ...existing code...
}

export function interactiveLineplotDifferenceInputs(config, tokenizerPath = `${PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl`, threshold = 0.1, getPrediction = false, filenamePrefix = "") {
    // ...existing code...
}
