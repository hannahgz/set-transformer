import { lineplotSpecific, generateLineplot, lineplotDifferenceInputs, interactiveLineplotDifferenceInputs } from './graph.js';
import { GPTConfig44_BalancedSets } from './model.js';
import { prettyPrintInput } from './data_utils.js';

const PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp';

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("inputForm");
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const input1 = document.getElementById("input1").value.split(",");
        const input2 = document.getElementById("input2").value.split(",");
        lineplotDifferenceInputs(
            GPTConfig44_BalancedSets,
            input1,
            input2,
            `${PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl`,
            true,
            "input1_input2_allsame"
        );
    });
});
