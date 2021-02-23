use crate::neural_net::NeuralNet;
use fancy_garbling::circuit::CircuitBuilder;

pub fn print_boolean_circuit(nn: &NeuralNet, bitwidth: &[usize]) {
    let mut builder = CircuitBuilder::new();
    let mut inputs = Vec::with_capacity(nn.num_inputs());
    for _ in 0..nn.num_inputs() {
        let inp = builder.bin_evaluator_input(bitwidth[0]);
        inputs.push(inp);
    }
    let mut pb = pbr::ProgressBar::new(nn.nlayers() as u64);
    nn.eval_boolean_no_secret_weights(&mut builder, &inputs, bitwidth, Some(&mut pb));
}
