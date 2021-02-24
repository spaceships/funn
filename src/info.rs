use crate::neural_net::NeuralNet;
use colored::*;
use fancy_garbling::depth_informer::DepthInformer;
use fancy_garbling::{BinaryGadgets, FancyInput};
use itertools::Itertools;

/// Get info for neural network computation
pub fn bool_info(nn: &NeuralNet, bitwidth: usize, _field_size: Option<usize>) {
    println!("{}", "* Getting circuit info".green());

    ////////////////////////////////////////////////////////////////////////////////
    // run the neural network with Informer
    let mut pb = pbr::ProgressBar::new(nn.nlayers() as u64);
    let mut informer = DepthInformer::new();

    let inps = (0..nn.num_inputs())
        .map(|_| informer.bin_encode(0, bitwidth).unwrap())
        .collect_vec();

    let outs = nn.eval_boolean_no_secret_weights(&mut informer, &inps, bitwidth, Some(&mut pb));

    informer.bin_outputs(&outs).unwrap();

    pb.finish();
    println!("{}", informer);
}
