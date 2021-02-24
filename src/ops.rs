use crate::util;
use fancy_garbling::util as numbers;
use fancy_garbling::{
    BinaryBundle, BinaryGadgets, Bundle, BundleGadgets, CrtBundle, CrtGadgets, Fancy, FancyInput,
    HasModulus,
};
use itertools::Itertools;

/// The accuracy of each kind of activation function.
#[derive(Clone, Debug)]
pub struct Accuracy {
    pub(crate) relu: String,
    pub(crate) sign: String,
    pub(crate) max: String,
}

/// NeuralNetOps encodes the particular way that we evaluate a neural net - whether it is
/// directly over `i64` or as an arithmetic circuit, or whatever. The first argument to
/// these functions could be a `Fancy` object.
pub struct NeuralNetOps<B, T> {
    // Encode a constant.
    pub enc: Box<dyn Fn(&mut B, i64) -> T>,
    // Encode a secret.
    pub sec: Box<dyn Fn(&mut B, Option<i64>) -> T>,
    // Add two values.
    pub add: Box<dyn Fn(&mut B, &T, &T) -> T>,
    // Scalar multiplication.
    pub cmul: Box<dyn Fn(&mut B, &T, i64) -> T>,
    // Apply secret weight to an input.
    pub proj: Box<dyn Fn(&mut B, &T, Option<i64>) -> T>,
    // Maximum of a slice of encodings.
    pub max: Box<dyn Fn(&mut B, &[T]) -> T>,
    // Activation function chosen based on string name.
    pub act: Box<dyn Fn(&mut B, &str, &T) -> T>,
    pub zero: Box<dyn Fn(&mut B) -> T>,
}

impl NeuralNetOps<usize, i64> {
    pub fn plaintext() -> NeuralNetOps<usize, i64> {
        NeuralNetOps {
            enc: Box::new(|_, x| x),
            sec: Box::new(|_, _| panic!("secret not supported for plaintext eval")),
            add: Box::new(|_, x, y| x + y),
            cmul: Box::new(|_, x, y| x * y),
            proj: Box::new(|_, _, _| panic!("secret not supported for plaintext eval")),
            max: Box::new(|_, xs| *xs.iter().max().unwrap()),
            act: Box::new(|_, a, x| match a {
                "sign" => {
                    if *x >= 0 {
                        1
                    } else {
                        -1
                    }
                }
                "relu" => std::cmp::max(*x, 0),
                "id" => *x,
                act => panic!("unsupported activation {}", act),
            }),
            zero: Box::new(|_| 0),
        }
    }
}

impl<F: Fancy<Item = W> + FancyInput<Item = W>, W: Clone + HasModulus> NeuralNetOps<F, W> {
    pub fn arithmetic(
        q: u128, // input modulus
        output_mod: u128,
        secret_weights_owned: bool,
        accuracy: &Accuracy,
    ) -> NeuralNetOps<F, CrtBundle<W>> {
        let relu_accuracy = accuracy.relu.clone();
        let sign_accuracy = accuracy.sign.clone();
        let max_accuracy = accuracy.max.clone();
        let output_ps = numbers::factor(output_mod);
        NeuralNetOps {
            enc: Box::new(move |b: &mut F, x| {
                b.crt_constant_bundle(util::to_mod_q(x, q), q).unwrap()
            }),

            sec: if secret_weights_owned {
                Box::new(move |b: &mut F, opt_x| {
                    b.crt_encode(util::to_mod_q(opt_x.unwrap(), q), q)
                        .ok()
                        .expect("error encoding secret CRT value")
                })
            } else {
                Box::new(move |b: &mut F, _| {
                    b.crt_receive(q)
                        .ok()
                        .expect("error receiving secret CRT value")
                })
            },

            add: Box::new(move |b: &mut F, x: &CrtBundle<W>, y: &CrtBundle<W>| {
                b.crt_add(x, y).unwrap()
            }),

            cmul: Box::new(move |b: &mut F, x: &CrtBundle<W>, y| {
                b.crt_cmul(x, util::to_mod_q(y, q)).unwrap()
            }),

            proj: Box::new(move |b: &mut F, inp, opt_w| {
                if let Some(w) = opt_w {
                    // convert the weight to crt mod q
                    let ws = util::to_mod_q_crt(w, q);
                    CrtBundle::new(
                        inp.wires()
                            .iter()
                            .zip(ws.iter())
                            .map(|(wire, weight)| {
                                let q = wire.modulus();
                                let tab = (0..q).map(|x| x * weight % q).collect_vec();
                                // project each input x to x*w
                                b.proj(wire, q, Some(tab)).unwrap()
                            })
                            .collect_vec(),
                    )
                } else {
                    CrtBundle::new(
                        inp.wires()
                            .iter()
                            .map(|wire| {
                                // project the input, without knowing the weight
                                b.proj(wire, wire.modulus(), None).unwrap()
                            })
                            .collect_vec(),
                    )
                }
            }),
            max: Box::new(move |b: &mut F, xs: &[CrtBundle<W>]| {
                b.crt_max(xs, &max_accuracy).unwrap()
            }),
            act: Box::new(move |b: &mut F, a: &str, x: &CrtBundle<W>| match a {
                "sign" => b.crt_sgn(x, &sign_accuracy, Some(&output_ps)).unwrap(),
                "relu" => b.crt_relu(x, &relu_accuracy, Some(&output_ps)).unwrap(),
                "id" => x.clone(),
                act => panic!("unsupported activation {}", act),
            }),
            zero: Box::new(move |b: &mut F| b.crt_constant_bundle(0, q).unwrap()),
        }
    }

    pub fn binary(nbits: usize, secret_weights_owned: bool) -> NeuralNetOps<F, BinaryBundle<W>> {
        NeuralNetOps {
            enc: Box::new(move |b: &mut F, x| {
                let twos = util::i64_to_twos_complement(x, nbits);
                b.bin_constant_bundle(twos, nbits).unwrap()
            }),

            sec: Box::new(move |b: &mut F, opt_x| {
                if secret_weights_owned {
                    let xbits = util::i64_to_twos_complement(opt_x.unwrap(), nbits);
                    b.bin_encode(xbits, nbits)
                        .ok()
                        .expect("error encoding binary secret value")
                } else {
                    b.bin_receive(nbits)
                        .ok()
                        .expect("error receiving binary secret value")
                }
            }),

            add: Box::new(move |b: &mut F, x: &BinaryBundle<W>, y: &BinaryBundle<W>| {
                b.bin_addition_no_carry(x, y).unwrap()
            }),

            cmul: Box::new(move |b: &mut F, x: &BinaryBundle<W>, y| {
                b.bin_cmul(x, util::i64_to_twos_complement(y, nbits), nbits)
                    .unwrap()
            }),

            proj: Box::new(move |b: &mut F, inp, opt_w| {
                // ignore the input weight - it needs to be a garbler input
                let weight_bits = opt_w.map(|w| util::i64_to_twos_complement(w, nbits));
                let w = if secret_weights_owned {
                    b.bin_encode(weight_bits.unwrap(), nbits)
                        .ok()
                        .expect("could not encode binary secret")
                } else {
                    b.bin_receive(nbits)
                        .ok()
                        .expect("could not receive binary secret")
                };
                b.bin_multiplication_lower_half(inp, &w).unwrap()
            }),

            max: Box::new(move |b: &mut F, xs: &[BinaryBundle<W>]| b.bin_max(xs).unwrap()),

            act: Box::new(move |b: &mut F, a: &str, x: &BinaryBundle<W>| match a {
                "sign" => {
                    let sign = x.wires().last().unwrap();
                    let neg1 = (1 << nbits) - 1;
                    b.bin_multiplex_constant_bits(sign, 1, neg1, nbits).unwrap()
                }
                "relu" => {
                    let sign = x.wires().last().unwrap();
                    let zeros = b.bin_constant_bundle(0u128, nbits).unwrap();
                    BinaryBundle::from(b.multiplex(&sign, &x, &zeros).unwrap())
                }
                "id" => x.clone(),
                act => panic!("unsupported activation {}", act),
            }),

            zero: Box::new(move |b: &mut F| b.bin_constant_bundle(0u128, nbits).unwrap()),
        }
    }
}

impl<F: Fancy<Item = W>, W: Clone + HasModulus> NeuralNetOps<F, W> {
    pub fn binary_no_secret_weights(nbits: usize) -> NeuralNetOps<F, BinaryBundle<W>> {
        NeuralNetOps {
            enc: Box::new(move |b: &mut F, x| {
                let twos = util::i64_to_twos_complement(x, nbits);
                b.bin_constant_bundle(twos, nbits).unwrap()
            }),
            sec: Box::new(move |_b: &mut F, _opt_x| panic!("no sec")),
            add: Box::new(move |b: &mut F, x: &BinaryBundle<W>, y: &BinaryBundle<W>| {
                b.bin_addition_no_carry(x, y).unwrap()
            }),
            cmul: Box::new(move |b: &mut F, x: &BinaryBundle<W>, y| {
                b.bin_cmul(x, util::i64_to_twos_complement(y, nbits), nbits)
                    .unwrap()
            }),
            proj: Box::new(move |_b: &mut F, _inp, _opt_w| panic!("no sec")),
            max: Box::new(move |b: &mut F, xs: &[BinaryBundle<W>]| b.bin_max(xs).unwrap()),
            act: Box::new(move |b: &mut F, a: &str, x: &BinaryBundle<W>| match a {
                "sign" => {
                    let sign = x.wires().last().unwrap();
                    let neg1 = (1 << nbits) - 1;
                    b.bin_multiplex_constant_bits(sign, 1, neg1, nbits).unwrap()
                }
                "relu" => {
                    let sign = x.wires().last().unwrap();
                    let zeros = b.bin_constant_bundle(0u128, nbits).unwrap();
                    BinaryBundle::from(b.multiplex(&sign, &x, &zeros).unwrap())
                }
                "id" => x.clone(),
                act => panic!("unsupported activation {}", act),
            }),
            zero: Box::new(move |b: &mut F| b.bin_constant_bundle(0u128, nbits).unwrap()),
        }
    }

    #[allow(unused_variables)]
    pub fn finite_field(field_size: usize, n_field_elems: usize) -> NeuralNetOps<F, Bundle<W>> {
        NeuralNetOps {
            sec: Box::new(move |_b: &mut F, _opt_x| panic!("no sec")),
            proj: Box::new(move |_b: &mut F, _inp, _opt_w| panic!("no sec")),

            enc: Box::new(move |b: &mut F, x| {
                let twos = util::i64_to_twos_complement(x, field_size * n_field_elems);
                let digits = util::u128_to_field(twos, field_size, n_field_elems);
                b.constant_bundle(&digits, &vec![1 << field_size; n_field_elems])
                    .unwrap()
            }),

            add: Box::new(move |b: &mut F, x: &Bundle<W>, y: &Bundle<W>| {
                b.mixed_radix_addition(&[x.clone(), y.clone()]).unwrap()
            }),

            cmul: Box::new(move |b: &mut F, x: &Bundle<W>, y| {
                unimplemented!()
                // b.bin_cmul(x, util::i64_to_twos_complement(y, nbits), nbits)
                // .unwrap()
            }),

            max: Box::new(move |b: &mut F, xs: &[Bundle<W>]| {
                // b.bin_max(xs).unwrap()
                unimplemented!()
            }),

            act: Box::new(move |b: &mut F, a: &str, x: &Bundle<W>| match a {
                "sign" => {
                    // let sign = x.wires().last().unwrap();
                    // let neg1 = (1 << nbits) - 1;
                    // b.bin_multiplex_constant_bits(sign, 1, neg1, nbits).unwrap()
                    unimplemented!()
                }
                "relu" => {
                    // let sign = x.wires().last().unwrap();
                    // let zeros = b.bin_constant_bundle(0u128, nbits).unwrap();
                    // Bundle::from(b.multiplex(&sign, &x, &zeros).unwrap())
                    unimplemented!()
                }
                "id" => x.clone(),
                act => panic!("unsupported activation {}", act),
            }),

            zero: Box::new(move |b: &mut F| {
                b.constant_bundle(
                    &vec![0; n_field_elems],
                    &vec![1 << field_size; n_field_elems],
                )
                .unwrap()
            }),
        }
    }
}
