//! The lowest level of the Neural Network abstraction is a Layer. We describe how to
//! evaluate a NN Layer polymorphically by encoding basic operations - adding, encoding,
//! max, etc - in a struct, which could be Fancy - Arithemtic or Boolean - or plaintext in
//! the clear. We even use the same struct to evaluate the maximum bitwith. The upshot is
//! that we only have to say how to evaluate each kind of layer once (in `Layer::eval`),
//! minimizing NN evaluation bugs.

use crate::ops::{Accuracy, NeuralNetOps, ComputationInfo};
use fancy_garbling::{BinaryBundle, Bundle, CrtBundle, Fancy, FancyInput, HasModulus};
use itertools::{iproduct, Itertools};
use ndarray::Array3;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;
use std::cmp::max;

/// Each layer optionally contains weights and biases. If they are not present, the
/// weights and biases will be treated as secret (garbler inputs).
#[derive(Clone)]
pub enum Layer {
    Dense {
        weights: Vec<Array3<Option<i64>>>,
        biases: Vec<Option<i64>>,
        activation: String,
    },

    Convolutional {
        filters: Vec<Array3<Option<i64>>>,
        biases: Vec<Option<i64>>,
        input_shape: (usize, usize, usize),
        kernel_shape: (usize, usize, usize),
        stride: (usize, usize),
        activation: String,
        pad: bool,
    },

    MaxPooling2D {
        input_shape: (usize, usize, usize),
        stride: (usize, usize),
        size: (usize, usize),
        pad: bool,
    },

    Flatten {
        input_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
    },

    Activation {
        // Layer just does activations, one per input
        activation: String,
        input_shape: (usize, usize, usize),
    },
}

impl Layer {
    /// Get the name of the type of this Layer.
    pub fn name(&self) -> &str {
        match self {
            Layer::Dense { .. } => "Dense",
            Layer::Convolutional { .. } => "Convolutional",
            Layer::MaxPooling2D { .. } => "MaxPooling2D",
            Layer::Flatten { .. } => "Flatten",
            Layer::Activation { .. } => "Activation",
        }
    }

    /// Output info about this layer.
    pub fn info(&self) -> String {
        match self {
            Layer::Dense { activation, .. } => {
                let (x, _, _) = self.output_dims();
                format!("Dense[{}] activation={}", x, activation)
            }

            Layer::Convolutional {
                kernel_shape,
                stride,
                filters,
                activation,
                ..
            } => format!(
                "Conv[{}] activation={} stride={:?} kernel_shape={:?}",
                filters.len(),
                activation,
                stride,
                kernel_shape
            ),

            Layer::MaxPooling2D { stride, size, .. } => {
                format!("MaxPooling2D stride={:?} size={:?}", stride, size)
            }

            Layer::Flatten { .. } => format!("Flatten"),

            Layer::Activation { activation, .. } => format!("Activation {}", activation),
        }
    }

    /// Returns (height, width, depth).
    pub fn input_dims(&self) -> (usize, usize, usize) {
        match self {
            Layer::Dense { weights, .. } => weights.iter().next().map_or((0, 0, 0), |w0| w0.dim()),
            Layer::Convolutional { input_shape, .. } => *input_shape,
            Layer::MaxPooling2D { input_shape, .. } => *input_shape,
            Layer::Flatten { input_shape, .. } => *input_shape,
            Layer::Activation { input_shape, .. } => *input_shape,
        }
    }

    /// Get the number of items in the input.
    pub fn input_size(&self) -> usize {
        let (x, y, z) = self.input_dims();
        x * y * z
    }

    /// Get the dimensions of the output in (height, width, depth).
    pub fn output_dims(&self) -> (usize, usize, usize) {
        match self {
            Layer::Dense { biases, .. } => (biases.len(), 1, 1),

            Layer::Convolutional {
                input_shape,
                kernel_shape,
                stride,
                filters,
                pad,
                ..
            } => {
                let (height, width, _) = input_shape;
                let (ker_height, ker_width, _) = kernel_shape;
                let (stride_y, stride_x) = stride;

                if *pad {
                    (*height, *width, filters.len())
                } else {
                    (
                        (height - ker_height) / stride_y + 1,
                        (width - ker_width) / stride_x + 1,
                        filters.len(),
                    )
                }
            }

            Layer::MaxPooling2D {
                input_shape,
                stride,
                size,
                pad,
            } => {
                let (height, width, depth) = input_shape;
                let (pool_height, pool_width) = size;
                let (stride_y, stride_x) = stride;

                if *pad {
                    *input_shape
                } else {
                    (
                        (height - pool_height) / stride_y + 1,
                        (width - pool_width) / stride_x + 1,
                        *depth,
                    )
                }
            }

            Layer::Flatten { output_shape, .. } => *output_shape,

            Layer::Activation { input_shape, .. } => *input_shape,
        }
    }

    /// Get the number of items in the output.
    pub fn output_size(&self) -> usize {
        let (x, y, z) = self.output_dims();
        x * y * z
    }

    pub fn computation_info(&self, input: &Array3<usize>, info: Rc<RefCell<ComputationInfo>>)
        -> Array3<usize>
    {
        let _info = info.clone();
        let enc = move |_: &mut usize, _x: i64| {
            _info.borrow_mut().nenc += 1;
            0
        };

        let _info = info.clone();
        let sec = move |_: &mut usize, _x: Option<i64>| {
            _info.borrow_mut().nsec += 1;
            0
        };

        let _info = info.clone();
        let proj = move |_: &mut usize, inp: &usize, _opt_w: Option<i64>| {
            _info.borrow_mut().nproj += 1;
            inp + 1
        };

        let _info = info.clone();
        let add = move |_: &mut usize, x: &usize, y: &usize| {
            _info.borrow_mut().nadd += 1;
            max(*x, *y)
        };

        let _info = info.clone();
        let cmul = move |_: &mut usize, x: &usize, _y: i64| {
            _info.borrow_mut().ncmul += 1;
            x + 1
        };

        let _info = info.clone();
        let max = move |_: &mut usize, xs: &[usize]| {
            _info.borrow_mut().nmax += 1;
            xs.iter().max().unwrap() + 1
        };

        let _info = info.clone();
        let act = move |_: &mut usize, a: &str, x: &usize| {
            if _info.borrow().act_calls.contains_key(a) {
                *_info.borrow_mut().act_calls.get_mut(a).unwrap() += 1
            } else {
                _info.borrow_mut().act_calls.insert(a.to_string(), 1);
            }
            x + 1
        };

        let ops = NeuralNetOps {
            enc: Box::new(enc),
            sec: Box::new(sec),
            add: Box::new(add),
            cmul: Box::new(cmul),
            proj: Box::new(proj),
            max: Box::new(max),
            act: Box::new(act),
            zero: Box::new(|_| 0),
        };

        let layer_output = self.eval(&mut 0, input, &ops, false);
        layer_output
    }

    /// Evaluate this layer in plaintext while finding the max value on a wire.
    pub fn max_bitwidth(&self, input: &Array3<i64>, _: usize) -> (Array3<i64>, i64) {
        let max_atomic: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let thread_atomic = max_atomic.clone();
        let store_max_base = Arc::new(move |x: i64| {
            thread_atomic.fetch_max(x.abs() as usize, Ordering::SeqCst);
        });

        let store_max = store_max_base.clone();
        let enc = move |_: &mut usize, x: i64| {
            store_max(x);
            x
        };

        let store_max = store_max_base.clone();
        let proj = move |_: &mut usize, inp: &i64, opt_w: Option<i64>| {
            if let Some(w) = opt_w {
                let x = w * inp;
                store_max(x);
                x
            } else {
                *inp
            }
        };

        let store_max = store_max_base.clone();
        let add = move |_: &mut usize, x: &i64, y: &i64| {
            let res = x + y;
            store_max(res);
            res
        };

        let store_max = store_max_base.clone();
        let cmul = move |_: &mut usize, x: &i64, y: i64| {
            let res = x * y;
            store_max(res);
            res
        };

        let store_max = store_max_base.clone();
        let max = move |_: &mut usize, xs: &[i64]| {
            xs.iter()
                .map(|&x| {
                    store_max(x);
                    x
                })
                .max()
                .unwrap()
        };

        let act = |_: &mut usize, a: &str, x: &i64| match a {
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
        };

        let ops = NeuralNetOps {
            enc: Box::new(enc),
            sec: Box::new(move |_, _| 0),
            add: Box::new(add),
            cmul: Box::new(cmul),
            proj: Box::new(proj),
            max: Box::new(max),
            act: Box::new(act),
            zero: Box::new(|_| 0),
        };

        let layer_output = self.eval(&mut 0, input, &ops, false);
        let max_val = max_atomic.load(Ordering::SeqCst) as i64;
        (layer_output, max_val)
    }

    /// Evaluate this layer in plaintext.
    pub fn as_plaintext(&self, input: &Array3<i64>, _: usize) -> Array3<i64> {
        let ops = NeuralNetOps::plaintext();
        self.eval(&mut 0, input, &ops, false)
    }

    /// Perform an arithmetic fancy computation for this layer
    pub fn as_arith<W, F>(
        &self,
        b: &mut F,
        q: u128, // input modulus
        output_mod: u128,
        input: &Array3<CrtBundle<W>>,
        secret_weights: bool,
        secret_weights_owned: bool,
        accuracy: &Accuracy,
    ) -> Array3<CrtBundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W>,
    {
        let ops = NeuralNetOps::arithmetic(q, output_mod, secret_weights_owned, accuracy);
        self.eval(b, &input, &ops, secret_weights)
    }

    /// Perform a binary fancy computation for this layer
    pub fn as_binary<W, F>(
        &self,
        b: &mut F,
        nbits: usize,
        input: &Array3<BinaryBundle<W>>,
        secret_weights: bool,
        secret_weights_owned: bool,
    ) -> Array3<BinaryBundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W>,
    {
        let ops = NeuralNetOps::binary(nbits, secret_weights_owned);
        self.eval(b, &input, &ops, secret_weights)
    }

    pub fn as_binary_no_secret_weights<W, F>(
        &self,
        b: &mut F,
        nbits: usize,
        input: &Array3<BinaryBundle<W>>,
    ) -> Array3<BinaryBundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W>,
    {
        let ops = NeuralNetOps::binary_no_secret_weights(nbits);
        self.eval(b, &input, &ops, false)
    }

    pub fn as_finite_field<W, F>(
        &self,
        b: &mut F,
        field_size: usize,
        n_field_elems: usize,
        input: &Array3<Bundle<W>>,
    ) -> Array3<Bundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W>,
    {
        let ops = NeuralNetOps::finite_field(field_size, n_field_elems);
        self.eval(b, &input, &ops, false)
    }

    /// Polymorphic evaluation so we can run on `i64` directly as well as use this
    /// function to build `Circuit`s.
    fn eval<T, B>(
        &self,
        b: &mut B,
        input: &Array3<T>,
        ops: &NeuralNetOps<B, T>,
        secret_weights: bool,
    ) -> Array3<T>
    where
        T: Clone,
    {
        assert_eq!(self.input_dims(), input.dim());
        let (height, width, depth) = self.input_dims();

        let mut output: Array3<Option<T>> = Array3::default(self.output_dims());
        let nouts = self.output_size();

        match self {
            Layer::Dense {
                weights,
                biases,
                activation,
            } => {
                for neuron in 0..nouts {
                    let mut x = if secret_weights {
                        (ops.sec)(b, biases[neuron])
                    } else {
                        (ops.enc)(b, biases[neuron].expect("biases required for evaluation"))
                    };

                    for i in 0..height {
                        for j in 0..width {
                            for k in 0..depth {
                                let prod = if secret_weights {
                                    (ops.proj)(b, &input[(i, j, k)], weights[neuron][(i, j, k)])
                                } else {
                                    let w = weights[neuron][(i, j, k)].expect(
                                        "Dense layer eval: weights required for evaluation",
                                    );
                                    (ops.cmul)(b, &input[(i, j, k)], w)
                                };
                                x = (ops.add)(b, &x, &prod);
                            }
                        }
                    }

                    let z = (ops.act)(b, activation, &x);
                    output[(neuron, 0, 0)] = Some(z);
                }
            }

            Layer::Convolutional {
                filters,
                biases,
                kernel_shape,
                stride,
                activation,
                pad,
                ..
            } => {
                let (kheight, kwidth, kdepth) = *kernel_shape;
                let (stride_y, stride_x) = *stride;

                let zero_rows = if *pad {
                    (stride_y - 1) * height + kheight - stride_y
                } else {
                    0
                };
                let zero_cols = if *pad {
                    (stride_x - 1) * width + kwidth - stride_x
                } else {
                    0
                };

                let shift_y = ((zero_rows as f32) / 2.0).floor() as usize;
                let shift_x = ((zero_cols as f32) / 2.0).floor() as usize;

                let zero = (ops.zero)(b);

                for filterno in 0..filters.len() {
                    let mut h = 0;
                    while stride_y * h <= height - kheight + zero_rows {
                        // && h < oheight { // drop indices that dont make sense (padding=valid)

                        let mut w = 0;
                        while stride_x * w <= width - kwidth + zero_cols {
                            // && w < owidth {

                            let mut x = if secret_weights {
                                (ops.sec)(b, biases[filterno])
                            } else {
                                (ops.enc)(b, biases[filterno].expect("no bias"))
                            };

                            for i in 0..kheight {
                                let idx_y = stride_y * h + i;
                                for j in 0..kwidth {
                                    let idx_x = stride_x * w + j;
                                    for k in 0..kdepth {
                                        let pad_condition = *pad
                                            && ((idx_y < shift_y || idx_x < shift_x)
                                                || (idx_y >= height + shift_y
                                                    || idx_x >= width + shift_x));

                                        let input_val = if pad_condition {
                                            &zero
                                        } else {
                                            &input[(idx_y - shift_y, idx_x - shift_x, k)]
                                        };

                                        let prod = if secret_weights {
                                            (ops.proj)(b, &input_val, filters[filterno][(i, j, k)])
                                        } else {
                                            (ops.cmul)(
                                                b,
                                                &input_val,
                                                filters[filterno][(i, j, k)].expect("no weight"),
                                            )
                                        };
                                        x = (ops.add)(b, &x, &prod);
                                    }
                                }
                            }

                            let z = (ops.act)(b, activation, &x);
                            assert!(output[(h, w, filterno)].is_none());
                            output[(h, w, filterno)] = Some(z);
                            w += 1;
                        }
                        h += 1;
                    }
                }
            }

            Layer::MaxPooling2D {
                stride, size, pad, ..
            } => {
                let (pheight, pwidth) = *size;
                let (stride_y, stride_x) = *stride;

                let zero_rows = if *pad {
                    (stride_y - 1) * height + pheight - stride_y
                } else {
                    0
                };
                let zero_cols = if *pad {
                    (stride_x - 1) * width + pwidth - stride_x
                } else {
                    0
                };

                let shift_y = ((zero_rows as f32) / 2.0).floor() as usize;
                let shift_x = ((zero_cols as f32) / 2.0).floor() as usize;

                let zero = (ops.zero)(b);

                // create windows
                let mut windows = Vec::new();
                let mut y = 0;
                while stride_y * y <= height - pheight + zero_rows {
                    let mut x = 0;
                    while stride_x * x <= width - pwidth + zero_cols {
                        for z in 0..depth {
                            let mut vals = Vec::with_capacity(pheight * pwidth);
                            for h in 0..pheight {
                                let idx_y = stride_y * y + h;
                                for w in 0..pwidth {
                                    let idx_x = stride_x * x + w;

                                    let pad_condition = *pad
                                        && ((idx_y < shift_y || idx_x < shift_x)
                                            || (idx_y >= height + shift_y
                                                || idx_x >= width + shift_x));

                                    let val = if pad_condition {
                                        zero.clone()
                                    } else {
                                        input[(idx_y - shift_y, idx_x - shift_x, z)].clone()
                                    };

                                    vals.push(val);
                                }
                            }
                            windows.push(((y, x, z), vals));
                        }
                        x += 1;
                    }
                    y += 1;
                }

                for (coordinate, window) in windows.into_iter() {
                    let val = (ops.max)(b, &window);
                    output[coordinate] = Some(val);
                }
            }

            Layer::Flatten { output_shape, .. } => {
                output = input.map(|v| Option::Some(v.clone()));
                output = output.into_shape(*output_shape).unwrap();
            }

            Layer::Activation { activation, .. } => {
                let coordinates = iproduct!(0..height, 0..width, 0..depth).collect_vec();
                for c in coordinates.into_iter() {
                    let z = (ops.act)(b, activation, &input[c]);
                    output[c] = Some(z);
                }
            }
        }

        for (coordinate, val) in output.indexed_iter() {
            if val.is_none() {
                println!("{}: uninitialized output at {:?}", self.name(), coordinate);
                println!("exiting...");
                std::process::exit(1);
            }
        }

        output.mapv(|elem| {
            elem.unwrap_or_else(|| {
                println!("{}: uninitialized output", self.name());
                println!("exiting...");
                std::process::exit(1);
            })
        })
    }
}
