pub mod lib;
use lib::{activations::SIGMOID, network::Network};

// A project implemented using Neal Wangs tutorial https://www.youtube.com/watch?v=FI-8L-hobDY&list=LL
// X-or prediction
// 0, 0 -> 0
// 1, 0 -> 1
// 0, 1 -> 1
// 1, 1 -> 0

fn main() {
    let inputs = vec![
        vec![0.0,0.0],
        vec![0.0,1.0],
        vec![1.0,0.0],
        vec![1.0,1.0]
    ];

    let targets = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    let mut network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);

    println!("0 and 0: {:?}", network.feed_forward(vec![0.0,0.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0,0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0,1.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0,1.0]));

    network.train(inputs.clone(), targets.clone(), 500);
    println!("0 and 0: {:?}", network.feed_forward(vec![0.0,0.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0,0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0,1.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0,1.0]));

    network.train(inputs, targets, 700);
    println!("0 and 0: {:?}", network.feed_forward(vec![0.0,0.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0,0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0,1.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0,1.0]));

}
