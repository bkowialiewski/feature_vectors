mod linear_algebra;

use rand_distr::{Distribution, Normal, Uniform};
use rand::thread_rng;
use crate::linear_algebra::{dot, standardize};
use std::collections::HashMap;
use serde_json;
use std::fs::File;

fn main() {

    let n_items = 50;
    let n_features = 100_000;
    let sim: f64 = 0.9;

    let decay: Vec<f64> = Uniform::new(0.5, 1.0).sample_iter(&mut thread_rng()).take(n_items).collect();

    // items' current strength
    let mut strength = vec![1.0; n_items];
    // compute their original similarity
    let mut sim_x = vec![vec![1.0; n_items]; n_items];
    for i in 0..n_items {
        for j in 0..n_items {
            if i != j {
                sim_x[i][j] = sim;
            } 
        }
    }

    // updated similarity
    let mut sim_x_bis = vec![vec![1.0; n_items]; n_items];
    // updated strength
    (0..n_items).for_each(|i| strength[i] *= decay[i]);
    // updated similarity
    for i in 0..n_items {
        for j in 0..n_items {
            sim_x_bis[i][j] = strength[i] * strength[j] * (sim_x[i][i] * sim_x[i][j]);
        }
    }

    // create original vectors
    let mut vectors = feature_vector(n_items, n_features, sim);
    // apply decay to them
    for i in 0..n_items {
        vectors[i].iter_mut().for_each(|x| *x *= decay[i]);
    }
    // then compute their similarity
    let mut sim_x_features = vec![vec![0.0; n_items]; n_items];
    for i in 0..n_items {
        for j in 0..n_items {
            sim_x_features[i][j] = dot(&vectors[i], &vectors[j]);
        }
    }

    // print_matrix(&sim_x_features);
    // print_matrix(&sim_x_bis);
    let mut map = HashMap::new();
    map.insert("sim_x_features".to_string(), sim_x_features);
    map.insert("sim_x_closed_form".to_string(), sim_x_bis);

    let file = File::create("output.txt").unwrap();
    serde_json::to_writer(file, &mut map).unwrap();

}

#[warn(dead_code)]
fn print_matrix(m: &Vec<Vec<f64>>) {
    m.iter().for_each(|x| println!("{:.2?}", x));
    println!("");
}

pub fn generate_vector(n_features: usize) -> Vec<f64> {

    Normal::new(0.0, 1.0)
        .unwrap()
        .sample_iter(&mut thread_rng())
        .take(n_features)
        .collect()

}

pub fn feature_vector(n_items: usize, n_features: usize, sim: f64) -> Vec<Vec<f64>> {

    let mut vectors = vec![vec![0.0; n_features]; n_items];

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(0.0, 1.0);

    let prototype = generate_vector(n_features);
    for i in 0..n_items {
        vectors[i] = prototype.to_vec();
        for j in vectors[i].iter_mut() {
            if uniform.sample(&mut rng) > sim.sqrt() {
                *j = normal.sample(&mut rng);
            }
        }
    }

    for i in 0..n_items {
        standardize(&mut vectors[i]);
    }

    vectors

}
