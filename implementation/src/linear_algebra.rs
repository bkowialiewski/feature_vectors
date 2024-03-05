pub fn standardize(v: &mut Vec<f64>)  {
    let vec_norm = norm(v);
    v.iter_mut().for_each(|x| *x /= vec_norm);
}

pub fn norm(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| x.powf(2.0)).sum::<f64>().sqrt()
}

pub fn dot(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    (0..v1.len()).map(|i| v1[i] * v2[i]).sum()
}


