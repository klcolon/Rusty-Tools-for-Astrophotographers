//import dependencies
use wasm_bindgen::prelude::*;
use image::GenericImage;

#[wasm_bindgen]
// alert funtion to be called in greet
extern {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
// greet function 
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

#[wasm_bindgen]
//image reading function
pub fn read_image(file_path:&str) {

    // Read the image file into an ImageBuffer
    let img = image::open(file_path).expect("failed to open image file");

    // Get the width and height of the image
    let (width, height) = img.dimensions();

    // Print some basic information about the image
    println!("Image dimensions: {}x{}", width, height);
}




