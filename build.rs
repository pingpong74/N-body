use std::fs;
use std::process::Command;

fn compile_shader(src: &str, out: &str) {
    println!("cargo:rerun-if-changed={}", src);

    let status = Command::new("slangc")
        .args([
            src, "-target", "spirv", "-entry", "main", "-stage", "compute", "-o", out,
        ])
        .stderr(std::process::Stdio::inherit()) // so you can see slangc errors
        .status()
        .expect("failed to run slangc — is it in PATH?");

    assert!(status.success(), "shader compilation failed: {}", src);
}

fn main() {
    fs::create_dir_all("compiled").unwrap();

    // morton codes
    compile_shader("shaders/build/morton_code.slang", "compiled/morton_code.spv");

    // radix sort
    compile_shader("shaders/build/radix_sort/histogram.slang", "compiled/histogram.spv");
    compile_shader("shaders/build/radix_sort/sort.slang", "compiled/sort.spv");
}
