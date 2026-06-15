use std::fs;
use std::process::Command;

fn compile_shader(src: &str, out: &str) {
    println!("cargo:rerun-if-changed={}", src);

    let status = Command::new("slangc")
        .args([
            src, "-target", "spirv", "-entry", "main", "-o", out,
        ])
        .stderr(std::process::Stdio::inherit())
        .status()
        .expect("failed to run slangc — is it in PATH?");

    assert!(status.success(), "shader compilation failed: {}", src);
}

fn main() {
    fs::create_dir_all("compiled").unwrap();

    // bounds calculation
    compile_shader("shaders/build/bounds.slang", "compiled/bounds.spv");

    // morton codes
    compile_shader("shaders/build/morton_code.slang", "compiled/morton_code.spv");

    // radix sort
    compile_shader("shaders/build/radix_sort/histogram.slang", "compiled/histogram.spv");
    compile_shader("shaders/build/radix_sort/sort.slang", "compiled/sort.spv");

    // tree building
    compile_shader("shaders/build/tree.slang", "compiled/tree.spv");
    compile_shader("shaders/build/parent.slang", "compiled/parent.spv");
    compile_shader("shaders/build/com.slang", "compiled/com.spv");

    // integration
    compile_shader("shaders/build/integrate.slang", "compiled/integrate.spv");

    // drawing
    compile_shader("shaders/draw/vertex.slang", "compiled/vertex.spv");
    compile_shader("shaders/draw/fragment.slang", "compiled/fragment.spv");
}
