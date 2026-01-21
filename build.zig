const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // Detect WASM target
    const is_wasm = target.result.cpu.arch == .wasm32 or target.result.cpu.arch == .wasm64;

    // ONNX Runtime library configuration
    // Users can override with: zig build -Donnxruntime_path=/path/to/onnxruntime
    const onnxruntime_path = b.option([]const u8, "onnxruntime_path", "Path to ONNX Runtime installation") orelse null;

    // =========================================================================
    // WASM Build
    // =========================================================================
    if (is_wasm) {
        // Create WASM library for browser deployment
        const wasm_lib = b.addExecutable(.{
            .name = "onnx_zig",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/wasm_exports.zig"),
                .target = target,
                .optimize = if (optimize == .Debug) .ReleaseSmall else optimize,
            }),
        });

        // Export all declared export functions
        wasm_lib.entry = .disabled;
        wasm_lib.rdynamic = true;

        // Install WASM artifact
        const install_wasm = b.addInstallArtifact(wasm_lib, .{
            .dest_dir = .{ .override = .{ .custom = "wasm" } },
        });
        b.getInstallStep().dependOn(&install_wasm.step);

        // Create a step for building WASM specifically
        const wasm_step = b.step("wasm", "Build WebAssembly module for browser");
        wasm_step.dependOn(&install_wasm.step);

        // Copy JavaScript loader and HTML demos to output
        const install_js = b.addInstallFile(b.path("wasm/loader.js"), "wasm/loader.js");
        const install_html = b.addInstallFile(b.path("wasm/index.html"), "wasm/index.html");
        const install_embedding_demo = b.addInstallFile(b.path("wasm/embedding_demo.html"), "wasm/embedding_demo.html");
        wasm_step.dependOn(&install_js.step);
        wasm_step.dependOn(&install_html.step);
        wasm_step.dependOn(&install_embedding_demo.step);

        return;
    }

    // =========================================================================
    // Native Build (with ONNX Runtime)
    // =========================================================================

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("onnx_zig", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });

    // Link ONNX Runtime library to the module
    if (onnxruntime_path) |ort_path| {
        mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{ort_path}) });
        mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{ort_path}) });
    }
    mod.linkSystemLibrary("onnxruntime", .{});

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "onnx_zig",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "onnx_zig" is the name you will use in your source code to
                // import this module (e.g. `@import("onnx_zig")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "onnx_zig", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // =========================================================================
    // CLI Tool
    // =========================================================================
    const cli = b.addExecutable(.{
        .name = "onnx-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnx_zig", .module = mod },
            },
        }),
    });
    b.installArtifact(cli);

    const cli_step = b.step("cli", "Build the CLI tool");
    cli_step.dependOn(&b.addInstallArtifact(cli, .{}).step);

    const run_cli = b.addRunArtifact(cli);
    if (b.args) |args| run_cli.addArgs(args);
    const cli_run_step = b.step("cli-run", "Run the CLI tool");
    cli_run_step.dependOn(&run_cli.step);

    // =========================================================================
    // Examples
    // =========================================================================

    // Basic inference example
    const example_basic = b.addExecutable(.{
        .name = "example-basic",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/zig/basic_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnx_zig", .module = mod },
            },
        }),
    });

    const install_example_basic = b.addInstallArtifact(example_basic, .{});
    const example_basic_step = b.step("example-basic", "Build basic inference example");
    example_basic_step.dependOn(&install_example_basic.step);

    const run_example_basic = b.addRunArtifact(example_basic);
    if (b.args) |args| run_example_basic.addArgs(args);
    const run_example_basic_step = b.step("run-example-basic", "Run basic inference example");
    run_example_basic_step.dependOn(&run_example_basic.step);

    // MNIST classifier example
    const example_mnist = b.addExecutable(.{
        .name = "example-mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/zig/mnist_classifier.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnx_zig", .module = mod },
            },
        }),
    });

    const install_example_mnist = b.addInstallArtifact(example_mnist, .{});
    const example_mnist_step = b.step("example-mnist", "Build MNIST classifier example");
    example_mnist_step.dependOn(&install_example_mnist.step);

    const run_example_mnist = b.addRunArtifact(example_mnist);
    if (b.args) |args| run_example_mnist.addArgs(args);
    const run_example_mnist_step = b.step("run-example-mnist", "Run MNIST classifier example");
    run_example_mnist_step.dependOn(&run_example_mnist.step);

    // BERT embeddings example
    const example_bert = b.addExecutable(.{
        .name = "example-bert",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/zig/bert_embeddings.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnx_zig", .module = mod },
            },
        }),
    });

    const install_example_bert = b.addInstallArtifact(example_bert, .{});
    const example_bert_step = b.step("example-bert", "Build BERT embeddings example");
    example_bert_step.dependOn(&install_example_bert.step);

    const run_example_bert = b.addRunArtifact(example_bert);
    if (b.args) |args| run_example_bert.addArgs(args);
    const run_example_bert_step = b.step("run-example-bert", "Run BERT embeddings example");
    run_example_bert_step.dependOn(&run_example_bert.step);

    // Build all examples
    const examples_step = b.step("examples", "Build all examples");
    examples_step.dependOn(&install_example_basic.step);
    examples_step.dependOn(&install_example_mnist.step);
    examples_step.dependOn(&install_example_bert.step);
}
