//! BERT Embeddings Example
//!
//! Demonstrates text encoding using BERT (bert-base-uncased).
//! Takes input text, tokenizes it, and generates contextual embeddings.
//!
//! Setup:
//!   1. Export BERT model:
//!      uv run --with transformers --with torch --with onnx scripts/export_bert.py
//!
//! Build: zig build example-bert
//! Run:   ./zig-out/bin/example-bert "Hello, how are you?"
//!
//! The model outputs:
//!   - last_hidden_state: [batch, sequence, 768] - Token embeddings
//!   - pooler_output: [batch, 768] - Sentence embedding (CLS token transformed)

const std = @import("std");
const onnx_zig = @import("onnx_zig");

const Session = onnx_zig.Session;
const WordPieceTokenizer = onnx_zig.WordPieceTokenizer;
const SpecialTokens = onnx_zig.SpecialTokens;

const MODEL_PATH = "models/bert/bert-base-uncased.onnx";
const VOCAB_PATH = "models/bert/vocab.txt";
const MAX_SEQ_LENGTH: usize = 128;
const HIDDEN_SIZE: usize = 768;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage(args[0]);
        return;
    }

    const input_text = args[1];

    // Load tokenizer
    std.debug.print("Loading tokenizer from {s}...\n", .{VOCAB_PATH});
    var tokenizer = WordPieceTokenizer.initFromFile(allocator, VOCAB_PATH) catch |err| {
        std.debug.print("Error loading vocabulary: {s}\n", .{@errorName(err)});
        std.debug.print("\nPlease run the export script first:\n", .{});
        std.debug.print("  uv run --with transformers --with torch --with onnx scripts/export_bert.py\n", .{});
        return;
    };
    defer tokenizer.deinit();

    std.debug.print("  Vocabulary size: {d}\n", .{tokenizer.vocabSize()});

    // Load BERT model
    std.debug.print("\nLoading BERT model from {s}...\n", .{MODEL_PATH});
    var session = Session.init(allocator, MODEL_PATH) catch |err| {
        std.debug.print("Error loading model: {s}\n", .{@errorName(err)});
        std.debug.print("\nPlease run the export script first:\n", .{});
        std.debug.print("  uv run --with transformers --with torch --with onnx scripts/export_bert.py\n", .{});
        return;
    };
    defer session.deinit();

    std.debug.print("  Inputs: {d}\n", .{session.getInputCount()});
    for (session.getInputNames()) |name| {
        std.debug.print("    - {s}\n", .{name});
    }
    std.debug.print("  Outputs: {d}\n", .{session.getOutputCount()});
    for (session.getOutputNames()) |name| {
        std.debug.print("    - {s}\n", .{name});
    }

    // Tokenize input
    std.debug.print("\nTokenizing: \"{s}\"\n", .{input_text});
    var encoded = try tokenizer.encode(allocator, input_text, MAX_SEQ_LENGTH);
    defer encoded.deinit(allocator);

    // Show tokens
    std.debug.print("  Tokens: [", .{});
    var token_count: usize = 0;
    for (encoded.input_ids, 0..) |id, i| {
        if (id == SpecialTokens.PAD) break;
        token_count = i + 1;
        if (i > 0) std.debug.print(", ", .{});
        if (tokenizer.idToToken(id)) |token| {
            std.debug.print("\"{s}\"", .{token});
        } else {
            std.debug.print("{d}", .{id});
        }
    }
    std.debug.print("]\n", .{});
    std.debug.print("  Token count: {d} (padded to {d})\n", .{ token_count, MAX_SEQ_LENGTH });

    // Prepare inputs for ONNX Runtime
    // BERT expects int64 inputs
    const input_shape = [_]i64{ 1, MAX_SEQ_LENGTH };

    // Run inference
    std.debug.print("\nRunning BERT inference...\n", .{});
    const start = std.time.nanoTimestamp();

    const outputs = try session.runI64(
        &[_][]const i64{
            encoded.input_ids,
            encoded.attention_mask,
            encoded.token_type_ids,
        },
        &[_][]const i64{
            &input_shape,
            &input_shape,
            &input_shape,
        },
    );
    defer session.freeOutputs(outputs);

    const end = std.time.nanoTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    std.debug.print("  Inference time: {d:.2} ms\n", .{elapsed_ms});

    // Process outputs
    // outputs[0] = last_hidden_state [1, seq_len, 768]
    // outputs[1] = pooler_output [1, 768]

    const last_hidden_state = outputs[0].data;
    const pooler_output = outputs[1].data;

    std.debug.print("\n=== Results ===\n", .{});

    // Pooler output (sentence embedding)
    std.debug.print("\nPooler output (sentence embedding):\n", .{});
    std.debug.print("  Shape: [1, {d}]\n", .{HIDDEN_SIZE});
    std.debug.print("  First 10 values: [", .{});
    for (0..10) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.4}", .{pooler_output[i]});
    }
    std.debug.print(", ...]\n", .{});

    // Compute L2 norm of pooler output
    var norm: f32 = 0;
    for (pooler_output) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);
    std.debug.print("  L2 norm: {d:.4}\n", .{norm});

    // Token embeddings (CLS token)
    std.debug.print("\n[CLS] token embedding (first token):\n", .{});
    std.debug.print("  First 10 values: [", .{});
    for (0..10) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.4}", .{last_hidden_state[i]});
    }
    std.debug.print(", ...]\n", .{});

    // Show embedding for each actual token
    std.debug.print("\nPer-token embedding norms:\n", .{});
    for (0..token_count) |t| {
        const offset = t * HIDDEN_SIZE;
        var token_norm: f32 = 0;
        for (0..HIDDEN_SIZE) |i| {
            const v = last_hidden_state[offset + i];
            token_norm += v * v;
        }
        token_norm = @sqrt(token_norm);

        const token_str = tokenizer.idToToken(encoded.input_ids[t]) orelse "???";
        std.debug.print("  [{d:2}] {s:12} norm={d:.4}\n", .{ t, token_str, token_norm });
    }

    std.debug.print("\nDone!\n", .{});
}

fn printUsage(program: []const u8) void {
    std.debug.print(
        \\BERT Embeddings Generator
        \\
        \\Generate contextual embeddings using BERT (bert-base-uncased).
        \\
        \\USAGE:
        \\  {s} "<text>"
        \\
        \\EXAMPLES:
        \\  {s} "Hello, how are you?"
        \\  {s} "The quick brown fox jumps over the lazy dog."
        \\  {s} "Machine learning is fascinating."
        \\
        \\SETUP:
        \\  First, export the BERT model:
        \\    uv run --with transformers --with torch --with onnx scripts/export_bert.py
        \\
    , .{ program, program, program, program });
}
