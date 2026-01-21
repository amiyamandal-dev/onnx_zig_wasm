//! WordPiece Tokenizer for BERT models.
//!
//! Implements the WordPiece tokenization algorithm used by BERT and similar
//! transformer models. Supports loading vocabularies from file and tokenizing
//! text into token IDs.
//!
//! ## Example
//!
//! ```zig
//! var tokenizer = try WordPieceTokenizer.initFromFile(allocator, "vocab.txt");
//! defer tokenizer.deinit();
//!
//! const tokens = try tokenizer.encode(allocator, "Hello, world!", 128);
//! defer allocator.free(tokens.input_ids);
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Special token IDs for BERT
pub const SpecialTokens = struct {
    pub const PAD: i64 = 0;
    pub const UNK: i64 = 100;
    pub const CLS: i64 = 101;
    pub const SEP: i64 = 102;
    pub const MASK: i64 = 103;
};

/// Encoded output from tokenizer
pub const EncodedInput = struct {
    input_ids: []i64,
    attention_mask: []i64,
    token_type_ids: []i64,

    pub fn deinit(self: *EncodedInput, allocator: Allocator) void {
        allocator.free(self.input_ids);
        allocator.free(self.attention_mask);
        allocator.free(self.token_type_ids);
    }
};

/// WordPiece tokenizer for BERT models
pub const WordPieceTokenizer = struct {
    const Self = @This();

    allocator: Allocator,
    vocab: std.StringHashMap(i64),
    vocab_list: [][]const u8,
    vocab_count: usize,
    unk_token_id: i64,
    max_word_len: usize,

    /// Initialize tokenizer from a vocabulary file (one token per line)
    pub fn initFromFile(allocator: Allocator, vocab_path: []const u8) !Self {
        // Read entire file
        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const content = try allocator.alloc(u8, file_size);
        defer allocator.free(content);

        const bytes_read = try file.readAll(content);
        const data = content[0..bytes_read];

        var vocab = std.StringHashMap(i64).init(allocator);
        errdefer vocab.deinit();

        // Count lines first
        var line_count: usize = 0;
        var iter = std.mem.splitScalar(u8, data, '\n');
        while (iter.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len > 0) line_count += 1;
        }

        // Allocate vocab list
        const vocab_list = try allocator.alloc([]const u8, line_count);
        errdefer allocator.free(vocab_list);

        // Parse lines
        var token_id: i64 = 0;
        var idx: usize = 0;
        var line_iter = std.mem.splitScalar(u8, data, '\n');

        while (line_iter.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            // Copy the token
            const token_copy = try allocator.dupe(u8, trimmed);
            errdefer allocator.free(token_copy);

            try vocab.put(token_copy, token_id);
            vocab_list[idx] = token_copy;
            idx += 1;
            token_id += 1;
        }

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_list = vocab_list,
            .vocab_count = idx,
            .unk_token_id = SpecialTokens.UNK,
            .max_word_len = 200,
        };
    }

    /// Free tokenizer resources
    pub fn deinit(self: *Self) void {
        for (self.vocab_list[0..self.vocab_count]) |item| {
            self.allocator.free(item);
        }
        self.allocator.free(self.vocab_list);
        self.vocab.deinit();
    }

    /// Get vocabulary size
    pub fn vocabSize(self: *const Self) usize {
        return self.vocab_count;
    }

    /// Look up token ID by string
    pub fn tokenToId(self: *const Self, token: []const u8) i64 {
        return self.vocab.get(token) orelse self.unk_token_id;
    }

    /// Look up token string by ID
    pub fn idToToken(self: *const Self, id: i64) ?[]const u8 {
        if (id < 0 or id >= @as(i64, @intCast(self.vocab_count))) {
            return null;
        }
        return self.vocab_list[@intCast(id)];
    }

    /// Encode text into token IDs with padding to max_length
    pub fn encode(self: *const Self, allocator: Allocator, text: []const u8, max_length: usize) !EncodedInput {
        // Tokenize into a dynamically grown buffer
        var tokens = std.ArrayListUnmanaged(i64){};
        defer tokens.deinit(allocator);

        // Add [CLS] token
        try tokens.append(allocator, SpecialTokens.CLS);

        // Tokenize the text
        try self.tokenizeInto(allocator, text, &tokens);

        // Add [SEP] token
        try tokens.append(allocator, SpecialTokens.SEP);

        // Truncate if necessary
        const actual_len = @min(tokens.items.len, max_length);

        // Allocate output arrays
        const input_ids = try allocator.alloc(i64, max_length);
        errdefer allocator.free(input_ids);

        const attention_mask = try allocator.alloc(i64, max_length);
        errdefer allocator.free(attention_mask);

        const token_type_ids = try allocator.alloc(i64, max_length);

        // Fill arrays
        for (0..max_length) |i| {
            if (i < actual_len) {
                input_ids[i] = tokens.items[i];
                attention_mask[i] = 1;
            } else {
                input_ids[i] = SpecialTokens.PAD;
                attention_mask[i] = 0;
            }
            token_type_ids[i] = 0; // Single segment
        }

        return EncodedInput{
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .token_type_ids = token_type_ids,
        };
    }

    /// Tokenize text and append token IDs to the list
    fn tokenizeInto(self: *const Self, allocator: Allocator, text: []const u8, tokens: *std.ArrayListUnmanaged(i64)) !void {
        // Simple whitespace + punctuation tokenization, then WordPiece
        var i: usize = 0;
        while (i < text.len) {
            // Skip whitespace
            if (isWhitespace(text[i])) {
                i += 1;
                continue;
            }

            // Handle punctuation as single tokens
            if (isPunctuation(text[i])) {
                const punct = text[i .. i + 1];
                const id = self.tokenToId(punct);
                try tokens.append(allocator, id);
                i += 1;
                continue;
            }

            // Extract word (sequence of non-whitespace, non-punctuation)
            const word_start = i;
            while (i < text.len and !isWhitespace(text[i]) and !isPunctuation(text[i])) {
                i += 1;
            }
            const word = text[word_start..i];

            // Apply WordPiece to the word
            try self.wordPiece(allocator, word, tokens);
        }
    }

    /// Apply WordPiece algorithm to a single word
    fn wordPiece(self: *const Self, allocator: Allocator, word: []const u8, tokens: *std.ArrayListUnmanaged(i64)) !void {
        if (word.len == 0) return;

        // Convert to lowercase for BERT uncased
        var lower_buf: [256]u8 = undefined;
        const lower_word = toLower(word, &lower_buf);

        if (lower_word.len > self.max_word_len) {
            try tokens.append(allocator, self.unk_token_id);
            return;
        }

        // Try to find whole word first
        if (self.vocab.get(lower_word)) |id| {
            try tokens.append(allocator, id);
            return;
        }

        // WordPiece subword tokenization
        var start: usize = 0;
        var sub_tokens = std.ArrayListUnmanaged(i64){};
        defer sub_tokens.deinit(allocator);

        while (start < lower_word.len) {
            var end = lower_word.len;
            var found = false;

            while (start < end) {
                var substr_buf: [258]u8 = undefined; // 256 + "##"
                const substr = if (start > 0) blk: {
                    substr_buf[0] = '#';
                    substr_buf[1] = '#';
                    const len = end - start;
                    @memcpy(substr_buf[2 .. 2 + len], lower_word[start..end]);
                    break :blk substr_buf[0 .. 2 + len];
                } else lower_word[start..end];

                if (self.vocab.get(substr)) |id| {
                    try sub_tokens.append(allocator, id);
                    found = true;
                    break;
                }
                end -= 1;
            }

            if (!found) {
                // Unable to tokenize - return UNK for whole word
                tokens.clearRetainingCapacity();
                try tokens.append(allocator, self.unk_token_id);
                return;
            }
            start = end;
        }

        // Add all sub-tokens
        try tokens.appendSlice(allocator, sub_tokens.items);
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return switch (c) {
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
            else => false,
        };
    }

    fn toLower(s: []const u8, buf: []u8) []const u8 {
        const len = @min(s.len, buf.len);
        for (0..len) |i| {
            buf[i] = std.ascii.toLower(s[i]);
        }
        return buf[0..len];
    }
};

// =============================================================================
// Tests
// =============================================================================

test "WordPieceTokenizer - special tokens" {
    try std.testing.expectEqual(@as(i64, 0), SpecialTokens.PAD);
    try std.testing.expectEqual(@as(i64, 100), SpecialTokens.UNK);
    try std.testing.expectEqual(@as(i64, 101), SpecialTokens.CLS);
    try std.testing.expectEqual(@as(i64, 102), SpecialTokens.SEP);
}
