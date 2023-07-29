const std = @import("std");
const mem = std.mem;

const Allocator = mem.Allocator;

const Error = error{ InvalidPrompt, InvalidArgs, InvalidModelFile, InvalidTokenizerFile };

// ----------------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

const Config = struct {
    dim: u32, // transformer dimension
    hidden_dim: u32, // for ffn layers
    n_layers: u32, // number of layers
    n_heads: u32, // number of query heads
    n_kv_heads: u32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: u32, // vocabulary size, usually 256 (byte-level)
    seq_len: u32, // max sequence length
};

const TransformerWeights = struct {
    // token embedding table
    token_embedding_table: []f32, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: []f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: []f32, // (layer, dim)
    // weights for matmuls
    wq: []f32, // (layer, dim, dim)
    wk: []f32, // (layer, dim, dim)
    wv: []f32, // (layer, dim, dim)
    wo: []f32, // (layer, dim, dim)
    // weights for ffn
    w1: []f32, // (layer, hidden_dim, dim)
    w2: []f32, // (layer, dim, hidden_dim)
    w3: []f32, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: []f32, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: []f32, // (seq_len, dim/2)
    freq_cis_imag: []f32, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: []f32,
};

const RunState = struct {
    const Self = @This();

    allocator: Allocator,

    // current wave of activations
    x: []f32, // activation at current time stamp (dim,)
    xb: []f32, // same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, // query (dim,)
    k: []f32, // key (dim,)
    v: []f32, // value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // kv cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    pub fn init(allocator: Allocator, p: *const Config) !Self {
        return Self{
            .allocator = allocator,
            .x = try allocator.alloc(f32, @as(usize, p.dim)),
            .xb = try allocator.alloc(f32, @as(usize, p.dim)),
            .xb2 = try allocator.alloc(f32, @as(usize, p.dim)),
            .hb = try allocator.alloc(f32, @as(usize, p.hidden_dim)),
            .hb2 = try allocator.alloc(f32, @as(usize, p.hidden_dim)),
            .q = try allocator.alloc(f32, @as(usize, p.dim)),
            .k = try allocator.alloc(f32, @as(usize, p.dim)),
            .v = try allocator.alloc(f32, @as(usize, p.dim)),
            .att = try allocator.alloc(f32, @as(usize, p.n_heads * p.seq_len)),
            .logits = try allocator.alloc(f32, @as(usize, p.vocab_size)),
            .key_cache = try allocator.alloc(f32, @as(usize, p.n_layers * p.seq_len * p.dim)),
            .value_cache = try allocator.alloc(f32, @as(usize, p.n_layers * p.seq_len * p.dim)),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.x);
        self.allocator.free(self.xb);
        self.allocator.free(self.xb2);
        self.allocator.free(self.hb);
        self.allocator.free(self.hb2);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.att);
        self.allocator.free(self.logits);
        self.allocator.free(self.key_cache);
        self.allocator.free(self.value_cache);
    }
};

// ----------------------------------------------------------------------------------
// initialization: read from checkpoint

fn checkpoint_init_weights(w: *TransformerWeights, p: *const Config, f: []f32, shared_weights: bool) void {
    var ptr: u32 = 0;
    var l: u32 = 0;

    l = p.vocab_size * p.dim;
    w.token_embedding_table = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim;
    w.rms_att_weight = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.dim;
    w.wq = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.dim;
    w.wk = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.dim;
    w.wv = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.dim;
    w.wo = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim;
    w.rms_ffn_weight = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.hidden_dim;
    w.w1 = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.hidden_dim * p.dim;
    w.w2 = f[ptr .. ptr + l];
    ptr += l;

    l = p.n_layers * p.dim * p.hidden_dim;
    w.w3 = f[ptr .. ptr + l];
    ptr += l;

    l = p.dim;
    w.rms_final_weight = f[ptr .. ptr + l];
    ptr += l;

    const head_size = p.dim / p.n_heads;
    l = p.seq_len * head_size / 2;
    w.freq_cis_real = f[ptr .. ptr + l];
    ptr += l;

    l = p.seq_len * head_size / 2;
    w.freq_cis_imag = f[ptr .. ptr + l];
    ptr += l;

    l = p.dim * p.vocab_size;
    w.wcls = if (shared_weights) w.token_embedding_table else f[ptr .. ptr + l];
}

// ----------------------------------------------------------------------------------
// neural net blocks

fn accum(a: []f32, b: []f32, size: u32) void {
    var i: u32 = 0;
    while (i < size) : (i += 1) {
        a[i] += b[i];
    }
}

fn rmsnorm(o: []f32, x: []f32, weight: []f32, size: u32) void {
    // calculate sum of squares
    var ss: f32 = 0.0;
    var j: u32 = 0;
    while (j < size) : (j += 1) {
        ss += x[j] * x[j];
    }
    const div: f32 = @floatFromInt(size);
    ss = ss / div;
    ss += 1e-5;
    ss = 1.0 / @sqrt(ss);
    // normalize and scale
    j = 0;
    while (j < size) : (j += 1) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn softmax(x: []f32, size: u32) void {
    // find max value (for numerical stability)
    var max_val = x[0];
    var i: u32 = 1;
    while (i < size) : (i += 1) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    var sum: f32 = 0.0;
    i = 0;
    while (i < size) : (i += 1) {
        x[i] = @exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    i = 0;
    while (i < size) : (i += 1) {
        x[i] /= sum;
    }
}

// TODO: parallism. Zig does not support openmp
fn matmul(xout: []f32, x: []f32, w: []f32, n: u32, d: u32) void {
    // W (d,n) @ x(n,) -> xout (d,)
    var i: u32 = 0;
    while (i < d) : (i += 1) {
        var val: f32 = 0.0;
        var j: u32 = 0;
        while (j < n) : (j += 1) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn transformer(token: u32, pos: u32, p: *Config, s: *RunState, w: *TransformerWeights) void {

    // a few convenience variables
    const x = s.x;
    const dim = p.dim;
    const hidden_dim = p.hidden_dim;
    const head_size: u32 = dim / p.n_heads;

    // copy the token embedding into x
    const content_row = w.token_embedding_table[token * dim .. token * dim + dim];
    @memcpy(x, content_row);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    const freq_cis_real_row = w.freq_cis_real[pos * head_size / 2 ..];
    const freq_cis_imag_row = w.freq_cis_imag[pos * head_size / 2 ..];

    // forward all the layers
    var l: u32 = 0;
    while (l < p.n_layers) : (l += 1) {

        // attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight[l * dim .. (l + 1) * dim], dim);

        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq[l * dim * dim .. (l + 1) * dim * dim], dim, dim);
        matmul(s.k, s.xb, w.wk[l * dim * dim .. (l + 1) * dim * dim], dim, dim);
        matmul(s.v, s.xb, w.wv[l * dim * dim .. (l + 1) * dim * dim], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        var h: u32 = 0;
        while (h < p.n_heads) : (h += 1) {
            // get the q and k vectors for this head
            const q = s.q[h * head_size .. (h + 1) * head_size];
            const k = s.k[h * head_size .. (h + 1) * head_size];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            var i: u32 = 0;
            while (i < head_size) : (i += 2) {
                const q0 = q[i];
                const q1 = q[i + 1];
                const k0 = k[i];
                const k1 = k[i + 1];
                const fcr = freq_cis_real_row[i / 2];
                const fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key, value at this time step (pos) to our kv cache
        const loff = l * p.seq_len * dim; // kv cache layer offset for convenience
        const key_cache_row = s.key_cache[loff + pos * dim .. loff + (pos + 1) * dim];
        const value_cache_row = s.value_cache[loff + pos * dim .. loff + (pos + 1) * dim];
        @memcpy(key_cache_row, s.k);
        @memcpy(value_cache_row, s.v);

        // multihead attention. iterate over all heads
        h = 0;
        while (h < p.n_heads) : (h += 1) {
            // get the query vector for this head
            const q = s.q[h * head_size .. (h + 1) * head_size];
            // attention scores for this head
            const att = s.att[h * p.seq_len .. h * p.seq_len + pos + 1];
            // iterate over all timesteps, including the current one
            var t: u32 = 0;
            while (t <= pos) : (t += 1) {
                // get the key vector for this head and at this timestep
                const k = s.key_cache[loff + t * dim + h * head_size .. loff + t * dim + (h + 1) * head_size];
                // calculate the attention score as the dot product of q and k
                var score: f32 = 0.0;
                var i: u32 = 0;
                while (i < head_size) : (i += 1) {
                    score += q[i] * k[i];
                }
                const sqrt: f32 = @floatFromInt(head_size);
                score /= @sqrt(sqrt);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            const xb = s.xb[h * head_size .. (h + 1) * head_size];
            @memset(xb, 0.0);
            t = 0;
            while (t <= pos) : (t += 1) {
                // get the value vector for this head and at this timestep
                const v = s.value_cache[loff + t * dim + h * head_size .. loff + t * dim + (h + 1) * head_size];
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value into xb
                var i: u32 = 0;
                while (i < head_size) : (i += 1) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the ouput of the attention
        matmul(s.xb2, s.xb, w.wo[l * dim * dim .. (l + 1) * dim * dim], dim, dim);

        // residual connection back into x
        accum(x, s.xb2, dim);

        // ffn rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim .. (l + 1) * dim], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], dim, hidden_dim);

        // F.silu; silu(x)=x*sigma(x), where sigma(x) is the logistic sigmoid
        var i: u32 = 0;
        while (i < hidden_dim) : (i += 1) {
            s.hb[i] = s.hb[i] * (1.0 / (1.0 + @exp(-s.hb[i])));
        }

        // elementwise multiply with w3(x)
        i = 0;
        while (i < hidden_dim) : (i += 1) {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], hidden_dim, dim);

        // residual connection
        accum(x, s.xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);

    // classifier into logits
    matmul(s.logits, x, w.wcls, dim, p.vocab_size);
}

// ----------------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

fn str_lookup(str: []const u8, vocab: []const []const u8, vocab_size: u32) ?u32 {
    // find the first perfect match for str in vocab, return its index or null if not found
    var i: u32 = 0;
    while (i < vocab_size) : (i += 1) {
        if (std.mem.eql(u8, str, vocab[i])) {
            return i;
        }
    }
    return null;
}

fn bpe_encode(allocator: Allocator, text: []const u8, vocab: []const []const u8, vocab_scores: []f32, vocab_size: u32, max_token_length: u32, tokens: []u32, n_tokens: *u32) !void {
    const str_buffer = try allocator.alloc(u8, max_token_length * 2 + 1);
    defer allocator.free(str_buffer);

    // first encode every individual byte in the input string
    n_tokens.* = 0; // the number of tokens
    for (text) |char| {
        const str = try std.fmt.bufPrint(str_buffer, "{c}\x00", .{char});
        const id_option = str_lookup(str, vocab, vocab_size);
        if (id_option) |id| {
            tokens[n_tokens.*] = id;
            n_tokens.* += 1;
        } else {
            return Error.InvalidPrompt;
        }
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        var best_score: f32 = -1e10;
        var best_id_option: ?u32 = null;
        var best_idx_option: ?u32 = null;

        var i: u32 = 0;
        while (i < (n_tokens.* - 1)) : (i += 1) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            const str = try std.fmt.bufPrint(str_buffer, "{s}{s}\x00", .{ vocab[tokens[i]], vocab[tokens[i + 1]] });
            const id_option = str_lookup(str, vocab, vocab_size);
            if (id_option) |id| {
                if (vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocab_scores[id];
                    best_id_option = id;
                    best_idx_option = i;
                }
            }
        }

        if (best_idx_option) |best_idx| {
            if (best_id_option) |best_id| {
                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                tokens[best_idx] = best_id;
            }

            // delete token at position best_idx+1; shift the entire sequence back 1
            i = best_idx + 1;
            while (i < n_tokens.* - 1) : (i += 1) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens.* -= 1; // token length decreased
        } else {
            break; // we couldn't find any more pairs to merge, so we're done
        }
    }
}

// ----------------------------------------------------------------------------------
// utilities

fn time_in_ms() i64 {
    // return time in milliseconds, for benchmarking the model speed
    return std.time.milliTimestamp();
}

fn random_u32(rng: *std.rand.DefaultPrng) u32 {
    return rng.random().int(u32);
}

fn random_f32(rng: *std.rand.DefaultPrng) f32 {
    return rng.random().float(f32);
}

fn sample(rng: *std.rand.DefaultPrng, probabilities: []f32, n: u32) u32 {
    // sample index from probabilities, they must sum to 1
    const r = random_f32(rng);
    var cdf: f32 = 0.0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

fn argmax(v: []f32, n: u32) u32 {
    // return argmax of v in elements 0..n
    var max_i: u32 = 0;
    var max_p: f32 = v[0];
    var i: u32 = 1;
    while (i < n) : (i += 1) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------------

pub fn main() anyerror!void {
    // Initialize allocator.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var rng = std.rand.DefaultPrng.init(@intCast(time_in_ms()));

    var checkpoint: []const u8 = undefined; // e.g. out/model.bin
    var temperature: f32 = 0.9; // e.g. 1.0 or 0.0
    var steps: u32 = 256; // max number of steps to run for, 0: use seq_len
    var prompt: ?[]const u8 = null; // prompt string

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var argc: u32 = 0;
    var binary: []const u8 = undefined;
    var config: *Config = undefined;
    while (true) {
        const arg_option = args.next();
        if (arg_option) |arg| {
            if (argc == 0) {
                binary = arg;
            }
            if (argc == 1) {
                checkpoint = arg;
            }
            if (argc == 2) {
                // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
                temperature = try std.fmt.parseFloat(f32, arg);
            }
            if (argc == 3) {
                steps = try std.fmt.parseInt(u32, arg, 10);
            }
            if (argc == 4) {
                prompt = arg;
            }
        } else {
            // 'checkpoint' is necessary arg
            if (argc < 2) {
                std.log.err("Usage: {s} <checkpoint_file>", .{binary});
                return Error.InvalidArgs;
            }
            break;
        }
        argc += 1;
    }

    var weights: TransformerWeights = undefined;

    // read in the model.bin file
    var file = try std.fs.cwd().openFile(checkpoint, .{});

    // read in the config header
    const config_struct_size = @sizeOf(Config);
    var buffer = try allocator.alignedAlloc(u8, 4, config_struct_size);
    defer allocator.free(buffer);
    const bytes_read = try file.read(buffer);
    if (bytes_read != config_struct_size) {
        std.log.err("error reading model file", .{});
        return Error.InvalidModelFile;
    }
    config = @ptrCast(buffer);

    // negative vocab size is hacky way of signaling unshared weights.
    const vocab_size_cast: *i32 = @ptrCast(&config.vocab_size);
    const shared_weights = if (vocab_size_cast.* > 0) true else false;

    var abs_cast = try std.math.absInt(vocab_size_cast.*);
    const abs: *u32 = @ptrCast(&abs_cast);
    config.vocab_size = abs.*;

    // figure out the file size
    const file_data = try file.metadata();
    const file_size = file_data.size();

    file.close();

    // memory map the Transformer weights into the data pointer
    const fd = try std.os.open(checkpoint, std.os.O.RDONLY, 0);
    var data_mmap = try std.os.mmap(null, file_size, std.os.linux.PROT.READ, std.os.linux.MAP.PRIVATE, fd, 0);
    const data_ptr: *[]f32 = @ptrCast(&data_mmap);
    const data: []f32 = data_ptr.*;
    const weights_ptr: []f32 = data[@sizeOf(Config) / @sizeOf(f32) ..];
    checkpoint_init_weights(&weights, config, weights_ptr, shared_weights);

    // right now we cannot run for more than config.seq_len steps
    if ((steps <= 0) or (steps > config.seq_len)) {
        steps = config.seq_len;
    }

    // read in the tokenizer.bin file
    const vocab: [][]u8 = try allocator.alloc([]u8, config.vocab_size);

    const vocab_scores: []f32 = try allocator.alloc(f32, config.vocab_size);
    var max_token_length: [1]u32 = .{0};
    var max_token_length_arr: *[4]u8 = @ptrCast(&max_token_length);

    var token_file = try std.fs.cwd().openFile("tokenizer.bin", .{});

    const token_bytes_read = try token_file.read(max_token_length_arr[0..]);
    if (token_bytes_read != 4) {
        return Error.InvalidTokenizerFile;
    }
    var len: u32 = 0;
    var i: u32 = 0;
    while (i < config.vocab_size) : (i += 1) {
        var score_arr = vocab_scores[i .. i + 1];
        var score_arr_cast: *[4]u8 = @ptrCast(score_arr);
        const score_read = try token_file.read(score_arr_cast);
        if (score_read != 4) {
            return Error.InvalidTokenizerFile;
        }
        var len_cast: *[4]u8 = @ptrCast(&len);
        const len_read = try token_file.read(len_cast);
        if (len_read != 4) {
            return Error.InvalidModelFile;
        }
        vocab[i] = try allocator.alloc(u8, len + 1);
        const vocab_read = try token_file.read(vocab[i][0..len]);
        if (vocab_read != len) {
            return Error.InvalidModelFile;
        }
        vocab[i][len] = '\x00';
    }

    token_file.close();

    // create and init the application RunState
    var state = try RunState.init(allocator, config);

    // process the prompt, if any
    var prompt_tokens_option: ?[]u32 = null;
    var num_prompt_tokens: u32 = 0;
    if (prompt) |p| {
        prompt_tokens_option = try allocator.alloc(u32, config.seq_len);
        if (prompt_tokens_option) |prompt_tokens| {
            try bpe_encode(allocator, p, vocab, vocab_scores, config.vocab_size, max_token_length[0], prompt_tokens, &num_prompt_tokens);
        }
    }

    // start the main loop
    var start: i64 = 0; // used to time our code, only initiated after first iteration
    var next: u32 = 0; // will store the next token in the sequence
    var token: u32 = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var pos: u32 = 0; // position in the sequence
    std.log.info("<s>", .{});
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, config, &state, &weights);

        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            if (prompt_tokens_option) |prompt_tokens| {
                next = prompt_tokens[pos];
            }
        } else {
            // sample the next token
            if (temperature == 0.0) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            } else {
                // apply the temperature to the logits
                var q: u32 = 0;
                while (q < config.vocab_size) : (q += 1) {
                    state.logits[q] /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size);
                // we sample from this distribution to get the next token
                next = sample(&rng, state.logits, config.vocab_size);
            }
        }

        // following BOS token (1), sentencepiece decoder strips any leading whitespace
        const token_str = if (token == 1 and vocab[next][0] == ' ') vocab[next][1..] else vocab[next];
        std.debug.print("{s}", .{token_str});

        // advance forward
        token = next;
        pos += 1;
        // init our timer here because the first iteration is slow due to memmap
        if (start == 0) {
            start = time_in_ms();
        }
    }

    // report achieved tok/s
    var end: i64 = time_in_ms();
    const step_cast = @as(i64, steps - 1);
    const tokps: i64 = @divFloor(step_cast * 1000, end - start);
    std.debug.print("\nachieved tok/s: {}\n", .{tokps});

    // memory and file handles cleanup
    state.deinit();
    i = 0;
    while (i < config.vocab_size) : (i += 1) {
        allocator.free(vocab[i]);
    }
    allocator.free(vocab);
    allocator.free(vocab_scores);
    std.os.munmap(data_mmap);
    if (fd != -1) {
        std.os.close(fd);
    }
}
