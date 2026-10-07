#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void decode(llama_context * ctx, const llama_token * tokens, int count, int position, bool all_logits) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_logits || i == count - 1;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (result != 0) {
        throw std::runtime_error("decode failed: " + std::to_string(result));
    }
}

static llama_context * context(llama_model * model, int snapshots) {
    auto params = llama_context_default_params();
    params.n_ctx = 4096;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_rs_seq = snapshots;
    params.n_threads = 8;
    params.n_threads_batch = 8;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * result = llama_init_from_model(model, params);
    if (!result) {
        throw std::runtime_error("context allocation failed");
    }
    return result;
}

static void prefill(llama_context * ctx, const std::vector<llama_token> & prompt) {
    for (size_t start = 0; start < prompt.size(); start += 512) {
        const int count = std::min<size_t>(512, prompt.size() - start);
        decode(ctx, prompt.data() + start, count, start, false);
    }
}

static int argmax(const float * values, int n) {
    if (std::any_of(values, values + n, [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("nonfinite logits");
    }
    return std::max_element(values, values + n) - values;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: logit-parity MODEL PROMPT_IDS OUT_CSV SNAPSHOTS\n";
        return 2;
    }
    const int snapshots = std::stoi(argv[4]);
    if (snapshots < 0 || snapshots > 32) {
        std::cerr << "snapshots must be in 0..32\n";
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        return 2;
    }
    std::ifstream input(argv[2]);
    std::vector<llama_token> prompt;
    int token;
    while (input >> token) {
        prompt.push_back(token);
    }
    if (prompt.empty() || prompt.size() > 2048) {
        std::cerr << "prompt must contain 1..2048 token ids\n";
        return 2;
    }
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (std::any_of(prompt.begin(), prompt.end(), [vocabulary](llama_token id) { return id < 0 || id >= vocabulary; })) {
        std::cerr << "prompt token outside vocabulary\n";
        return 2;
    }
    std::vector<llama_token> continuation;
    std::vector<std::vector<float>> reference;
    llama_context * ctx = context(model, snapshots);
    prefill(ctx, prompt);
    for (int i = 0; i <= 130; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        reference.emplace_back(logits, logits + vocabulary);
        if (i < 130) {
            continuation.push_back(argmax(logits, vocabulary));
            decode(ctx, &continuation.back(), 1, prompt.size() + i, true);
        }
    }
    llama_free(ctx);
    std::ofstream output(argv[3]);
    if (!output) {
        throw std::runtime_error("cannot open output CSV");
    }
    output << "columns,bit_differences,argmax_differences,max_absolute_error,rollback,rollback_bit_differences,rollback_argmax_difference\n";
    bool exact = true;
    for (int width : {1, 2, 4, 5, 8, 9, 16, 32, 33, 64, 65}) {
        ctx = context(model, snapshots);
        prefill(ctx, prompt);
        int64_t bits = 0;
        int choices = 0;
        float maximum = 0;
        int rollback = 0;
        int64_t rollback_bits = 0;
        int rollback_choice = 0;
        auto compare = [&](const float * logits, const std::vector<float> & expected) {
            for (int j = 0; j < vocabulary; ++j) {
                bits += std::memcmp(logits + j, expected.data() + j, sizeof(float)) != 0;
                maximum = std::max(maximum, std::abs(logits[j] - expected[j]));
            }
            choices += argmax(logits, vocabulary) != argmax(expected.data(), vocabulary);
        };
        compare(llama_get_logits_ith(ctx, -1), reference[0]);
        for (int start = 0; start < 130; start += width) {
            const int count = std::min(width, 130 - start);
            decode(ctx, continuation.data() + start, count, prompt.size() + start, true);
            for (int i = 0; i < count; ++i) {
                compare(llama_get_logits_ith(ctx, i), reference[start + i + 1]);
            }
            const int remove = std::min(snapshots, count - 1);
            if (remove > 0) {
                rollback = std::max(rollback, remove);
                const int position = start + count - remove;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, prompt.size() + position, -1)) {
                    throw std::runtime_error("rollback refused");
                }
                decode(ctx, continuation.data() + position, 1, prompt.size() + position, true);
                const float * logits = llama_get_logits_ith(ctx, -1);
                for (int j = 0; j < vocabulary; ++j) {
                    rollback_bits += std::memcmp(logits + j, reference[position + 1].data() + j, sizeof(float)) != 0;
                }
                rollback_choice += argmax(logits, vocabulary) != argmax(reference[position + 1].data(), vocabulary);
                if (remove > 1) {
                    decode(ctx, continuation.data() + position + 1, remove - 1, prompt.size() + position + 1, false);
                }
                compare(llama_get_logits_ith(ctx, -1), reference[start + count]);
            }
        }
        output << width << ',' << bits << ',' << choices << ',' << maximum << ',' << rollback << ',' << rollback_bits << ',' << rollback_choice << '\n';
        output.flush();
        if (!output) {
            throw std::runtime_error("cannot write output CSV");
        }
        std::cout << "width=" << width << " bits=" << bits << " argmax=" << choices << " max_error=" << maximum
                  << " rollback_bits=" << rollback_bits << '\n';
        if (width <= 64) {
            exact &= bits == 0 && rollback_bits == 0;
        }
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return exact ? 0 : 1;
}
