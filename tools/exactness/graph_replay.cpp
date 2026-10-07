#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto gpu = ggml_backend_dev_init(ggml_backend_dev_by_name("CUDA0"), nullptr);
    auto cpu = ggml_backend_dev_init(ggml_backend_dev_by_name("CPU"), nullptr);
    if (!gpu || !cpu) { return 2; }
    for (int columns : {8, 65, 512}) {
      for (const auto layout : {std::pair{0, 0}, std::pair{4, 0}, std::pair{4, 4}}) {
        const int padding = layout.first, input_offset = layout.second;
        const int k = 2560, rows = 256;
        auto * ctx = ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true});
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_64, k, rows);
        auto * owner = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k + padding, columns);
        auto * b = ggml_view_2d(ctx, owner, k, columns, owner->nb[1], input_offset);
        auto * out = ggml_mul_mat(ctx, a, b);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        if (!buffer) { return 2; }
        std::vector<float> weights(k*rows);
        for (size_t i = 0; i < weights.size(); ++i) { weights[i] = std::sin(float(i)*0.031f); }
        std::vector<uint8_t> packed(ggml_nbytes(a));
        ggml_quantize_chunk(GGML_TYPE_Q4_64, weights.data(), packed.data(), 0, rows, k, nullptr);
        ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
        auto copy = ggml_backend_graph_copy(cpu, graph);
        if (!copy.buffer) { return 2; }
        auto * reference = ggml_graph_node(copy.graph, -1);
        std::vector<float> values((k + padding)*columns), actual(rows*columns), expected(rows*columns), previous;
        for (int run = 0; run < 5; ++run) {
            for (size_t i = 0; i < values.size(); ++i) { values[i] = std::cos(float(i)*0.007f + run*0.43f); }
            ggml_backend_tensor_set(owner, values.data(), 0, values.size()*sizeof(float));
            ggml_backend_tensor_set(reference->src[1]->view_src, values.data(), 0, values.size()*sizeof(float));
            if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS ||
                ggml_backend_graph_compute(cpu, copy.graph) != GGML_STATUS_SUCCESS) { return 2; }
            ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
            ggml_backend_tensor_get(reference, expected.data(), 0, expected.size()*sizeof(float));
            double error = 0, norm = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) { return 1; }
                error += std::pow(double(actual[i]) - expected[i], 2);
                norm += double(expected[i])*expected[i];
            }
            const double nmse = error / std::max(norm, 1e-30);
            std::cout << "padding=" << padding << " input_offset=" << input_offset << " columns=" << columns << " run=" << run << " nmse=" << nmse << '\n';
            if (nmse > 5e-4 || (!previous.empty() && actual == previous)) { return 1; }
            previous = actual;
        }
        ggml_backend_graph_copy_free(copy);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
      }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return 0;
}
