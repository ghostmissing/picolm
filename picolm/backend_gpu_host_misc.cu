// backend_gpu_host_misc.cu - KV cache, pipeline, attention, rmsnorm, rope, residual, misc
#include "backend_gpu_kernels.cuh"

/* PICOLM_GPU_PROF hooks, defined at the bottom of this file. */
extern "C" int  picolm_gpu_prof_slot(const char *name);
extern "C" void picolm_gpu_prof_begin(int device);
extern "C" void picolm_gpu_prof_end(int slot);

/* Profile one prefill-side helper call. GPU_PROF had slots for the matmuls and
 * the decode attention only, so prefill attention, the KV store, rmsnorm,
 * silu_mul and residual_add between them accounted for none of the ~730 ms of
 * layer-loop wall time that the kernel sum could not explain. */
#define PF_PROF(name, dev, call)                                          \
    do {                                                                  \
        int _pfps = picolm_gpu_prof_slot(name);                           \
        picolm_gpu_prof_begin(dev);                                       \
        int _pfrc = (call);                                               \
        picolm_gpu_prof_end(_pfps);                                       \
        return _pfrc;                                                     \
    } while (0)

extern "C" void picolm_gpu_prof_begin(int device);
extern "C" void picolm_gpu_prof_end(int slot);

extern "C" {
__global__ void picolm_gpu_kv_pack_store_kernel(uint16_t *dst_row, const float *src, int n);
}

extern "C" int
picolm_gpu_kv_store_dev_batched_strided(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                         const float *src_dev, int n_kv_heads, int head_dim,
                                         int max_seq_len, int device, int src_stride);


/* Print once per kernel type when first dispatched */
static void gpu_dispatch_print(const char *name) {
    static char seen_names[64][64] = {0};
    for (int i = 0; i < 64; i++) {
        if (seen_names[i][0] == '\0') {
            snprintf(seen_names[i], sizeof(seen_names[i]), "%s", name);
            fprintf(stderr, "[GPU] kernel: %s\n", name);
            return;
        }
        if (strcmp(seen_names[i], name) == 0) return;
    }
}

/* Static globals (were file-static in monolithic backend_gpu.cu) */
static uint16_t *g_kv_k_dev[PICOLM_GPU_MAX_DEVICES];
static uint16_t *g_kv_v_dev[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_k_cap[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_v_cap[PICOLM_GPU_MAX_DEVICES];
static float *g_verify_buf_dev[PICOLM_GPU_MAX_DEVICES];

/* How the device KV cache rows are laid out. Default F16 (0) so that a caller
 * which never calls picolm_gpu_kv_set_type() gets exactly the historical
 * behaviour. g_kv_types_set[dev] is 0 until set_type() runs, which is what
 * lets the store path tell "defaulted to F16" from "explicitly F16". */
static int g_kv_type_k[PICOLM_GPU_MAX_DEVICES] = {0};
static int g_kv_type_v[PICOLM_GPU_MAX_DEVICES] = {0};
static size_t g_kv_head_stride_k[PICOLM_GPU_MAX_DEVICES] = {0};
static size_t g_kv_head_stride_v[PICOLM_GPU_MAX_DEVICES] = {0};
static int g_kv_types_set[PICOLM_GPU_MAX_DEVICES] = {0};

extern "C" int
picolm_gpu_kv_set_type(int kv_type_k, int kv_type_v,
                        size_t kv_head_stride_k, size_t kv_head_stride_v,
                        int device) {
    if (device < 0 || device >= PICOLM_GPU_MAX_DEVICES) return 0;
    /* Only F16, Q8_0 and Q4_0 have a device dequant path (kv_block_bytes() in
     * backend_gpu_dequant.cuh). TQ3/TQ4 stay CPU-only: refuse rather than let
     * the attention kernels read a format they cannot decode. */
    if (kv_type_k < 0 || kv_type_k > 2) return 0;
    if (kv_type_v < 0 || kv_type_v > 2) return 0;
    g_kv_type_k[device] = kv_type_k;
    g_kv_type_v[device] = kv_type_v;
    g_kv_head_stride_k[device] = kv_head_stride_k;
    g_kv_head_stride_v[device] = kv_head_stride_v;
    g_kv_types_set[device] = 1;
    return 1;
}
extern "C" int
picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    /* Align to 256 bytes for coalesced access */
    kv_k_bytes = (kv_k_bytes + 255) & ~(size_t)255;
    kv_v_bytes = (kv_v_bytes + 255) & ~(size_t)255;

    if (g_kv_k_cap[device] < kv_k_bytes) {
        if (g_kv_k_dev[device]) gpuFree(g_kv_k_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_k_dev[device], kv_k_bytes), "KV K allocation")) return 0;
        /* Zero-initialize */
        gpuMemset(g_kv_k_dev[device], 0, kv_k_bytes);
        g_kv_k_cap[device] = kv_k_bytes;
    }
    if (g_kv_v_cap[device] < kv_v_bytes) {
        if (g_kv_v_dev[device]) gpuFree(g_kv_v_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_v_dev[device], kv_v_bytes), "KV V allocation")) return 0;
        gpuMemset(g_kv_v_dev[device], 0, kv_v_bytes);
        g_kv_v_cap[device] = kv_v_bytes;
    }

    return 1;
}

extern "C" void
picolm_gpu_kv_cache_clear(int device) {
    if (device < 0 || device >= PICOLM_GPU_MAX_DEVICES) return;
    if (g_kv_k_dev[device] && g_kv_k_cap[device])
        gpuMemset(g_kv_k_dev[device], 0, g_kv_k_cap[device]);
    if (g_kv_v_dev[device] && g_kv_v_cap[device])
        gpuMemset(g_kv_v_dev[device], 0, g_kv_v_cap[device]);
}

extern "C" void
picolm_gpu_kv_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        if (g_kv_k_dev[i]) { gpuFree(g_kv_k_dev[i]); g_kv_k_dev[i] = NULL; g_kv_k_cap[i] = 0; }
        if (g_kv_v_dev[i]) { gpuFree(g_kv_v_dev[i]); g_kv_v_dev[i] = NULL; g_kv_v_cap[i] = 0; }
    }
}

/* Phase 2: allocate the fixed-size device-resident pipeline buffers for
 * model_forward_gpu() decode (S=1 only). Called once at model load,
 * right after picolm_gpu_kv_alloc(), with sizes derived from model
 * config. Idempotent: safe to call again with the same sizes (no-op if
 * ctx->pipe_ready already set for this device -- these buffers never
 * need to grow, unlike reserve()-based scratch, since decode is always
 * S=1). Returns 1 on success. */
extern "C" int
picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                           int vocab_size, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_ready) return 1;

    size_t db = (size_t)dim * sizeof(float);
    size_t qb = (size_t)q_dim * sizeof(float);
    size_t kvb = (size_t)kv_dim * sizeof(float);
    size_t fb = (size_t)ffn_hidden * sizeof(float);
    size_t vb = (size_t)vocab_size * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x, db), "pipe_x alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb, db), "pipe_xb alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q, qb), "pipe_q alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k, kvb), "pipe_k alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v, kvb), "pipe_v alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out, qb), "pipe_attn_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm, db), "pipe_ffn_norm alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate, fb), "pipe_gate alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up, fb), "pipe_up alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_logits, vb), "pipe_logits alloc");
    if (!ok) return 0;

    ctx->pipe_ready = 1;
    return 1;
}

/* Allocate SSM pipeline buffers for hybrid SSM+attention layers.
 * Called from model_load after SSM eligibility passes. */
extern "C" int
picolm_gpu_ssm_pipeline_alloc(int conv_dim, int ssm_d_inner, int n_v_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->ssm_ready) return 1;

    size_t conv_b = (size_t)conv_dim * sizeof(float);
    size_t inner_b = (size_t)ssm_d_inner * sizeof(float);
    size_t heads_b = (size_t)n_v_heads * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_qkv_raw, conv_b), "ssm_qkv_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_conv_out, conv_b), "ssm_conv_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2, inner_b), "ssm_xb2 alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2_remap, inner_b), "ssm_xb2_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_v_remap, inner_b), "ssm_v_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_alpha_raw, heads_b), "ssm_alpha_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta_raw, heads_b), "ssm_beta_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_gate_exp, heads_b), "ssm_gate_exp alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta, heads_b), "ssm_beta alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_output, inner_b), "ssm_output alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_final_output, inner_b), "ssm_final_output alloc");
    if (!ok) return 0;

    ctx->ssm_ready = 1;
    return 1;
}

/* Allocate prefill batch buffers: [max_ubatch][max_stride] for S>1 pipeline
 * xb_stride = max(q_dim, conv_dim, dim) to accommodate all uses:
 * - attention: RMSNorm output + QKV projections (q_dim stride)
 * - SSM prefill: attn_qkv output (conv_dim = 2*d_state*n_k + ssm_d_inner)
 * - residual add: dim stride
 *
 * max_ubatch is the largest number of tokens a single prefill launch will
 * process (picolm_prefill_ubatch()), NOT the context length. See model.h. */
extern "C" int
picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                 int xb_stride, int max_ubatch, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_b_ready) return 1;

    size_t bsz = (size_t)max_ubatch;
    size_t qb = bsz * q_dim * sizeof(float);
    size_t xb = bsz * xb_stride * sizeof(float);
    size_t kvb = bsz * kv_dim * sizeof(float);
    size_t fb = bsz * ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x_b, xb), "pipe_x_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb_b, xb), "pipe_xb_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q_b, qb), "pipe_q_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k_b, kvb), "pipe_k_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v_b, kvb), "pipe_v_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out_b, qb), "pipe_attn_out_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm_b, xb), "pipe_ffn_norm_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate_b, fb), "pipe_gate_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up_b, fb), "pipe_up_b alloc");
    ok &= gpu_ok(gpuMalloc(&g_verify_buf_dev[ctx->device], 64 * sizeof(float)), "verify_buf alloc");
    if (!ok) return 0;

    ctx->pipe_b_ready = 1;
    return 1;
}

extern "C" void
picolm_gpu_pipeline_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_ready) continue;
        if (ctx->pipe_x) gpuFree(ctx->pipe_x);
        if (ctx->pipe_xb) gpuFree(ctx->pipe_xb);
        if (ctx->pipe_q) gpuFree(ctx->pipe_q);
        if (ctx->pipe_k) gpuFree(ctx->pipe_k);
        if (ctx->pipe_v) gpuFree(ctx->pipe_v);
        if (ctx->pipe_attn_out) gpuFree(ctx->pipe_attn_out);
        if (ctx->pipe_ffn_norm) gpuFree(ctx->pipe_ffn_norm);
        if (ctx->pipe_gate) gpuFree(ctx->pipe_gate);
        if (ctx->pipe_up) gpuFree(ctx->pipe_up);
        if (ctx->pipe_logits) gpuFree(ctx->pipe_logits);
        ctx->pipe_x = ctx->pipe_xb = ctx->pipe_q = ctx->pipe_k = ctx->pipe_v =
            ctx->pipe_attn_out = ctx->pipe_ffn_norm = ctx->pipe_gate = ctx->pipe_up =
            ctx->pipe_logits = NULL;
        ctx->pipe_ready = 0;
    }
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_b_ready) continue;
        if (ctx->pipe_x_b) gpuFree(ctx->pipe_x_b);
        if (ctx->pipe_xb_b) gpuFree(ctx->pipe_xb_b);
        if (ctx->pipe_q_b) gpuFree(ctx->pipe_q_b);
        if (ctx->pipe_k_b) gpuFree(ctx->pipe_k_b);
        if (ctx->pipe_v_b) gpuFree(ctx->pipe_v_b);
        if (ctx->pipe_attn_out_b) gpuFree(ctx->pipe_attn_out_b);
        if (ctx->pipe_ffn_norm_b) gpuFree(ctx->pipe_ffn_norm_b);
        if (ctx->pipe_gate_b) gpuFree(ctx->pipe_gate_b);
        if (ctx->pipe_up_b) gpuFree(ctx->pipe_up_b);
        if (g_verify_buf_dev[ctx->device]) { gpuFree(g_verify_buf_dev[ctx->device]); g_verify_buf_dev[ctx->device] = NULL; }
        ctx->pipe_x_b = ctx->pipe_xb_b = ctx->pipe_q_b = ctx->pipe_k_b = ctx->pipe_v_b =
            ctx->pipe_attn_out_b = ctx->pipe_ffn_norm_b = ctx->pipe_gate_b = ctx->pipe_up_b = NULL;
        ctx->pipe_b_ready = 0;
    }
    /* SSM pipeline buffers */
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->ssm_ready) continue;
        if (ctx->ssm_qkv_raw) gpuFree(ctx->ssm_qkv_raw);
        if (ctx->ssm_conv_out) gpuFree(ctx->ssm_conv_out);
        if (ctx->ssm_xb2) gpuFree(ctx->ssm_xb2);
        if (ctx->ssm_xb2_remap) gpuFree(ctx->ssm_xb2_remap);
        if (ctx->ssm_v_remap) gpuFree(ctx->ssm_v_remap);
        if (ctx->ssm_alpha_raw) gpuFree(ctx->ssm_alpha_raw);
        if (ctx->ssm_beta_raw) gpuFree(ctx->ssm_beta_raw);
        if (ctx->ssm_gate_exp) gpuFree(ctx->ssm_gate_exp);
        if (ctx->ssm_beta) gpuFree(ctx->ssm_beta);
        if (ctx->ssm_output) gpuFree(ctx->ssm_output);
        if (ctx->ssm_final_output) gpuFree(ctx->ssm_final_output);
        ctx->ssm_qkv_raw = ctx->ssm_conv_out = ctx->ssm_xb2 = ctx->ssm_xb2_remap =
            ctx->ssm_v_remap = ctx->ssm_alpha_raw = ctx->ssm_beta_raw = ctx->ssm_gate_exp =
            ctx->ssm_beta = ctx->ssm_output = ctx->ssm_final_output = NULL;
        ctx->ssm_ready = 0;
    }
}

/* Pre-allocate Q8_0 scratch buffers to a fixed maximum size.
 * This eliminates runtime reallocation races where cudaFree of a
 * scratch buffer races with a kernel still reading from it. */
extern "C" int
picolm_gpu_prealloc_q8(size_t max_xq_bytes, size_t max_xd_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!ctx->q8_xq) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xq, max_xq_bytes), "prealloc q8_xq")) return 0;
        ctx->q8_xq_cap = max_xq_bytes;
    }
    if (!ctx->q8_xd) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xd, max_xd_bytes), "prealloc q8_xd")) return 0;
        ctx->q8_xd_cap = max_xd_bytes;
    }
    return 1;
}

/* Device pointer accessors, so model.c doesn't need gpu_device_ctx_t's
 * internal layout (it's file-static to backend_gpu.cu). */
extern "C" float *picolm_gpu_pipe_x(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x : NULL; }
extern "C" float *picolm_gpu_pipe_xb(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb : NULL; }
extern "C" float *picolm_gpu_pipe_q(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q : NULL; }
extern "C" float *picolm_gpu_pipe_k(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k : NULL; }
extern "C" float *picolm_gpu_pipe_v(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm : NULL; }
extern "C" float *picolm_gpu_pipe_gate(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate : NULL; }
extern "C" float *picolm_gpu_pipe_up(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up : NULL; }
extern "C" float *picolm_gpu_pipe_logits(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_logits : NULL; }

extern "C" float *picolm_gpu_pipe_x_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x_b : NULL; }
extern "C" float *picolm_gpu_pipe_xb_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb_b : NULL; }
extern "C" float *picolm_gpu_pipe_q_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q_b : NULL; }
extern "C" float *picolm_gpu_pipe_k_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k_b : NULL; }
extern "C" float *picolm_gpu_pipe_v_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v_b : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out_b : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm_b : NULL; }
extern "C" float *picolm_gpu_pipe_gate_b(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate_b : NULL; }
extern "C" float *picolm_gpu_pipe_up_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up_b : NULL; }

/* Pinned host staging buffer accessor. Returns a page-locked host pointer
 * that is safe for async H2D copies via gpuMemcpyAsync. Auto-grows to
 * accommodate the requested byte count. Returns NULL on OOM. */
extern "C" float *picolm_gpu_staging_host(int device, size_t bytes) {
    gpu_device_ctx_t *c = find_ctx(device);
    if (!c) return NULL;
    if (!reserve_pinned(&c->host_x, &c->host_x_cap, bytes)) return NULL;
    return c->host_x;
}

extern "C" float *picolm_gpu_verify_buf(int device)        { return g_verify_buf_dev[device]; }

/* SSM pipeline buffer accessors */
extern "C" float *picolm_gpu_ssm_qkv_raw(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_qkv_raw : NULL; }
extern "C" float *picolm_gpu_ssm_conv_out(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_conv_out : NULL; }
extern "C" float *picolm_gpu_ssm_xb2(int device)           { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2 : NULL; }
extern "C" float *picolm_gpu_ssm_xb2_remap(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2_remap : NULL; }
extern "C" float *picolm_gpu_ssm_v_remap(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_v_remap : NULL; }
extern "C" float *picolm_gpu_ssm_alpha_raw(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_alpha_raw : NULL; }
extern "C" float *picolm_gpu_ssm_beta_raw(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta_raw : NULL; }
extern "C" float *picolm_gpu_ssm_gate_exp(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_gate_exp : NULL; }
extern "C" float *picolm_gpu_ssm_beta(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta : NULL; }
extern "C" float *picolm_gpu_ssm_output(int device)        { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_output : NULL; }
extern "C" float *picolm_gpu_ssm_final_output(int device)  { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_final_output : NULL; }
extern "C" uint16_t *picolm_gpu_kv_k_dev(int device)       { return g_kv_k_dev[device]; }
extern "C" uint16_t *picolm_gpu_kv_v_dev(int device)       { return g_kv_v_dev[device]; }

/* Bit-exact device port of the host fp32_to_fp16() in quant.c -- NOT
 * CUDA's __float2half (different rounding/tie behavior in edge cases),
 * so that PICOLM_DBG_PIPELINE logit comparisons against the CPU path
 * stay meaningful all the way through the KV cache, not just up to
 * whatever tolerance a different rounding rule would introduce. */
__device__ __forceinline__ uint16_t
picolm_gpu_fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) return (uint16_t)sign;
    if (((bits >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }
    mant += 0x00001000;
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Strided KV pack+store kernel: src has tokens packed at src_stride (not kv_dim).
 * Used by GPU prefill ubatch loop where bk/bv have xb_stride layout. */
__global__ void
picolm_gpu_kv_pack_store_strided_kernel(uint16_t *dst_row, const float *src,
    int kv_dim, int n_tokens, int src_stride) {
    for (int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
         i < n_tokens * kv_dim;
         i += (int)gridDim.x * gpuBlockDim_x) {
        int t = i / kv_dim;
        int d = i % kv_dim;
        dst_row[i] = picolm_gpu_fp32_to_fp16(src[(size_t)t * src_stride + d]);
    }
}

/* Pack a device-resident F32 [n_kv_heads*head_dim] vector to F16 and
 * write it directly into the device KV cache row for (layer_ordinal,
 * pos). One thread per element, grid-stride, no shared memory. This
 * replaces the D2H -> CPU convert -> H2D round trip that
 * model_forward_gpu() otherwise needs every layer of every token: with
 * this kernel the KV store never touches the host on the hot path. */
__global__ void
picolm_gpu_kv_pack_store_kernel(uint16_t *dst_row, const float *src, int n) {
    for (int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x; i < n;
         i += (int)gridDim.x * gpuBlockDim_x) {
        dst_row[i] = picolm_gpu_fp32_to_fp16(src[i]);
    }
}

/* ---- Quantized KV store ----
 *
 * Writes the SAME bytes the CPU's quantize_row_q8_0 / quantize_row_q4_0 produce
 * for the same inputs, so a GPU-quantized KV cache and a CPU-quantized one are
 * interchangeable (and picolm_gpu_kv_flush_to_cpu is a plain memcpy).
 *
 * Layout per position: [n_kv_heads][head_dim] values, quantized in place as one
 * contiguous run of blocks -- exactly what quantize_row_*(k_pos, kp, kv_dim)
 * does on the CPU, because a block never straddles a head when head_dim is a
 * multiple of 32 (the host gate guarantees this when it enables quantized KV).
 *
 * One warp per 32-value group: the warp is the natural scope for the absmax
 * reduction, so it goes through the gpuShflXorSync shim (defined for CUDA and
 * HIP alike) and the kernel needs no shared memory and no block-wide barrier.
 *
 * kv_type: 1 = Q8_0, 2 = Q4_0.
 */
__global__ void
picolm_gpu_kv_quant_store_kernel(uint8_t *dst_row, const float *src,
                                  int n_values, int kv_type) {
    /* Assign exactly one 32-value block to each warp. No shared memory and no
     * block-wide barriers: the absmax is exchanged with an XOR butterfly over
     * the warp, and there is no way for a partial warp to reach a barrier the
     * rest of the block does not (an earlier version early-returned
     * out-of-range groups before a gpuSyncthreads(), which is undefined
     * behaviour and corrupted the Q4_0 store). */
    int tid = (int)gpuThreadIdx_x;
    int lane = tid % 32;
    int warp_in_block = tid / 32;
    int warps_per_block = (int)gpuBlockDim_x / 32;
    int blk = (int)gpuBlockIdx_x * warps_per_block + warp_in_block;
    int nblk = n_values / 32;
    if (blk >= nblk) return;

    float x = src[(size_t)blk * 32 + lane];

    /* Butterfly absmax across the 32 lanes of this warp. Every lane ends with
     * the same value, so no lane needs to read another lane's scratch. */
    float amax = fabsf(x);
    for (int off = 16; off > 0; off >>= 1) {
        float o = gpuShflXorSync(0xFFFFFFFFu, amax, off);
        if (o > amax) amax = o;
    }

    if (kv_type == 1) { /* Q8_0: 34 bytes = fp16 d + 32 int8 */
        uint8_t *b = dst_row + (size_t)blk * 34;
        float d = amax / 127.0f;
        float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        int q = (int)nearbyintf(x * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        if (lane == 0) {
            uint16_t dh = gpu_fp32_to_fp16(d);
            b[0] = (uint8_t)(dh & 0xFF);
            b[1] = (uint8_t)(dh >> 8);
        }
        b[2 + lane] = (uint8_t)(int8_t)q;
    } else { /* Q4_0: 18 bytes = fp16 d + 16 packed nibbles */
        uint8_t *b = dst_row + (size_t)blk * 18;
        float d = amax / 8.0f;
        float id = (amax != 0.0f) ? 8.0f / amax : 0.0f;
        /* quantize_row_q4_0: v = (uint8_t)(x*id + 8.5f), clamped to 15. */
        int v = (int)(x * id + 8.5f);
        if (v > 15) v = 15;
        if (v < 0) v = 0;
        if (lane == 0) {
            uint16_t dh = gpu_fp32_to_fp16(d);
            b[0] = (uint8_t)(dh & 0xFF);
            b[1] = (uint8_t)(dh >> 8);
        }
        /* Lanes 0..15 own byte lane (low nibble = element lane), lanes
         * 16..31 own the high nibble of byte lane-16 (= element lane).
         * Each byte is written entirely by its low-nibble lane, which
         * reads the partner's value with a shuffle instead of racing on a
         * read-modify-write. */
        int hi_v = (int)gpuShflXorSync(0xFFFFFFFFu, (float)v, 16); /* lane<16 gets lane+16 */
        if (lane < 16)
            b[2 + lane] = (uint8_t)((v & 0x0F) | (((int)hi_v & 0x0F) << 4));
    }
}

/* Device-native KV store: src_dev is pipe_k/pipe_v (F32, already on
 * device, kv_dim = n_kv_heads*head_dim elements). No transfer, no sync --
 * same ctx->stream ordering argument as the other _dev primitives.
 * Caller must call this (K then V) before picolm_gpu_attention_decode_dev
 * for the same (layer_ordinal, pos), same as picolm_gpu_kv_store_rows.
 *
 * Delegates to the batched strided implementation with n_positions=1 so the
 * decode path and the prefill path CANNOT disagree about the row stride or
 * the storage format. They used to: this function wrote F16 at
 * kv_dim*sizeof(uint16_t) per row unconditionally, so with a Q8_0/Q4_0 cache
 * every decode step scribbled 2 bytes/element into a cache sized for 34/18
 * bytes per 32 values, corrupting every subsequent read. */
extern "C" int
picolm_gpu_kv_store_dev(int is_k, int layer_ordinal, int pos,
                         const float *src_dev, int n_kv_heads, int head_dim,
                         int max_seq_len, int device) {
    return picolm_gpu_kv_store_dev_batched_strided(is_k, layer_ordinal, pos, 1,
                                                   src_dev, n_kv_heads, head_dim,
                                                   max_seq_len, device, 0);
}

/* Batched variant: src_dev is [S][kv_dim] contiguous F32, positions
 * start_pos..start_pos+S-1. Reuses picolm_gpu_kv_pack_store_kernel --
 * since the device KV cache has no per-row padding, S contiguous source
 * rows map to S contiguous destination rows. One launch for the whole
 * prefill chunk. */
extern "C" int
picolm_gpu_kv_store_dev_batched(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                 const float *src_dev, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device) {
    return picolm_gpu_kv_store_dev_batched_strided(is_k, layer_ordinal, start_pos,
        n_positions, src_dev, n_kv_heads, head_dim, max_seq_len, device, 0);
}

extern "C" int
picolm_gpu_kv_store_dev_batched_strided(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                         const float *src_dev, int n_kv_heads, int head_dim,
                                         int max_seq_len, int device, int src_stride) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    int kv_dim = n_kv_heads * head_dim;
    int kv_type = is_k ? g_kv_type_k[device] : g_kv_type_v[device];
    /* Row size in bytes depends on the storage format: F16 is 2 bytes per
     * element, Q8_0 ~1.06, Q4_0 ~0.56. The CPU row size (kv_row_size_*) is
     * exactly this, and the device cache was allocated from it. */
    size_t row_bytes;
    if (g_kv_types_set[device] && kv_type != 0) {
        int block_bytes = (kv_type == 1) ? 34 : 18;
        row_bytes = (size_t)(kv_dim / 32) * block_bytes;
    } else {
        row_bytes = (size_t)kv_dim * sizeof(uint16_t);
    }

    uint8_t *dst_base = (uint8_t *)(is_k ? g_kv_k_dev[device] : g_kv_v_dev[device]);
    uint8_t *dst_row = dst_base + (size_t)layer_ordinal * max_seq_len * row_bytes
                                 + (size_t)start_pos * row_bytes;

    if (kv_type != 0) {
        /* Quantized: the kernel works on 32-value groups and needs one thread
         * per value, so run one position at a time and give each launch
         * (kv_dim/32) groups of 32 lanes. src_stride is 0 when the source rows
         * are already packed at kv_dim. */
        if ((kv_dim % 32) != 0) return 0;   /* host gate should prevent this */
        int n_threads = 128;                 /* 4 warps = 4 blocks per launch */
        int groups = kv_dim / 32;
        int n_blocks = (groups + 3) / 4;
        for (int t = 0; t < n_positions; t++) {
            const float *src_pos = src_dev + (size_t)t * (src_stride > 0 ? src_stride : kv_dim);
            uint8_t *dst_pos = dst_row + (size_t)t * row_bytes;
            picolm_gpu_kv_quant_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
                dst_pos, src_pos, kv_dim, kv_type);
        }
        if (!gpu_ok(gpuGetLastError(), "kv quant store batched (dev)")) return 0;
        return 1;
    }

    uint16_t *dst_row16 = (uint16_t *)dst_row;
    int total = n_positions * kv_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    if (src_stride > 0 && src_stride != kv_dim) {
        picolm_gpu_kv_pack_store_strided_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
            dst_row16, src_dev, kv_dim, n_positions, src_stride);
    } else {
        picolm_gpu_kv_pack_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
            dst_row16, src_dev, total);
    }
    if (!gpu_ok(gpuGetLastError(), "kv pack+store batched (dev)")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_kv_debug_dump(int is_k, int layer_ordinal, int pos,
                          uint16_t *dst, int n_elements, int n_kv_heads,
                          int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device]) return 0;
    /* Only meaningful for an F16 cache: the element-indexed layout below is
     * the F16 one. A quantized cache has no uint16 element addressing. */
    if (g_kv_types_set[device] && (g_kv_type_k[device] != 0 || g_kv_type_v[device] != 0))
        return 0;
    uint16_t *src_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    /* Was `* kv_dim`, which is only the row size for F16 with head_dim
     * elements per head -- i.e. it silently assumed the row stride equals the
     * element count. Use the explicit row stride so this stays correct. */
    size_t row_stride = (size_t)n_kv_heads * head_dim;
    uint16_t *src_row = src_base + (size_t)layer_ordinal * max_seq_len * row_stride
                                  + (size_t)pos * row_stride;
    if (!gpu_ok(gpuMemcpy(dst, src_row, n_elements * sizeof(uint16_t), gpuMemcpyDeviceToHost),
                "kv_debug_dump D2H"))
        return 0;
    return 1;
}

extern "C" int
picolm_gpu_kv_store_rows(int is_k, int layer_ordinal, int start_pos, int n_positions,
                          const void *host_rows, size_t row_bytes,
                          int n_kv_heads, int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    /* row_bytes must be the row size of the format this cache was allocated
     * for. F16 is n_kv_heads*head_dim*2; Q8_0/Q4_0 use the block layouts. The
     * CPU and device layouts are byte-identical for a given type, so a plain
     * H2D of the CPU row is correct for every supported type. */
    size_t expect_row_bytes;
    {
        int t = is_k ? g_kv_type_k[device] : g_kv_type_v[device];
        if (g_kv_types_set[device] && t != 0) {
            size_t hs = is_k ? g_kv_head_stride_k[device] : g_kv_head_stride_v[device];
            expect_row_bytes = (size_t)n_kv_heads * hs;
        } else {
            expect_row_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
        }
    }
    if (row_bytes != expect_row_bytes) return 0;

    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    /* Layout: [layer_ordinal][pos][kv_head][head_dim], contiguous rows.
     * n_positions contiguous rows starting at start_pos == one memcpy. */
    size_t layer_off_bytes = (size_t)layer_ordinal * max_seq_len * row_bytes;
    size_t pos_off_bytes = (size_t)start_pos * row_bytes;
    uint8_t *dst = (uint8_t *)dst_base + layer_off_bytes + pos_off_bytes;

    size_t copy_bytes = (size_t)n_positions * row_bytes;
    if (!gpu_ok(gpuMemcpyAsync(dst, host_rows, copy_bytes,
                               gpuMemcpyHostToDevice, ctx->stream),
                "kv_store_rows async copy"))
        return 0;
    return 1;
}

/* Bulk upload: copy all n_positions for one layer from host F16 cache to device.
 * host_rows: [n_positions][n_kv_heads][head_dim] contiguous uint16_t. */
/* Defined below; needed here for the cache row stride. */
static void kv_strides_for(int device, int is_k, int n_kv_heads, int head_dim,
                            size_t *pos_stride_out, size_t *head_stride_out, int *kv_type_out);

extern "C" int
picolm_gpu_kv_upload_layer(int is_k, int layer_ordinal, int n_positions,
                            const void *host_rows, int n_kv_heads,
                            int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    /* Row size comes from the CACHE's own layout, not from head_dim*2.
     *
     * This used to hardcode sizeof(uint16_t) per element, i.e. F16 only, and
     * the caller in picolm.c consequently gated the whole restore path on
     * "-ctk f16 -ctv f16". With a q4_0 cache a row is 0.5625 bytes/element, so
     * the old version would have written 3.5x too many bytes per row and
     * silently corrupted every position past the first. Reading the stride
     * back out also keeps this correct for the case where K and V differ,
     * which the decode kernels already support. */
    size_t pos_stride_k = 0, head_stride_k = 0, pos_stride_v = 0, head_stride_v = 0;
    int kv_type_k = 0, kv_type_v = 0;
    kv_strides_for(device, 1, n_kv_heads, head_dim, &pos_stride_k, &head_stride_k, &kv_type_k);
    kv_strides_for(device, 0, n_kv_heads, head_dim, &pos_stride_v, &head_stride_v, &kv_type_v);
    (void)head_stride_k; (void)head_stride_v; (void)kv_type_k; (void)kv_type_v;

    size_t row_bytes = is_k ? pos_stride_k : pos_stride_v;
    uint8_t *dst_base = (uint8_t *)(is_k ? g_kv_k_dev[device] : g_kv_v_dev[device]);
    size_t layer_off_bytes = (size_t)layer_ordinal * max_seq_len * row_bytes;
    uint8_t *dst = dst_base + layer_off_bytes;

    size_t copy_bytes = (size_t)n_positions * row_bytes;
    if (copy_bytes == 0) return 1;
    if (!gpu_ok(gpuMemcpyAsync(dst, host_rows, copy_bytes,
                               gpuMemcpyHostToDevice, ctx->stream),
                "kv_upload_layer async copy"))
        return 0;
    return 1;
}

/* Resolve the KV cache row/head strides and storage format for one cache
 * (1 = K, 0 = V). Before picolm_gpu_kv_set_type() runs this is F16 with the
 * historical implicit strides, so every caller that does not opt in keeps the
 * exact previous behaviour. After it, the strides are whatever the allocator
 * used, which is what the GPU cache is actually laid out as. */
static void kv_strides_for(int device, int is_k, int n_kv_heads, int head_dim,
                            size_t *pos_stride_out, size_t *head_stride_out, int *kv_type_out) {
    if (device >= 0 && device < PICOLM_GPU_MAX_DEVICES && g_kv_types_set[device]) {
        int t = is_k ? g_kv_type_k[device] : g_kv_type_v[device];
        size_t hs = is_k ? g_kv_head_stride_k[device] : g_kv_head_stride_v[device];
        if (hs == 0) hs = (size_t)head_dim * sizeof(uint16_t);
        *kv_type_out = t;
        *head_stride_out = hs;
        *pos_stride_out = (size_t)n_kv_heads * hs;
        return;
    }
    *kv_type_out = 0;
    *head_stride_out = (size_t)head_dim * sizeof(uint16_t);
    *pos_stride_out = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
}

/* Shared dispatch for both picolm_gpu_attention_decode and _dev: chooses
 * single-pass (grid=n_kv_heads, matching the pre-split-K behavior
 * exactly -- no regression risk for short contexts) vs split-K (grid=
 * n_kv_heads*n_splits + a merge pass) based on how many KV positions
 * there are to walk. xb_dev/q_dev must already be device pointers. */
/* Decode attention on picolm_gpu_attention_decode_warp_kernel: F16 KV cache,
 * head_dim 64/128/256, kv_mul <= 8, CUDA (32-lane warps).
 * PICOLM_NO_ATTN_DECODE_WARP=1 keeps the single-pass/split kernels.
 *
 * The kv_type test is a HARD REQUIREMENT, not a conservative gate:
 * picolm_gpu_attention_decode_warp_kernel takes the cache as
 * `const uint16_t *` and casts straight to `__half` with no dequantisation
 * step at all (it never even receives kv_type). Pointing it at a q4_0 or q8_0
 * cache would read 2 bytes where 18/34 are packed and produce garbage, not a
 * slower answer.
 *
 * TODO(kv-quant): the same is true of picolm_gpu_attention_prefill_hmma_kernel
 * (Volta prefill attention), which likewise takes `const uint16_t *kv_k/v` and
 * assumes F16. So on Volta, QUANTISED KV CACHES (-ctk q4_0/q8_0) get neither
 * the tensor-core prefill attention nor the warp decode attention, and fall
 * back to the scalar/split kernels -- which is the configuration that matters
 * for long context, since q4_0 KV is what makes 120k fit in 24 GB. Reaching
 * parity there needs both kernels taught the kv_row_dequant()/kv_row_load_f16()
 * path the other six attention kernels already use (see
 * backend_gpu_dequant.cuh), plus a kv_type argument plumbed through.
 * Not testable here: no Volta hardware. */
static int attn_decode_warp_ok(int head_dim, int kv_mul, int kv_type) {
#ifdef __HIP__
    (void)head_dim; (void)kv_mul; (void)kv_type;
    return 0;
#else
    static int off = -1;
    if (off < 0) off = getenv("PICOLM_NO_ATTN_DECODE_WARP") ? 1 : 0;
    /* kv_type 0 == F16; the warp kernel has no quantised-KV read path. */
    return !off && kv_type == 0 && (head_dim == 64 || head_dim == 128 || head_dim == 256) && kv_mul <= 8;
#endif
}

static int
attn_decode_dispatch(float *xb_dev, const float *q_dev,
                      int layer_ordinal, int pos,
                      int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
                      gpu_device_ctx_t *ctx, int device, int n_swa) {
    /* The decode kernels score Q@K in head_dim/16 chunks with no tail loop
     * (unlike the single-pass kernel at picolm_gpu_attention_decode_kernel,
     * which does handle a remainder). A head_dim that is not a multiple of 16
     * would silently drop the trailing elements from every score, so refuse
     * the GPU path and let the caller fall back to the CPU. Matches the check
     * already present in picolm_gpu_attention_prefill_dev below. */
    if (head_dim <= 0 || (head_dim % 16) != 0) return 0;
    if (head_dim > 256) return 0;
    if (n_kv_heads <= 0 || n_heads <= 0 || n_heads % n_kv_heads != 0) return 0;

    int kv_mul = n_heads / n_kv_heads;
    size_t kv_pos_stride_bytes, kv_head_stride_bytes;
    size_t kv_pos_stride_v_bytes, kv_head_stride_v_bytes;
    int kv_type = 0, kv_type_v = 0;
    kv_strides_for(device, 1, n_kv_heads, head_dim,
                   &kv_pos_stride_bytes, &kv_head_stride_bytes, &kv_type);
    kv_strides_for(device, 0, n_kv_heads, head_dim,
                   &kv_pos_stride_v_bytes, &kv_head_stride_v_bytes, &kv_type_v);
    (void)kv_pos_stride_v_bytes;
    (void)kv_type_v;   /* K and V share the block format in every kernel below */

    int total_kv = pos + 1;
    if (n_swa > 0 && n_swa < total_kv) total_kv = n_swa; /* SWA shrinks the split-K work estimate */
    /* Split so each block scans ~ATTN_DECODE_TARGET_CHUNK keys. Using
     * ATTN_DECODE_MIN_CHUNK (16) here would ask for 615 splits at 9848 keys,
     * which is pointless: the merge cost and the partial buffer both grow
     * linearly while the scan per block is already tiny. 64 keys/split keeps
     * blocks plentiful (context/64 * n_kv_heads blocks) without making merge
     * dominate. */
    int n_splits = (total_kv + ATTN_DECODE_TARGET_CHUNK - 1) / ATTN_DECODE_TARGET_CHUNK;
    if (n_splits > ATTN_DECODE_MAX_SPLITS) n_splits = ATTN_DECODE_MAX_SPLITS;
    if (n_splits < 1) n_splits = 1;
    /* PICOLM_ATTN_DECODE_CPU=1: refuse the GPU decode attention entirely so
     * the caller falls back to the CPU kernel. Everything else (matmuls, SSM,
     * KV store/read) stays on the GPU. This is the switch that separates
     * "our attention kernel disagrees with the CPU's" from "some other part
     * of the forward pass disagrees", which is otherwise very hard to tell
     * apart when the logits differ. */
    {
        static int no_gpu_attn = -1;
        if (no_gpu_attn < 0) no_gpu_attn = getenv("PICOLM_ATTN_DECODE_CPU") ? 1 : 0;
        if (no_gpu_attn) return 0;
    }
    int warp_kernel = attn_decode_warp_ok(head_dim, kv_mul, kv_type);
    if (warp_kernel) {
        /* ~240 blocks fill the GPU; keep at least 16 keys per split.
         *
         * That 240 is a V100 (80 SM) figure measured by the author of this
         * kernel, and it has not been re-derived on anything else. It only
         * matters once this kernel is reachable on a wider device (it is F16-KV
         * only), so whoever enables it elsewhere should re-measure rather than
         * inherit the constant -- a 4090 has 128 SMs and would want ~384. */
        n_splits = (240 + n_kv_heads - 1) / n_kv_heads;
        if (n_splits > (total_kv + 15) / 16) n_splits = (total_kv + 15) / 16;
        if (n_splits < 1) n_splits = 1;
    }

    size_t shared_bytes = 2 * head_dim * sizeof(uint16_t) + 256 * sizeof(float)
                         + (size_t)kv_mul * head_dim * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + 2 * sizeof(float); /* rescale_sh + weight_sh */

    if (n_splits <= 1 && !warp_kernel) {
        gpu_dispatch_print("attn_decode");
        int _ps = picolm_gpu_prof_slot("attn_decode");
        picolm_gpu_prof_begin(device);
        dim3 grid((unsigned)n_kv_heads, 1, 1);
        picolm_gpu_attention_decode_kernel<<<grid, 256, (unsigned)shared_bytes, ctx->stream>>>(
            xb_dev, q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_swa, kv_type);
        picolm_gpu_prof_end(_ps);
        if (!gpu_ok(gpuGetLastError(), "attn decode kernel")) return 0;
        return 1;
    }

    /* Split-K still splits over the FULL [0,pos] range below (chunk_size
     * is computed from pos+1, not the shrunk total_kv, so split
     * boundaries stay simple); each split kernel clamps its own [t0,t1)
     * to the SWA window internally (see picolm_gpu_attention_decode_split_kernel). */
    int chunk_size = (pos + 1 + n_splits - 1) / n_splits;
    size_t need = (size_t)n_heads * n_splits * (head_dim + 2) * sizeof(float);
    if (!reserve(&ctx->attn_partial, &ctx->attn_partial_cap, need)) return 0;
    float *partial_max = ctx->attn_partial;
    float *partial_sum = partial_max + (size_t)n_heads * n_splits;
    float *partial_acc = partial_sum + (size_t)n_heads * n_splits;

    if (!warp_kernel) gpu_dispatch_print("attn_decode_split");
    /* The split-K path had NO prof slot, so every time context grew past
     * ATTN_DECODE_MIN_CHUNK the decode attention's cost simply left the
     * accounting and inflated the "unexplained" gap. That is precisely the
     * cost that scales with context, so it must be measured. */
    int _ps_split = picolm_gpu_prof_slot("attn_decode_split");
    picolm_gpu_prof_begin(device);
    dim3 grid_split((unsigned)n_kv_heads, (unsigned)n_splits, 1);
    if (warp_kernel) {
        gpu_dispatch_print("attn_decode_warp");
        picolm_gpu_attention_decode_warp_launch(grid_split, ctx->stream,
            partial_max, partial_sum, partial_acc,
            q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_splits, chunk_size, n_swa);
    } else
    picolm_gpu_attention_decode_split_kernel<<<grid_split, 256, (unsigned)shared_bytes, ctx->stream>>>(
        partial_max, partial_sum, partial_acc,
        q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_splits, chunk_size, n_swa, kv_type);
    picolm_gpu_prof_end(_ps_split);
    if (!gpu_ok(gpuGetLastError(), "attn decode split kernel")) return 0;

    /* Merge kernel dynamic shared: global_max/global_sum (kv_mul each) plus
     * the hoisted per-split weight table [kv_mul][ATTN_DECODE_MAX_SPLITS].
     * 6 * 512 * 4 = 12 KB for the largest realistic kv_mul, well under the
     * 48 KB no-opt-in limit. */
    size_t merge_shared = (size_t)kv_mul * 2 * sizeof(float)
                        + (size_t)kv_mul * ATTN_DECODE_MAX_SPLITS * sizeof(float);
    gpu_dispatch_print("attn_decode_merge");
    int _ps_merge = picolm_gpu_prof_slot("attn_decode_merge");
    picolm_gpu_prof_begin(device);
    dim3 grid_merge((unsigned)n_kv_heads, 1, 1);
    picolm_gpu_attention_decode_merge_kernel<<<grid_merge, 128, (unsigned)merge_shared, ctx->stream>>>(
        xb_dev, partial_max, partial_sum, partial_acc,
        n_heads, n_kv_heads, head_dim, n_splits);
    picolm_gpu_prof_end(_ps_merge);
    if (!gpu_ok(gpuGetLastError(), "attn decode merge kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_attention_decode(float *xb_out, const float *q_host,
                             int layer_ordinal, int pos,
                             int n_heads, int n_kv_heads, int head_dim,
                             int max_seq_len, int device, int n_swa) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    size_t x_bytes = (size_t)n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice),
                "attn decode Q upload")) return 0;

    if (!attn_decode_dispatch(ctx->y, ctx->x, layer_ordinal, pos,
                              n_heads, n_kv_heads, head_dim, max_seq_len,
                              ctx, device, n_swa)) return 0;

    if (!gpu_ok(gpuDeviceSynchronize(), "attn decode sync")) return 0;

    if (!gpu_ok(gpuMemcpy(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost),
                "attn decode output download")) return 0;
    return 1;
}

/* Device-native decode attention: q_dev and xb_out_dev are already
 * device-resident (Phase 2 pipeline buffers, e.g. the output of the Q
 * projection matmul_dev and rope_apply, both in place on the same
 * buffer). No H2D, no D2H, no gpuDeviceSynchronize() -- same ordering
 * argument as picolm_gpu_matmul_dev: this kernel and picolm_gpu_kv_store_rows
 * (the KV write for this position) are both launched on ctx->stream, so
 * in-stream ordering guarantees the K/V write for `pos` is visible before
 * this kernel reads it, PROVIDED the K/V store call for `pos` happens
 * first in program order on the same device/stream -- caller must store
 * K/V before calling this. q_dev/xb_out_dev must not alias ctx->x/ctx->y. */
extern "C" int
picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
                                 int layer_ordinal, int pos,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device, int n_swa) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0; /* stack array limit */

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    return attn_decode_dispatch(xb_out_dev, q_dev, layer_ordinal, pos,
                                 n_heads, n_kv_heads, head_dim, max_seq_len,
                                 ctx, device, n_swa);
}

/* Total dynamic shared memory picolm_gpu_attention_prefill_kernel needs:
 * K tile + V tile (u16) + reduce (256 float) + acc[tile_q][head_dim]
 * (float) + max_score[tile_q] + sum_exp[tile_q] (float).
 * tile_q is chosen by the caller based on available shared memory.
 * For head_dim=128 with tile_q=32 this is ~50KB -- over the default
 * 48KB/block limit, so the caller must opt in via gpuFuncSetAttribute. */
static size_t
attn_prefill_shared_bytes(int head_dim, int tile_q) {
    /* Dynamic shared memory (extern __shared__): K tile + V tile + reduce +
     * acc + max_score + sum_exp. Plus 8 bytes of static __shared__ for
     * rescale_sh/weight_sh. Total must be set via gpuFuncSetAttribute. */
    return 2 * (size_t)ATTN_TILE_K * head_dim * sizeof(uint16_t)
         + 256 * sizeof(float)
         + (size_t)tile_q * head_dim * sizeof(float)
         + (size_t)tile_q * sizeof(float)
         + (size_t)tile_q * sizeof(float)
         + 2 * sizeof(float); /* rescale_sh + weight_sh (static __shared__) */
}

/* Opts the kernel into a larger dynamic shared memory limit if needed.
 * Idempotent: tracks the largest size already configured so repeated
 * calls (every prefill call, every layer) are a cheap no-op after the
 * first. Returns 1 if `bytes` is safe to launch with, 0 if the opt-in
 * itself failed (e.g. device doesn't support this much shared memory). */
static int
ensure_attn_prefill_shared_mem(size_t bytes) {
    static size_t configured = 49152; /* default limit, no opt-in needed under this */
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn prefill shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

/* Query device for maximum shared memory per block (in bytes).
 * Returns 0 on failure. */
static size_t
get_device_shared_mem_per_block(int device) {
    static size_t cached[PICOLM_GPU_MAX_DEVICES] = {0};
    if (device < 0 || device >= PICOLM_GPU_MAX_DEVICES) return 0;
    if (cached[device]) return cached[device];
    int value = 0;
    if (!gpu_ok(gpuDeviceGetAttribute(&value, gpuDeviceAttributeMaxSharedMemoryPerBlockOptin, device),
                "get shared mem per block")) {
        /* Fall back to device total shared memory per multiprocessor / max blocks per SM.
         * Some older drivers don't support the optin attribute. */
        int smem_mp = 0, blocks_sm = 1;
        gpuDeviceGetAttribute(&smem_mp, gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor, device);
        gpuDeviceGetAttribute(&blocks_sm, gpuDeviceAttributeMaxBlocksPerMultiprocessor, device);
        if (smem_mp > 0 && blocks_sm > 0) {
            cached[device] = (size_t)smem_mp / blocks_sm;
            return cached[device];
        }
        cached[device] = 49152; /* conservative default */
        return cached[device];
    }
    cached[device] = (size_t)value;
    return cached[device];
}

static int
ensure_attn_fa3_shared_mem(size_t bytes) {
    static size_t configured = 49152;
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_fa3_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn fa3 shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

static int
ensure_attn_fa2_shared_mem(size_t bytes) {
    static size_t configured = 49152;
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_fa2_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn fa2 shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

/* Same layout as attn_prefill_shared_bytes but for the warpgrp kernel:
 * no reduce_sh buffer (reduction is warp-local now), so this is 1KB+8B
 * smaller for the same head_dim/tile_q -- more headroom before the
 * dynamic-shared-memory opt-in kicks in. */
/* Volta tensor-core attention prefill: has_hmma, head_dim 128, F16 KV cache.
 * PICOLM_NO_ATTN_HMMA=1 (or PICOLM_FORCE_SCALAR_ATTN) keeps the scalar kernels.
 *
 * The kv_type test is a HARD REQUIREMENT: picolm_gpu_attention_prefill_hmma_kernel
 * takes `const uint16_t *kv_k/v` and casts to __half, with no dequantisation and
 * no kv_type parameter at all.
 *
 * TODO(kv-quant): QUANTISED KV IS UNSUPPORTED HERE. With -ctk q4_0/q8_0 on
 * Volta this declines silently and the model runs the scalar prefill attention
 * instead -- which is precisely the configuration long context needs, since a
 * quantised cache is what makes 120k tokens fit in 24 GB of VRAM. Both Volta
 * kernels (this one and picolm_gpu_attention_decode_warp_kernel) need the same
 * kv_row_dequant()/kv_row_load_f16() treatment the other six attention kernels
 * already have (backend_gpu_dequant.cuh), plus a kv_type argument. Until then
 * Volta + quantised KV is correct but leaves both fast paths on the table.
 * Cannot be exercised or verified here: no Volta hardware available. */
static int picolm_gpu_attn_hmma(const gpu_device_ctx_t *ctx, int n_tokens, int head_dim, int kv_type) {
    static int off = -1;
    if (off < 0) off = getenv("PICOLM_NO_ATTN_HMMA") || getenv("PICOLM_FORCE_SCALAR_ATTN");
    /* kv_type 0 == F16; the HMMA prefill kernel has no quantised-KV read path. */
    return !off && ctx->has_hmma && n_tokens >= 64 && head_dim == 128 && kv_type == 0;
}

static size_t
attn_prefill_warpgrp_shared_bytes(int head_dim, int tile_q) {
    /* Q tile (FP16) + K tile (FP16) + V tile (FP16) + acc (FP32) + max/sum (FP32) */
    return (size_t)tile_q * head_dim * sizeof(uint16_t)
         + 2 * (size_t)ATTN_TILE_K * head_dim * sizeof(uint16_t)
         + (size_t)tile_q * head_dim * sizeof(float)
         + (size_t)tile_q * sizeof(float)
         + (size_t)tile_q * sizeof(float);
}

static int
ensure_attn_prefill_warpgrp_shared_mem(size_t bytes) {
    static size_t configured = 49152;
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn prefill warpgrp shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

#ifdef GPU_FP16_DOT2_AVAILABLE
/* Separate cache from the fmaf variant above: different kernel symbol,
 * needs its own gpuFuncSetAttribute call, same shared-mem layout. */
static int
ensure_attn_prefill_warpgrp_dot2_shared_mem(size_t bytes) {
    static size_t configured = 49152;
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_dot2_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn prefill warpgrp dot2 shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}
#endif

/* Default: warp-group scalar kernel (fast path).
 * PICOLM_ATTN_SLOW_SCALAR=1 forces the legacy block-wide-reduce scalar
 * kernel for debugging/comparison. FA2 always wins when has_imma is
 * available; this only affects the non-tensor-core fallback path. */
static int
use_attn_warpgrp(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_ATTN_SLOW_SCALAR") ? 0 : 1;
    return cached;
}

/* Opt-in on top of the warpgrp path: PICOLM_ATTN_DOT2=1 swaps the
 * fmaf+gpu_fp16_to_fp32 score loop for the v_dot2_f32_f16-equivalent
 * packed-half2 path. NOT bit-exact with the CPU reference -- see the
 * kernel comment in backend_gpu_kernels.cu. Off by default; validate
 * with PICOLM_LOGITS_DUMP before trusting it beyond a benchmark run.
 * Silently inert (falls back to the fmaf kernel) if GPU_FP16_DOT2_AVAILABLE
 * isn't defined for this build. */
#ifdef GPU_FP16_DOT2_AVAILABLE
static int
use_attn_dot2(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("PICOLM_ATTN_DOT2") ? 1 : 0;
    }
    return cached;
}
#endif

extern "C" int
picolm_gpu_attention_prefill(float *xb_out, const float *q_host,
                              int layer_ordinal, int start_pos, int n_tokens,
                              int n_heads, int n_kv_heads, int head_dim,
                              int max_seq_len, int device, int n_swa) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;
    /* Same reasoning as attn_decode_dispatch: the prefill kernels reduce over
     * head_dim/16 chunks and the warpgroup variants have no tail loop, so a
     * head_dim that is not a multiple of 16 would silently truncate every Q@K
     * score. Refuse the GPU path (caller falls back to batch_attention_layer).
     * This mirrors picolm_gpu_attention_prefill_dev's existing check. */
    if (head_dim <= 0 || (head_dim % 16) != 0) return 0;
    if (n_kv_heads <= 0 || n_heads <= 0 || n_heads % n_kv_heads != 0) return 0;

    size_t x_bytes = (size_t)n_tokens * n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpyAsync(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice, ctx->stream),
                "attn prefill Q upload")) return 0;

    size_t kv_pos_stride_bytes, kv_head_stride_bytes;
    size_t kv_pos_stride_v_bytes, kv_head_stride_v_bytes;
    int kv_type = 0, kv_type_v = 0;
    kv_strides_for(device, 1, n_kv_heads, head_dim,
                   &kv_pos_stride_bytes, &kv_head_stride_bytes, &kv_type);
    kv_strides_for(device, 0, n_kv_heads, head_dim,
                   &kv_pos_stride_v_bytes, &kv_head_stride_v_bytes, &kv_type_v);
    (void)kv_pos_stride_v_bytes;
    (void)kv_type_v;   /* K and V share the block format in every kernel below */

    /* Try FA2 (tensor core) kernel when available (NVIDIA IMMA only) */
    int use_fa2 = ctx->has_imma && n_tokens >= 64 && !getenv("PICOLM_FORCE_SCALAR_ATTN");
    if (use_fa2) {
        int n_q_tiles = (n_tokens + 63) / 64;
        /* Shared memory per block (one query head, 4 warps * 16 Q rows):
         * K(FA2_TILE_K*hd) + V(FA2_TILE_K*hd) + Q(64*hd) uint16_t
         * + score(64*FA2_TILE_K) + acc(64*hd) + max(64) + sum(64) float
         * For hd=128, TILE_K=32: 72.5KB. For hd=256, TILE_K=32: 137KB (exceeds limit, scalar fallback). */
        size_t fa2_shared = (size_t)FA2_TILE_K * head_dim * 2 * sizeof(uint16_t)
            + 64 * head_dim * sizeof(uint16_t)
            + 64 * FA2_TILE_K * sizeof(float)
            + 64 * head_dim * sizeof(float)
            + 128 * sizeof(float);
        if (fa2_shared <= 98304 && ensure_attn_fa2_shared_mem(fa2_shared)) {
            gpu_dispatch_print("attn_prefill_fa2_host");
            dim3 grid((unsigned)n_heads, (unsigned)n_q_tiles, 1);
            picolm_gpu_attention_prefill_fa2_kernel<<<grid, 128, (unsigned)fa2_shared, ctx->stream>>>(
                ctx->y, ctx->x,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_swa, kv_type);
            if (!gpu_ok(gpuGetLastError(), "attn prefill fa2 (host)")) return 0;
        } else {
            use_fa2 = 0;
        }
    }

    if (!use_fa2 && picolm_gpu_attn_hmma(ctx, n_tokens, head_dim, kv_type)) {
        gpu_dispatch_print("attn_prefill_hmma_host");
        dim3 grid((unsigned)n_heads, (unsigned)((n_tokens + 63) / 64), 1);
        picolm_gpu_attention_prefill_hmma_kernel<<<grid, 128, 0, ctx->stream>>>(
            ctx->y, ctx->x,
            g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_swa);
        if (!gpu_ok(gpuGetLastError(), "attn prefill hmma (host)")) return 0;
        use_fa2 = 1;   /* done: skip the scalar kernels below */
    }

    /* TODO(perf, head_dim=256): on NVIDIA IMMA GPUs, the warpgrp path below
     * is pure scalar F32 and does not use tensor cores. For Qwen3.5/3.6
     * (head_dim=256), FA2 falls through here because its shared memory
     * exceeds 96 KB. See /data4/work/notes/picolm/fa2_headdim256.md */
    if (!use_fa2) {
        int tile_q = ATTN_TILE_Q;
        size_t smem_max = get_device_shared_mem_per_block(device);
        int block_threads = 128;

        if (use_attn_warpgrp()) {
            /* 256 threads (8 x 32-lane subgroups) instead of 128:
             * pure occupancy increase, no precision impact -- more
             * subgroups working the same tile_q rows in parallel,
             * hiding K/V-load latency across more warps/wavefronts. */
            block_threads = 256;
            while (tile_q > 8 && attn_prefill_warpgrp_shared_bytes(head_dim, tile_q) > smem_max) {
                tile_q /= 2;
            }
            int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
            size_t shared_bytes = attn_prefill_warpgrp_shared_bytes(head_dim, tile_q);
            dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
#ifdef GPU_FP16_DOT2_AVAILABLE
            if (use_attn_dot2()) {
                gpu_dispatch_print("attn_prefill_warpgrp_dot2_host");
                if (!ensure_attn_prefill_warpgrp_dot2_shared_mem(shared_bytes)) return 0;
                picolm_gpu_attention_prefill_warpgrp_dot2_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
                    ctx->y, ctx->x,
                    g_kv_k_dev[device], g_kv_v_dev[device],
                    layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                    kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, tile_q, n_swa, kv_type);
            } else
#endif
            {
                gpu_dispatch_print("attn_prefill_warpgrp_host");
                if (!ensure_attn_prefill_warpgrp_shared_mem(shared_bytes)) return 0;
                picolm_gpu_attention_prefill_warpgrp_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
                    ctx->y, ctx->x,
                    g_kv_k_dev[device], g_kv_v_dev[device],
                    layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                    kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, tile_q, n_swa, kv_type);
            }
        } else {
            gpu_dispatch_print("attn_prefill_scalar_host");
            /* Choose tile_q that fits device shared memory */
            while (tile_q > 8 && attn_prefill_shared_bytes(head_dim, tile_q) > smem_max) {
                tile_q /= 2;
            }
            int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
            size_t shared_bytes = attn_prefill_shared_bytes(head_dim, tile_q);
            if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;

            dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
            picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
                ctx->y, ctx->x,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, tile_q, n_swa, kv_type);
        }
    }

    if (!gpu_ok(gpuGetLastError(), "attn prefill kernel")) return 0;

    /* Async D2H on ctx->stream. Must sync before CPU reads xb_out, since
     * the caller (model_core.c prefill loop) uses xb_batch immediately
     * for Hadamard transforms, output projection matmul, etc. */
    if (!gpu_ok(gpuMemcpyAsync(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost, ctx->stream),
                "attn prefill output download")) return 0;
    if (!gpu_ok(gpuStreamSynchronize(ctx->stream), "attn prefill D2H sync")) return 0;
    return 1;
}

/* Device-native prefill attention: xb_out_dev and q_dev are already
 * device-resident. No H2D, no D2H, no gpuDeviceSynchronize(). */
extern "C" int
attn_prefill_dev_impl(float *xb_out_dev, const float *q_dev,
                                  int layer_ordinal, int start_pos, int n_tokens,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, int device, int n_swa) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;
    if (head_dim % 16 != 0) return 0;

    size_t kv_pos_stride_bytes, kv_head_stride_bytes;
    size_t kv_pos_stride_v_bytes, kv_head_stride_v_bytes;
    int kv_type = 0, kv_type_v = 0;
    kv_strides_for(device, 1, n_kv_heads, head_dim,
                   &kv_pos_stride_bytes, &kv_head_stride_bytes, &kv_type);
    kv_strides_for(device, 0, n_kv_heads, head_dim,
                   &kv_pos_stride_v_bytes, &kv_head_stride_v_bytes, &kv_type_v);
    (void)kv_pos_stride_v_bytes;
    (void)kv_type_v;   /* K and V share the block format in every kernel below */

    /* FA3 first: register-resident O, so it fits head_dim=256 where FA2 cannot.
     * Any kv_type (it dequantises in the tile load via kv_row_dequant). */
    {
        /* FA3 is NOT enabled by default: it produces NaN logits.
         *
         * The kernel is written and it fits head_dim=256 in shared memory
         * (the whole point -- FA2 needs 136.5 KB against a 96 KB cap, which is
         * why Qwen3.5/3.6 prefill attention falls back to the scalar warpgrp
         * kernel and costs 26.6% of prefill at 4096 tokens, growing as O(n^2)).
         * But the Q@K tensor-core path is wrong: scores come out ~1e11 instead
         * of ~1e2, so the softmax collapses to the first key (sum=1.0) and the
         * output is garbage.
         *
         * Fixed along the way: a divergent gpuSyncthreads() when q0 >= n_tokens
         * for some warps only (the warp now stays alive and skips its maths).
         *
         * Still broken: the m16n8k16 operand mapping. The A/B fragment
         * assembly was copied from the FA2 kernel, which is verified, so the
         * difference is somewhere in the shared-memory layout this kernel uses
         * (per-warp Q tile of 16 rows) rather than in the instruction itself.
         * Opt in with PICOLM_FA3_ATTN=1 to work on it. */
        static int no_fa3 = -1;
        if (no_fa3 < 0) no_fa3 = getenv("PICOLM_FA3_ATTN") ? 0 : 1;
        int fa3_tk = ctx->has_imma ? 32 : 0;
        if (!no_fa3 && fa3_tk && n_tokens >= 32 && !getenv("PICOLM_FORCE_SCALAR_ATTN")
            && (head_dim % 32) == 0) {
            size_t f3 = (size_t)2 * fa3_tk * head_dim * sizeof(uint16_t)
                      + (size_t)FA3_WARPS * FA3_ROWS * head_dim * sizeof(uint16_t)
                      + (size_t)FA3_WARPS * FA3_ROWS * fa3_tk * sizeof(float)
                      + (size_t)FA3_WARPS * FA3_ROWS * fa3_tk * sizeof(uint16_t);
            static size_t f3_ok_for = 0;
            if (f3 != f3_ok_for) {
                if (!ensure_attn_fa3_shared_mem(f3)) { f3 = 0; }
                else f3_ok_for = f3;
            }
            if (f3) {
                gpu_dispatch_print("attn_prefill_fa3_dev");
                dim3 grid3((unsigned)n_heads,
                           (unsigned)((n_tokens + FA3_ROWS * FA3_WARPS - 1) / (FA3_ROWS * FA3_WARPS)), 1);
                picolm_gpu_attention_prefill_fa3_kernel<<<grid3, 32 * FA3_WARPS, (unsigned)f3, ctx->stream>>>(
                    xb_out_dev, q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
                    layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim,
                    max_seq_len, kv_pos_stride_bytes, kv_head_stride_bytes,
                    kv_head_stride_v_bytes, n_swa, kv_type, fa3_tk);
                if (gpu_ok(gpuGetLastError(), "attn prefill fa3")) return 1;
                gpuGetLastError();
            }
        }
    }

    /* Use FA2 (tensor core) kernel only on NVIDIA with IMMA.
     * HIP scalar fallback is too slow (10-15s for 200 tokens). */
    if (ctx->has_imma && n_tokens >= 64 && !getenv("PICOLM_FORCE_SCALAR_ATTN")) {
        int n_q_tiles = (n_tokens + 63) / 64;  /* ceil, kernel handles OOB */
        /* Shared memory per block (one query head):
         * K(FA2_TILE_K*hd) + V(FA2_TILE_K*hd) + Q(64*hd) uint16_t
         * + score(64*FA2_TILE_K) + acc(64*hd) + max/sum float
         * For hd=128, TILE_K=32: 72.5KB. For hd=256, TILE_K=32: 137KB (exceeds 96KB, scalar fallback). */
        size_t fa2_shared = (size_t)FA2_TILE_K * head_dim * 2 * sizeof(uint16_t)
            + 64 * head_dim * sizeof(uint16_t)
            + 64 * FA2_TILE_K * sizeof(float)
            + 64 * head_dim * sizeof(float)
            + 128 * sizeof(float);
        if (fa2_shared > 98304) {
            /* TODO: head_dim=256 (Qwen3.5/3.6) exceeds 96 KB shared memory limit.
             * acc_sh[64][head_dim] alone is 64 KB. FA2 processes 64 Q rows per block.
             * Fix: add FA2 variant with 32 Q rows per block (halves acc_sh to 32 KB,
             * total ~104 KB with TILE_K=32, or ~66 KB with TILE_K=16).
             * See /data4/work/notes/picolm/fa2_headdim256.md for full analysis.
             * Falls through to warpgrp (scalar F32, no tensor cores) below. */
        } else {
            if (!ensure_attn_fa2_shared_mem(fa2_shared)) return 0;
            gpu_dispatch_print("attn_prefill_fa2_dev");
            dim3 grid((unsigned)n_heads, (unsigned)n_q_tiles, 1);
            picolm_gpu_attention_prefill_fa2_kernel<<<grid, 128, (unsigned)fa2_shared, ctx->stream>>>(
                xb_out_dev, q_dev,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_swa, kv_type);
            if (!gpu_ok(gpuGetLastError(), "attn prefill fa2 (dev)")) return 0;
            return 1;
        }
    }

    if (picolm_gpu_attn_hmma(ctx, n_tokens, head_dim, kv_type)) {
        gpu_dispatch_print("attn_prefill_hmma_dev");
        dim3 grid((unsigned)n_heads, (unsigned)((n_tokens + 63) / 64), 1);
        picolm_gpu_attention_prefill_hmma_kernel<<<grid, 128, 0, ctx->stream>>>(
            xb_out_dev, q_dev,
            g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, n_swa);
        if (!gpu_ok(gpuGetLastError(), "attn prefill hmma (dev)")) return 0;
        return 1;
    }

    int tile_q = ATTN_TILE_Q;
    size_t smem_max = get_device_shared_mem_per_block(device);
    int block_threads = 128;

    if (use_attn_warpgrp()) {
        block_threads = 256;
        while (tile_q > 8 && attn_prefill_warpgrp_shared_bytes(head_dim, tile_q) > smem_max) {
            tile_q /= 2;
        }
        int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
        size_t shared_bytes = attn_prefill_warpgrp_shared_bytes(head_dim, tile_q);
        dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
#ifdef GPU_FP16_DOT2_AVAILABLE
        if (use_attn_dot2()) {
            gpu_dispatch_print("attn_prefill_warpgrp_dot2_dev");
            if (!ensure_attn_prefill_warpgrp_dot2_shared_mem(shared_bytes)) return 0;
            picolm_gpu_attention_prefill_warpgrp_dot2_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
                xb_out_dev, q_dev,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes, tile_q, n_swa);
            if (!gpu_ok(gpuGetLastError(), "attn prefill warpgrp dot2 (dev)")) return 0;
            return 1;
        }
#endif
        gpu_dispatch_print("attn_prefill_warpgrp_dev");
        if (!ensure_attn_prefill_warpgrp_shared_mem(shared_bytes)) return 0;
        picolm_gpu_attention_prefill_warpgrp_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
            xb_out_dev, q_dev,
            g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, tile_q, n_swa, kv_type);
        if (!gpu_ok(gpuGetLastError(), "attn prefill warpgrp (dev)")) return 0;
        return 1;
    }

    gpu_dispatch_print("attn_prefill_scalar_dev");
    /* Scalar fallback: choose tile_q that fits device shared memory.
     * Start with 32, halve until it fits. Minimum is 8. */
    while (tile_q > 8 && attn_prefill_shared_bytes(head_dim, tile_q) > smem_max) {
        tile_q /= 2;
    }
    int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
    size_t shared_bytes = attn_prefill_shared_bytes(head_dim, tile_q);
    if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;

    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes, kv_head_stride_v_bytes, tile_q, n_swa, kv_type);
    if (!gpu_ok(gpuGetLastError(), "attn prefill (dev)")) return 0;
    return 1;
}

/* Profiled wrapper. Prefill attention had NO slot, so it contributed nothing
 * to the kernel sum while the layer loop ran ~730 ms faster than that sum
 * explained. */
extern "C" int
picolm_gpu_attention_prefill_dev(float *xb_out_dev, const float *q_dev,
                                  int layer_ordinal, int start_pos, int n_tokens,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, int device, int n_swa) {
    int ps = picolm_gpu_prof_slot("attn_prefill");
    picolm_gpu_prof_begin(device);
    int rc = attn_prefill_dev_impl(xb_out_dev, q_dev, layer_ordinal, start_pos,
                                   n_tokens, n_heads, n_kv_heads, head_dim,
                                   max_seq_len, device, n_swa);
    picolm_gpu_prof_end(ps);
    return rc;
}

/* Device-native prefill attention with FP32 K/V (no KV cache read).
 * Used for SSM-hybrid models where FP16 KV round-trip causes too much
 * numerical drift. K and V are passed as FP32 projection buffers
 * with the same layout as the KV cache: [pos][kv_head][head_dim]. */
extern "C" int
picolm_gpu_attention_prefill_f32kv(float *xb_out_dev, const float *q_dev,
                                    const float *k_dev, const float *v_dev,
                                    int start_pos, int n_tokens,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    int device, int n_swa) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim > 256) return 0;

    /* FP32 K/V tiles are 4x larger than FP16.
     * shared_bytes(tile_q) = 2*ATTN_TILE_K*hd*4 + 256*4 + tile_q*hd*4 + 2*tile_q*4 + 8
     * For hd=256, tile_q=32: 65536+1024+32768+256+8 = 99592 (~97KB) -- too big for gfx906.
     * Choose tile_q adaptively. */
    int tile_q = ATTN_TILE_Q;
    size_t smem_max = get_device_shared_mem_per_block(device);
    auto f32kv_smem = [](int hd, int tq) -> size_t {
        return 2 * (size_t)ATTN_TILE_K * hd * sizeof(float)
             + 256 * sizeof(float)
             + (size_t)tq * hd * sizeof(float)
             + 2 * (size_t)tq * sizeof(float)
             + 2 * sizeof(float);
    };
    while (tile_q > 8 && f32kv_smem(head_dim, tile_q) > smem_max) {
        tile_q /= 2;
    }
    int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
    size_t shared_bytes = f32kv_smem(head_dim, tile_q);
    /* Opt-in FP32 kernel to larger shared memory */
    if (shared_bytes > 49152) {
        gpuError_t e = gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_f32kv_kernel,
            gpuFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
        if (e != gpuSuccess) {
            fprintf(stderr, "[GPU] f32kv opt-in fail: err=%d (%s) bytes=%zu\n",
                (int)e, gpuGetErrorString(e), shared_bytes);
        }
    }
    gpu_dispatch_print("attn_prefill_f32kv_dev");
    int block_threads = 128;
    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_f32kv_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev, k_dev, v_dev,
        start_pos, n_tokens, n_heads, n_kv_heads, head_dim, tile_q, n_swa);
    if (!gpu_ok(gpuGetLastError(), "attn prefill f32kv (dev)")) return 0;
    return 1;
}

/* ================================================================
 * Device-resident elementwise kernels
 * ================================================================ */

/* RMSNorm kernel: out[d] = x[d] * rsqrt(mean(x^2) + eps) * weight[d].
 * Single block, grid-stride loop -- correct for any dim (an earlier
 * version returned early before __syncthreads() for threads with
 * d >= dim, which deadlocks/UB whenever dim isn't an exact multiple of
 * blockDim). Launched with grid=1 by the host wrapper below. */
__global__ void
picolm_gpu_rmsnorm_kernel(float *out, const float *x, const float *weight,
                           int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += x[i] * x[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        out[d] = x[d] * inv_rms * weight[d];
    }
}

/* Batched rmsnorm: one block per row (S rows), same reduction per block.
 * weight is shared across all rows. Mirrors how matmul_dev batches over S. */
__global__ void
picolm_gpu_rmsnorm_batched_kernel(float *out, const float *x, const float *weight,
                                   int dim, float eps, int x_stride) {
    int row = (int)gpuBlockIdx_x;
    int stride = (x_stride > 0) ? x_stride : dim;
    const float *xr = x + (size_t)row * stride;
    float *outr = out + (size_t)row * stride;

    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += xr[i] * xr[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        outr[d] = xr[d] * inv_rms * weight[d];
    }
}

/* LayerNorm kernel (mean subtraction + bias): GPT-2/CodeGen.
 * Two-pass per block: pass 1 computes mean, pass 2 normalizes + bias.
 * out[d] = (x[d] - mean) / sqrt(var + eps) * weight[d] + bias[d] */
__global__ void
picolm_gpu_layernorm_kernel(float *out, const float *x, const float *weight,
                             const float *bias, int dim, float eps) {
    /* Pass 1: compute mean */
    float sum = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum += x[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float mean = ssum[0] / dim;

    /* Pass 2: compute variance and write output */
    float var = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    ssum[gpuThreadIdx_x] = var;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float inv_rms = 1.0f / sqrtf(ssum[0] / dim + eps);

    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        out[d] = (x[d] - mean) * inv_rms * weight[d] + bias[d];
    }
}

/* Batched LayerNorm: one block per row (S rows). */
__global__ void
picolm_gpu_layernorm_batched_kernel(float *out, const float *x, const float *weight,
                                     const float *bias, int dim, float eps, int x_stride) {
    int row = (int)gpuBlockIdx_x;
    int stride = (x_stride > 0) ? x_stride : dim;
    const float *xr = x + (size_t)row * stride;
    float *outr = out + (size_t)row * stride;

    /* Pass 1: mean */
    float sum = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum += xr[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float mean = ssum[0] / dim;

    /* Pass 2: variance + write */
    float var = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        float diff = xr[i] - mean;
        var += diff * diff;
    }
    ssum[gpuThreadIdx_x] = var;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float inv_rms = 1.0f / sqrtf(ssum[0] / dim + eps);

    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        outr[d] = (xr[d] - mean) * inv_rms * weight[d] + bias[d];
    }
}

/* Batched RoPE: each element computes its own absolute position's table
 * row. x is [S][n_heads][head_dim] contiguous, positions start_pos..start_pos+S-1.
 * One launch for the whole prefill chunk. */
__global__ void
picolm_gpu_rope_batched_kernel(float *x, int n_heads, int head_dim,
                                const float *cos_tbl, const float *sin_tbl,
                                int half_dim, int start_pos, int S,
                                int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int per_row = n_heads * head_dim;
    int total = S * per_row;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int row = i / per_row;
        int rem = i % per_row;
        int h = rem / head_dim;
        int d = rem % head_dim;
        float *xr = x + (size_t)row * per_row;
        int pos = start_pos + row;

        if (rope_type < 0) continue; /* no RoPE (GPT-2 learned pos embd) */
        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = xr[h * head_dim + h1];
            float x2 = xr[h * head_dim + h2];
            float c = cos_tbl[(size_t)pos * half_dim + h1];
            float s = sin_tbl[(size_t)pos * half_dim + h1];
            xr[h * head_dim + h1] = x1 * c - x2 * s;
            xr[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd index: handled by even partner */
            if (d + 1 >= head_dim) continue; /* safety: beyond head */
            int idx_pair = d >> 1; /* 0,1 -> pair 0; 2,3 -> pair 1; etc. */
            if (idx_pair >= half_dim) continue; /* beyond rope range */
            float x0 = xr[h * head_dim + d];
            float x1v = xr[h * head_dim + d + 1];
            float c = cos_tbl[(size_t)pos * half_dim + idx_pair];
            float s = sin_tbl[(size_t)pos * half_dim + idx_pair];
            xr[h * head_dim + d]     = x0 * c - x1v * s;
            xr[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* RoPE kernel: applies pairwise rotary position embedding */
__global__ void
picolm_gpu_rope_kernel(float *x, int n_heads, int head_dim,
                        const float *cos_tbl, const float *sin_tbl,
                        int half_dim, int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n_heads * head_dim;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int h = i / head_dim;
        int d = i % head_dim;

        if (rope_type < 0) continue; /* no RoPE (GPT-2 learned pos embd) */
        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = x[h * head_dim + h1];
            float x2 = x[h * head_dim + h2];
            float c = cos_tbl[h1];
            float s = sin_tbl[h1];
            x[h * head_dim + h1] = x1 * c - x2 * s;
            x[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd: handled by even partner */
            if (d + 1 >= head_dim) continue;
            int idx_pair = d >> 1;
            if (idx_pair >= half_dim) continue;
            float x0 = x[h * head_dim + d];
            float x1v = x[h * head_dim + d + 1];
            float c = cos_tbl[idx_pair];
            float s = sin_tbl[idx_pair];
            x[h * head_dim + d]     = x0 * c - x1v * s;
            x[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* Residual add kernel: out[i] = a[i] + b[i] */
__global__ void
picolm_gpu_residual_add_kernel(float *out, const float *a, const float *b, int n, int dim, int stride) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * dim;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int tok = i / dim;
        int off = i % dim;
        out[tok * stride + off] = a[tok * stride + off] + b[tok * stride + off];
    }
}

/* Q+gate de-interleave kernel for SSM attention layers.
 * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] per head, each head_dim floats.
 * Reads raw[head * 2 * head_dim + ...], writes Q to out_q[head * head_dim + ...]
 * and Gate to out_g[head * head_dim + ...].
 * One thread per head, thread-0 of each block does the memmove-like copy. */
__global__ void
picolm_gpu_qg_deinterleave_kernel(const float *raw, float *out_q, float *out_g,
                                   int n_heads, int head_dim) {
    int h = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (h >= n_heads) return;
    const float *src = raw + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];           /* Q portion */
        dst_g[d] = src[head_dim + d]; /* Gate portion */
    }
}

/* Elementwise sigmoid-multiply kernel: out[i] = a[i] * sigmoid(g[i])
 * For SSM attention: pipe_attn_out *= sigmoid(gate) before output projection. */
__global__ void
picolm_gpu_sigmoid_mul_kernel(float *out, const float *gate, int n) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    for (; i < n; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Batched Q+gate de-interleave: processes S sequences, each with n_heads heads.
 * raw is [S][n_heads * 2 * head_dim], out_q is [S][n_heads * head_dim],
 * out_g is [S][n_heads * head_dim]. */
__global__ void
picolm_gpu_qg_deinterleave_batched_kernel(const float *raw, float *out_q,
                                           float *out_g, int n_heads,
                                           int head_dim, int S) {
    int idx = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    int total = n_heads * S;
    if (idx >= total) return;
    int s = idx / n_heads;
    int h = idx % n_heads;
    size_t row_stride = (size_t)n_heads * 2 * head_dim;
    size_t q_stride = (size_t)n_heads * head_dim;
    const float *src = raw + (size_t)s * row_stride + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)s * q_stride + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)s * q_stride + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];
        dst_g[d] = src[head_dim + d];
    }
}

/* Batched sigmoid-multiply: processes S sequences, each n elements. */
__global__ void
picolm_gpu_sigmoid_mul_batched_kernel(float *out, const float *gate,
                                       int n, int S) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * S;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Inplace GELU: x[i] = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) */
__global__ void
picolm_gpu_gelu_inplace_kernel(float *x, size_t n) {
    size_t i = (size_t)gpuBlockIdx_x * (size_t)gpuBlockDim_x + gpuThreadIdx_x;
    size_t stride = (size_t)gridDim.x * gpuBlockDim_x;
    for (; i < n; i += stride) {
        float xi = x[i];
        float cubed = xi * xi * xi;
        x[i] = 0.5f * xi * (1.0f + tanhf(0.79788456f * (xi + 0.044715f * cubed)));
    }
}

/* Phase 2 host API */
/* Device-native rmsnorm: all pointers are device-resident, no H2D/D2H, no sync.
 * Used from model_forward_gpu() pipeline path. */
extern "C" int
picolm_gpu_rmsnorm_dev(float *out, const float *x, const float *weight,
                        int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm dev kernel")) return 0;
    return 1;
}

/* Host-side rmsnorm: takes host pointers, does H2D/D2H/sync.
 * Used from ssm_forward() QK-norm path and other host-side callers. */
extern "C" int
picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                    int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    size_t xb = dim * sizeof(float);
    size_t wb = dim * sizeof(float);
    void *dx, *dw, *do_;
    if (!gpu_ok(gpuMalloc(&dx, xb), "rmsnorm x") ||
        !gpu_ok(gpuMalloc(&dw, wb), "rmsnorm w") ||
        !gpu_ok(gpuMalloc(&do_, xb), "rmsnorm o")) return 0;
    if (!gpu_ok(gpuMemcpy(dx, x, xb, gpuMemcpyHostToDevice), "rmsnorm x h2d") ||
        !gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm w h2d")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        (float *)do_, (const float *)dx, (const float *)dw, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm kernel") ||
        !gpu_ok(gpuDeviceSynchronize(), "rmsnorm sync")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    gpuMemcpy(out, do_, xb, gpuMemcpyDeviceToHost);
    gpuFree(dx); gpuFree(dw); gpuFree(do_);
    return 1;
}

/* Device-native batched rmsnorm: all pointers device-resident, no H2D/D2H/sync.
 * Used from model_forward_gpu() / model_forward_prefill_gpu() pipeline paths. */
extern "C" int
picolm_gpu_rmsnorm_batched_dev(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched dev kernel")) return 0;
    return 1;
}

/* Host-side batched rmsnorm: takes host pointers, does H2D/D2H/sync.
 /* Used from ssm_forward() QK-norm path and prefill attention QK-norm.
 * Launches the kernel directly on ctx->stream.
 * x and out must be device-accessible.
 * weight: if it's a device pointer (from gpu_upload_f32), used directly.
 *   If it's a host pointer (from dequantize_row), uploaded once and cached. */
extern "C" int
picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                            int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    const void *w_dev = weight;
    gpu_mutex_lock();
    int cached = 0;
    for (int i = 0; i < ctx->rmsnorm_w_n; i++) {
        if (ctx->rmsnorm_w_keys[i] == weight) { w_dev = ctx->rmsnorm_w_dev[i]; cached = 1; break; }
    }
    if (!cached) {
        if ((uintptr_t)weight < 1024ull * 1024 * 1024) {
            /* Host pointer: upload to device and cache */
            size_t wb = dim * sizeof(float);
            if (ctx->rmsnorm_w_n >= 64) { gpu_mutex_unlock(); return 0; }
            void *dw = NULL;
            if (!gpu_ok(gpuMalloc(&dw, wb), "rmsnorm_b w")) { gpu_mutex_unlock(); return 0; }
            gpuDeviceSynchronize();
            if (!gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm_b w h2d")) {
                gpuFree(dw); gpu_mutex_unlock(); return 0;
            }
            ctx->rmsnorm_w_keys[ctx->rmsnorm_w_n] = weight;
            ctx->rmsnorm_w_dev[ctx->rmsnorm_w_n] = dw;
            ctx->rmsnorm_w_n++;
            w_dev = dw;
        }
    }
    gpu_mutex_unlock();

    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, (const float *)w_dev, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched kernel")) return 0;
    return 1;
}

/* Capability query: does this backend support LayerNorm? */
extern "C" int
picolm_gpu_has_layernorm(int device) {
    return 1; /* CUDA/HIP: LayerNorm kernel always available */
}

/* Capability query: push descriptors (Vulkan-only, stub for CUDA/HIP) */
extern "C" int
picolm_gpu_has_push_descriptors(void) {
    return 1; /* CUDA/HIP: always has "push" capability (kernel args) */
}

/* Timestamp write (Vulkan-only, stub for CUDA/HIP) */
extern "C" void
picolm_gpu_ts_write(const char *label) {
    (void)label; /* CUDA/HIP: no Vulkan-style timestamp writes */
}

/* GELU activation: inplace gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
extern "C" int
picolm_gpu_gelu_dev(float *x_dev, size_t n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = (int)((n + n_threads - 1) / n_threads);
    if (n_blocks > 65535) n_blocks = 65535;
    picolm_gpu_gelu_inplace_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(x_dev, n);
    if (!gpu_ok(gpuGetLastError(), "gelu inplace kernel")) return 0;
    return 1;
}

/* Logits D2H via mapped memory (Vulkan) or plain D2H (CUDA/HIP). */
extern "C" int
picolm_gpu_logits_d2h_mapped(float *dst, size_t bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx) || !dst || !ctx->pipe_logits) return 0;
    if (!gpu_ok(gpuDeviceSynchronize(), "logits sync")) return 0;
    if (!gpu_ok(gpuMemcpy(dst, ctx->pipe_logits, bytes, gpuMemcpyDeviceToHost), "logits d2h")) return 0;
    return 1;
}

/* Device-native LayerNorm: single token, device pointers, no sync. */
extern "C" int
picolm_gpu_layernorm_dev(float *out, const float *x, const float *weight,
                          const float *bias, int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_layernorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        out, x, weight, bias, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "layernorm dev kernel")) return 0;
    return 1;
}

/* Device-native batched LayerNorm: all device pointers, no H2D/D2H/sync. */
extern "C" int
picolm_gpu_layernorm_batched_dev(float *out, const float *x, const float *weight,
                                  const float *bias, int dim, float eps,
                                  int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_layernorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, weight, bias, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "layernorm batched dev kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                       const float *cos_tbl, const float *sin_tbl,
                       int half_dim, int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (rope_type < 0) return 1; /* no RoPE (GPT-2 learned pos embd) */

    int total = n_heads * head_dim;
    int n_threads = 128;
    int n_blocks = min((total + n_threads - 1) / n_threads, 128);
    picolm_gpu_rope_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl, sin_tbl, half_dim, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope kernel")) return 0;
    return 1;
}

/* Batched RoPE: cos_tbl_base/sin_tbl_base are the UNOFFSET [max_seq_len][half_dim]
 * base pointers. Each row computes its own position as start_pos + row. */
extern "C" int
picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                               const float *cos_tbl_base, const float *sin_tbl_base,
                               int half_dim, int start_pos, int S,
                               int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;
    if (rope_type < 0) return 1; /* no RoPE (GPT-2 learned pos embd) */

    int total = S * n_heads * head_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    picolm_gpu_rope_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl_base, sin_tbl_base, half_dim, start_pos, S, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_residual_add(float *out, const float *a, const float *b,
                         int n, int dim, int stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int n_threads = 256;
    int n_blocks = min((n * dim + n_threads - 1) / n_threads, 256);
    picolm_gpu_residual_add_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out, a, b, n, dim, stride);
    if (!gpu_ok(gpuGetLastError(), "residual_add kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_dev(const float *raw_dev, float *out_q_dev,
                                float *out_g_dev, int n_heads, int head_dim,
                                int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n_heads + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_dev(float *out_dev, const float *gate_dev,
                            int n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_batched_dev(const float *raw_dev, float *out_q_dev,
                                        float *out_g_dev, int n_heads,
                                        int head_dim, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n_heads * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim, S);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave_batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_batched_dev(float *out_dev, const float *gate_dev,
                                   int n, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n, S);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul_batched kernel")) return 0;
    return 1;
}

/* Standalone device-native host wrapper. picolm_silu_mul itself was
 * previously only ever launched inline inside picolm_gpu_expert_mlp()
 * (which does its own H2D/D2H); the Phase 2 pipeline needs to call it
 * on already-device-resident gate/up buffers between matmul_dev calls,
 * with no transfer and no per-call sync (same stream-ordering argument
 * as picolm_gpu_matmul_dev / picolm_gpu_attention_decode_dev). Result is
 * written in place into gate_dev. */
extern "C" int
picolm_gpu_silu_mul_dev(float *gate_dev, const float *up_dev, size_t n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(
        gate_dev, up_dev, n);
    if (!gpu_ok(gpuGetLastError(), "silu_mul (dev)")) return 0;
    return 1;
}

/* GELU-multiply: gate *= gelu(up), device-native. Used by GPT-2/Gemma3n arch. */
__global__ void picolm_gelu_mul(float *gate, const float *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float u = up[i];
        float g = gate[i];
        float u3 = u * u * u;
        float t = tanhf(0.79788456f * (u + 0.044715f * u3));
        gate[i] = g * 0.5f * u * (1.0f + t);
    }
}

extern "C" int
picolm_gpu_gelu_mul_dev(float *gate_dev, const float *up_dev, size_t n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_gelu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(
        gate_dev, up_dev, n);
    if (!gpu_ok(gpuGetLastError(), "gelu_mul (dev)")) return 0;
    return 1;
}

/* Vulkan command batch wrappers -- on CUDA/HIP, batch_end syncs the stream
 * so that benchmark timing in _prefill_gpu_ubatch captures actual GPU execution
 * time rather than just command recording overhead.
 * Vulkan uses explicit command buffers that need begin/end + fence sync. */
extern "C" int picolm_gpu_batch_begin(int device) { (void)device; return 1; }
extern "C" int picolm_gpu_batch_end(int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (ctx) gpuDeviceSynchronize();
    return 1;
}

/* D2H via mapped memory -- on CUDA/HIP, just use regular D2H memcpy. */
extern "C" int
picolm_gpu_d2h_via_mapped(void *dst, const void *src, size_t bytes, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyDeviceToHost, ctx->stream), "d2h mapped") &&
           gpu_ok(gpuDeviceSynchronize(), "d2h mapped sync");
}

/* Pipeline logits buffer allocation -- no-op on CUDA/HIP (uses shared ctx->y). */
extern "C" int
picolm_gpu_pipeline_logits_alloc(size_t bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx) return 0;
    (void)bytes; /* ctx->y auto-resizes on demand */
    return 1;
}

/* KV cache flush to CPU: copy positions [0, n_pos) of every layer
 * ([layer][pos] rows, layers max_seq rows apart). n_pos outside 1..max_seq
 * copies the whole cache. This used to copy all max_seq rows every time:
 * 1.5 GB per prefill for an 11B model at 8K context. */
extern "C" int
picolm_gpu_kv_flush_to_cpu(uint8_t *cpu_k, uint8_t *cpu_v,
    int n_attn_ord, int max_seq, int n_pos,
    size_t row_sz_k, size_t row_sz_v, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (!cpu_k || !cpu_v) return 0;
    if (n_pos < 1 || n_pos > max_seq) n_pos = max_seq;
    /* One strided copy per cache: n_attn_ord rows of n_pos * row_sz bytes,
     * max_seq * row_sz apart on both sides. */
    return gpu_ok(gpuMemcpy2D(cpu_k, (size_t)max_seq * row_sz_k, g_kv_k_dev[device], (size_t)max_seq * row_sz_k,
                              (size_t)n_pos * row_sz_k, (size_t)n_attn_ord, gpuMemcpyDeviceToHost), "kv flush K D2H") &&
           gpu_ok(gpuMemcpy2D(cpu_v, (size_t)max_seq * row_sz_v, g_kv_v_dev[device], (size_t)max_seq * row_sz_v,
                              (size_t)n_pos * row_sz_v, (size_t)n_attn_ord, gpuMemcpyDeviceToHost), "kv flush V D2H");
}

/* Upload CPU KV cache to GPU KV cache (H2D).
 * Used for GPT-2 CPU prefill -> GPU decode path: CPU prefill writes to
 * CPU KV cache, then this uploads to GPU KV cache for GPU decode. */
extern "C" int
picolm_gpu_kv_upload_from_cpu(const uint8_t *cpu_k, const uint8_t *cpu_v,
    int n_attn_ord, int max_seq_len, int n_pos,
    size_t row_sz_k, size_t row_sz_v, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (!cpu_k || !cpu_v) return 0;
    /* Copy full layers (all positions), not just the written positions.
     * CPU KV cache layout: [layer][max_seq_len][kv_dim]. Each layer is
     * contiguous. GPU KV cache has the same layout. Copy full layers to
     * maintain alignment. Unwritten positions are zero in both buffers. */
    size_t upload_bytes_k = (size_t)n_attn_ord * max_seq_len * row_sz_k;
    size_t upload_bytes_v = (size_t)n_attn_ord * max_seq_len * row_sz_v;
    if (!gpu_ok(gpuMemcpy(g_kv_k_dev[device], cpu_k, upload_bytes_k, gpuMemcpyHostToDevice),
                "kv upload K H2D")) return 0;
    if (!gpu_ok(gpuMemcpy(g_kv_v_dev[device], cpu_v, upload_bytes_v, gpuMemcpyHostToDevice),
                "kv upload V H2D")) return 0;
    return 1;
}

/* The single sync point for the whole model_forward_gpu() pass: call
 * this exactly once, after the last device-native op (typically the
 * final rmsnorm), before reading anything back via D2H. Every _dev
 * primitive above is launched on ctx->stream with no internal sync, so
 * this is what actually guarantees all of it has completed. */
extern "C" int
picolm_gpu_sync(int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline sync");
}

/* Synchronous memcpy wrapper for model.c (C file, no CUDA types).
 * dir: 1 = H2D, -1 = D2H, 0 = D2D. Returns 1 on success.
 * Uses ctx->stream for the actual transfer to avoid multi-engine
 * reordering with default stream on sm_121 (GB10/CUDA 13). */
/* Strided form of picolm_gpu_memcpy: `rows` rows of `width` bytes, `dpitch` /
 * `spitch` bytes apart. Same dir convention and the same ordering: D2H waits
 * for the stream first, H2D/D2D are queued on it. */
extern "C" int
picolm_gpu_memcpy_2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                     size_t width, size_t rows, int dir, int device) {
    if (!dst || !src || width < 1 || rows < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    gpuMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                          (dir < 0) ? gpuMemcpyDeviceToHost :
                                      gpuMemcpyDeviceToDevice;
    if (dir < 0) {
        if (!gpu_ok(gpuStreamSynchronize(ctx->stream), "memcpy_2d D2H stream sync")) return 0;
        return gpu_ok(gpuMemcpy2D(dst, dpitch, src, spitch, width, rows, kind), "memcpy_2d D2H");
    }
    return gpu_ok(gpuMemcpy2DAsync(dst, dpitch, src, spitch, width, rows, kind, ctx->stream), "memcpy_2d async");
}

extern "C" int
picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    gpuMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                          (dir < 0) ? gpuMemcpyDeviceToHost :
                                      gpuMemcpyDeviceToDevice;

    /* Use synchronous copy for D2H to avoid stream issues */
    if (dir < 0) {
        gpuError_t err = gpuMemcpy(dst, src, bytes, kind);
        if (!gpu_ok(err, "pipeline memcpy D2H sync")) return 0;
        return 1;
    }
    gpuError_t err = gpuMemcpyAsync(dst, src, bytes, kind, ctx->stream);
    if (!gpu_ok(err, "pipeline memcpy async")) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline memcpy sync");
}

/* Async variant: copies on ctx->stream without synchronizing.
 * Caller is responsible for ensuring ordering via subsequent
 * kernel launches or explicit syncs. Only safe for D2D copies
 * and for cases where the caller knows the stream ordering is sufficient. */
extern "C" int
picolm_gpu_memcpy_async(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    gpuMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                          (dir < 0) ? gpuMemcpyDeviceToHost :
                                      gpuMemcpyDeviceToDevice;
    return gpu_ok(gpuMemcpyAsync(dst, src, bytes, kind, ctx->stream), "pipeline memcpy async");
}
extern "C"
static int picolm_gpu_ssm_conv1d_batch_dev_impl(float *od, float *sd, const float *id, const float *wd,
    int cd, int dc, int nt, int dev, int stride) {
    if(cd<1||dc<1||nt<1) return 0;
    if(dc>PICOLM_SSM_CONV_MAX_D_CONV) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int nb=(cd+255)/256;
    /* Tokens CANNOT be split across blocks here: the caller aliases input and
     * output (conv1d_batch_dev(bxb, ..., bxb)) and the token walk must stay in
     * order. Only channels parallelise. */
    picolm_gpu_ssm_conv1d_batch_kernel<<<nb,256,0,ctx->stream>>>(od,sd,id,wd,cd,dc,nt,stride);
    return gpu_ok(gpuGetLastError(),"conv1d batch dev");
}
extern "C" int picolm_gpu_ssm_conv1d_batch_dev(float *od, float *sd, const float *id, const float *wd,
    int cd, int dc, int nt, int dev, int stride) {
    int _ps = picolm_gpu_prof_slot("ssm_conv1d");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_conv1d_batch_dev_impl(od, sd, id, wd, cd, dc, nt, dev, stride);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

extern "C"
static int picolm_gpu_ssm_l2norm_batch_dev_impl(float *xd, int hd, int nh, int nt, int ts, float eps, float es, int dev) {
    if(hd<1||nh<1||hd>256||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_l2norm_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(xd,hd,nh,nt,ts,eps,es);
    return gpu_ok(gpuGetLastError(),"l2norm batch dev");
}
extern "C" int picolm_gpu_ssm_l2norm_batch_dev(float *xd, int hd, int nh, int nt, int ts, float eps, float es, int dev) {
    int _ps = picolm_gpu_prof_slot("ssm_l2norm");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_l2norm_batch_dev_impl(xd, hd, nh, nt, ts, eps, es, dev);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

extern "C"
/* Forward declaration: defined later in this file */
__global__ void picolm_gpu_ssm_head_permute_batch_kernel(float *dst, const float *src,
    const int *head_map, int head_dim, int n_heads, int n_tokens,
    int src_stride, int dst_stride);

/* IMMA-accelerated vecdot: x[n_tokens][dim] @ weights[n_heads][dim]^T -> y[n_tokens][n_heads].
 * Mathematically a matmul with S=n_tokens, I=dim, O=n_heads.
 * For Q8_0 weights: quantize x to Q8_0 on-device, dispatch IMMA/tiled/scalar kernel.
 * For F32/BF16 weights: pre-quantize to Q8_0 on first use, cache in static buffer,
 *   then dispatch IMMA/tiled/scalar kernel.
 * For Q4_0: fall back to the serial vecdot kernel.
 * When head_map != NULL, post-permute output from GGUF head order to natural head order. */
static int picolm_gpu_ssm_vecdot_batch_dev_impl(float *od, const float *xd, const void *wd, gguf_type_t qt,
    int dim, int nvh, int nt, int rb, const int *hm, int dev, int in_stride, int out_stride) {
    if(nvh<=0||dim<=0||nt<=0) return 0;
    if(dim>PICOLM_SSM_VECDOT_MAX_DIM) return 0;
    if(!wd) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;

    /* IMMA fast path for Q8_0, F32, BF16 weights */
    if (qt == GGUF_TYPE_Q8_0 && !getenv("PICOLM_FORCE_F32_MATMUL") && !getenv("PICOLM_NO_VECDOT_IMMA") && !getenv("PICOLM_NO_IMMA")) {
        int n_blocks = dim / 32;
        if (n_blocks >= 1 && dim % 32 == 0) {
            void *wq8 = (void *)wd; /* Q8_0: use directly, others: use pre-quantized copy */
            int wq8_rb = rb; /* Q8_0: use original row_bytes */

            /* For F32/BF16: pre-quantize weights to Q8_0 once per (wd, qt, dim, nvh) */
            if (qt != GGUF_TYPE_Q8_0) {
                /* Static cache: one pre-quantized Q8_0 copy per device context.
                 * The weights don't change between calls, so this is a one-time cost. */
                static float *vd_wq8_cache = NULL;
                static const void *vd_wq8_key = NULL;
                static int vd_wq8_qt = -1;
                static size_t vd_wq8_cap = 0;
                size_t q8_rb = (size_t)n_blocks * GPU_BLOCK_Q8_0_SIZE;
                size_t q8_bytes = (size_t)nvh * q8_rb;

                if (vd_wq8_key != wd || vd_wq8_qt != (int)qt || q8_bytes > vd_wq8_cap) {
                    if (!reserve(&vd_wq8_cache, &vd_wq8_cap, q8_bytes)) goto vecdot_serial;
                    vd_wq8_key = wd;
                    vd_wq8_qt = (int)qt;
                    /* Quantize F32/BF16 weights to Q8_0 on-device */
                    int src_stride = (qt == 30) ? rb / (int)sizeof(uint16_t) : rb / (int)sizeof(float);
                    picolm_quantize_weights_to_q8_0<<<(unsigned)nvh, 32, 0, ctx->stream>>>(
                        (void *)vd_wq8_cache, (const float *)wd, qt, dim, nvh, src_stride, q8_rb);
                    if (!gpu_ok(gpuGetLastError(), "vecdot wq8 quantize")) goto vecdot_serial;
                }
                wq8 = (void *)vd_wq8_cache;
                wq8_rb = (int)q8_rb;
            }

            /* Quantize strided input to Q8_0 */
            int nt_padded = (nt + 15) & ~15;
            size_t xq_bytes = (size_t)nt_padded * dim;
            size_t xd_bytes = (size_t)nt_padded * n_blocks * sizeof(float);
            if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
                !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) goto vecdot_serial;

            dim3 q_grid((unsigned)n_blocks, (unsigned)nt);
            if (in_stride > 0 && in_stride != dim) {
                picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                    ctx->q8_xq, ctx->q8_xd, xd, dim, nt, in_stride);
            } else {
                picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                    ctx->q8_xq, ctx->q8_xd, xd, dim, nt);
            }
            if (!gpu_ok(gpuGetLastError(), "vecdot q8 quantize")) return 0;

            int ys = out_stride > 0 ? out_stride : nvh;
            float *y_target = od;
            int y_tmp_needed = 0;
            if (hm) {
                size_t tmp_bytes = (size_t)nt * nvh * sizeof(float);
                if (reserve(&ctx->gate, &ctx->gate_cap, tmp_bytes)) {
                    y_target = ctx->gate;
                    ys = nvh;
                    y_tmp_needed = 1;
                }
            }

            /* Dispatch: IMMA (NVIDIA Turing+), else tiled, else scalar */
            if (ctx->has_imma && nt >= 16 && nvh >= 8) {
                gpu_dispatch_print("vecdot_imma_dev");
                dim3 grid((unsigned)((nvh + 7) / 8), (unsigned)((nt + 15) / 16));
                picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            } else if (nt >= Q8_TILE_S && wq8_rb + 2048 <= 49152) {
                gpu_dispatch_print("vecdot_tiled_dev");
                dim3 grid((unsigned)nvh, (unsigned)((nt + Q8_TILE_S - 1) / Q8_TILE_S));
                picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)wq8_rb, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            } else {
                gpu_dispatch_print("vecdot_scalar_dev");
                dim3 grid((unsigned)nvh, (unsigned)nt);
                picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            }
            if (!gpu_ok(gpuGetLastError(), "vecdot q8 matmul")) return 0;

            if (y_tmp_needed) {
                picolm_gpu_ssm_head_permute_batch_kernel<<<
                    dim3((unsigned)nvh, (unsigned)nt), 1, 0, ctx->stream>>>(
                    od, y_target, hm, 1, nvh, nt, nvh, ys);
                if (!gpu_ok(gpuGetLastError(), "vecdot head permute")) return 0;
            }
            return 1;
        }
    }

vecdot_serial:
    /* Serial vecdot fallback for Q4_0 weights, misaligned dims, or OOM */
    gpu_dispatch_print("vecdot_serial_dev");
    /* Warp-per-(head,token). PICOLM_NO_VECDOT_BATCH_WARP=1 for A/B. */
    static int no_vbw = -1;
    if (no_vbw < 0) no_vbw = getenv("PICOLM_NO_VECDOT_BATCH_WARP") ? 1 : 0;
    if (!no_vbw && (qt == 0 || qt == 2 || qt == 8 || qt == 30)) {
        long long vtot = (long long)nvh * nt;
        unsigned vblk = (unsigned)((vtot + VECB_WARPS - 1) / VECB_WARPS);
        picolm_ssm_vecdot_batch_warp_kernel<<<vblk, 32 * VECB_WARPS, 0, ctx->stream>>>(
            od, xd, (const void *)wd, qt, dim, nvh, nt, rb, hm, in_stride, out_stride);
        if (gpu_ok(gpuGetLastError(), "ssm vecdot batch warp")) return 1;
        gpuGetLastError();
    }
    picolm_ssm_vecdot_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),256,0,ctx->stream>>>(
        od,xd,wd,qt,dim,nvh,nt,rb,hm,in_stride,out_stride);
    return gpu_ok(gpuGetLastError(),"vecdot batch dev");
}
extern "C" int picolm_gpu_ssm_vecdot_batch_dev(float *od, const float *xd, const void *wd, gguf_type_t qt,
    int dim, int nvh, int nt, int rb, const int *hm, int dev, int in_stride, int out_stride) {
    int _ps = picolm_gpu_prof_slot("ssm_vecdot_b");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_vecdot_batch_dev_impl(od, xd, wd, qt, dim, nvh, nt, rb, hm, dev, in_stride, out_stride);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

extern "C"
static int picolm_gpu_ssm_prefill_gated_norm_dev_impl(float *od, const float *zd, const float *nd,
    int hd, int nh, int nt, float eps, int so_stride, int z_stride, int dev) {
    if(hd<1||nh<1||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_prefill_gated_norm_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(od,zd,nd,hd,nh,nt,eps,so_stride,z_stride);
    return gpu_ok(gpuGetLastError(),"gated norm dev");
}
extern "C" int picolm_gpu_ssm_prefill_gated_norm_dev(float *od, const float *zd, const float *nd,
    int hd, int nh, int nt, float eps, int so_stride, int z_stride, int dev) {
    int _ps = picolm_gpu_prof_slot("ssm_gnorm");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_prefill_gated_norm_dev_impl(od, zd, nd, hd, nh, nt, eps, so_stride, z_stride, dev);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

__global__ void
picolm_gpu_ssm_head_permute_batch_kernel(float *dst, const float *src,
                                          const int *head_map,
                                          int head_dim, int n_heads, int n_tokens,
                                          int src_stride, int dst_stride) {
    int h = blockIdx.x; int t = blockIdx.y;
    if (h >= n_heads || t >= n_tokens) return;
    int gh = head_map ? head_map[h] : h;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        dst[(size_t)t * dst_stride + h * head_dim + d] = src[(size_t)t * src_stride + gh * head_dim + d];
}

extern "C"
static int picolm_gpu_ssm_head_permute_batch_dev_impl(float *dd, const float *sd, const int *hm,
    int hd, int nh, int nt, int ss, int ds, int dev) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_gpu_ssm_head_permute_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),128,0,ctx->stream>>>(dd,sd,hm,hd,nh,nt,ss,ds);
    return gpu_ok(gpuGetLastError(),"head permute batch dev");
}
extern "C" int picolm_gpu_ssm_head_permute_batch_dev(float *dd, const float *sd, const int *hm,
    int hd, int nh, int nt, int ss, int ds, int dev) {
    int _ps = picolm_gpu_prof_slot("ssm_permute_b");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_head_permute_batch_dev_impl(dd, sd, hm, hd, nh, nt, ss, ds, dev);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

__global__ void
/* Gate/beta post-process: outputs gate_log (log-space gate value) directly.
 * Previously this kernel did expf(gate) and the recurrence did logf(expf(gate))
 * -- a useless exp/log round-trip that compounded floating-point error.
 * Now it outputs gate_log = softplus(alpha + dt_w) * a_w, matching the CPU path. */
picolm_ssm_gate_beta_batch_kernel(float *ge, float *be,
    const float *ai, const float *bi, const float *dw, const float *aw,
    int nvh, int nt, int stride) {
    int h = blockIdx.x, t = blockIdx.y;
    if (h >= nvh || t >= nt) return;
    if (threadIdx.x != 0) return;
    int s = stride > 0 ? stride : nvh;
    float a = ai[t*s+h] + dw[h];
    float sp = (a>20.0f)?a:(a<-20.0f)?expf(a):logf(1.0f+expf(a));
    ge[t*s+h] = sp * aw[h];
    be[t*s+h] = 1.0f/(1.0f+expf(-bi[t*s+h]));
}

extern "C"
static int picolm_gpu_ssm_gate_beta_batch_dev_impl(float *ge, float *be, const float *ai, const float *bi,
    const float *dw, const float *aw, int nvh, int nt, int dev, int stride) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_ssm_gate_beta_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),1,0,ctx->stream>>>(ge,be,ai,bi,dw,aw,nvh,nt,stride);
    return gpu_ok(gpuGetLastError(),"gate beta batch dev");
}
extern "C" int picolm_gpu_ssm_gate_beta_batch_dev(float *ge, float *be, const float *ai, const float *bi,
    const float *dw, const float *aw, int nvh, int nt, int dev, int stride) {
    int _ps = picolm_gpu_prof_slot("ssm_gate_beta");
    picolm_gpu_prof_begin(dev);
    int _rc = picolm_gpu_ssm_gate_beta_batch_dev_impl(ge, be, ai, bi, dw, aw, nvh, nt, dev, stride);
    picolm_gpu_prof_end(_ps);
    return _rc;
}

extern "C"
int picolm_gpu_expert_mlp_dev(picolm_gpu_tensor_t *g, picolm_gpu_tensor_t *u, picolm_gpu_tensor_t *d,
    float *yd, const float *xd, int S, int x_stride, int y_stride, int dev) {
    if(!g||!u||!d||!xd||!yd||S<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int D=g->I, I=g->O;
    if(!reserve(&ctx->gate,&ctx->gate_cap,(size_t)S*I*sizeof(float))||
       !reserve(&ctx->up,&ctx->up_cap,(size_t)S*I*sizeof(float))) return 0;
    /* F32 fallback: always use picolm_gpu_matmul_dev for non-Q8_0 weights
     * (BF16, F32, etc.). For Q8_0 weights, the int8-MAC path is used below. */
    if (g->qtype != GGUF_TYPE_Q8_0 || u->qtype != GGUF_TYPE_Q8_0 || d->qtype != GGUF_TYPE_Q8_0 ||
        getenv("PICOLM_FORCE_F32_MLP") || getenv("PICOLM_FORCE_F32_MATMUL")) {
        if (!picolm_gpu_matmul_dev(g, ctx->gate, xd, S, dev, I, x_stride)) return 0;
        if (!picolm_gpu_matmul_dev(u, ctx->up, xd, S, dev, I, x_stride)) return 0;
        picolm_silu_mul<<<(unsigned)((S*I+255)/256),256,0,ctx->stream>>>(ctx->gate, ctx->up, S*I);
        if (!gpu_ok(gpuGetLastError(), "expert silu")) return 0;
        int ys = y_stride > 0 ? y_stride : D;
        if (!picolm_gpu_matmul_dev(d, yd, ctx->gate, S, dev, ys, I)) return 0;
        return 1;
    }
    /* Q8_0 fast path: quantize F32 input to Q8_0, then int8 MAC */
    int d_blocks = D / 32;
    int i_blocks = I / 32;
    if (d_blocks < 1 || i_blocks < 1) return 0;

    int S_padded = (S + 15) & ~15;
    size_t xq_d = (size_t)S_padded * D;
    size_t xd_d = (size_t)S_padded * d_blocks * sizeof(float);
    size_t xq_i = (size_t)S_padded * I;
    size_t xd_i = (size_t)S_padded * i_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_d > xq_i ? xq_d : xq_i) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_d > xd_i ? xd_d : xd_i)) return 0;

    /* Quantize F32 input [D*S] to Q8_0 */
    if (x_stride > 0 && x_stride != D) {
        picolm_quantize_q8_0_strided<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S, x_stride);
    } else {
        picolm_quantize_q8_0<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize input (dev)")) return 0;

    /* gate = q8_q8_tiled(input_q8, gate_weights) */
    /* gate = q8_q8 matmul(input_q8, gate_weights) */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    } else if (S > 1 && g->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)g->row_bytes,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 gate (dev)")) return 0;

    /* up = q8_q8 matmul(input_q8, up_weights) */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    } else if (S > 1 && u->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)u->row_bytes,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 up (dev)")) return 0;

    /* silu(gate) * up -> ctx->gate */
    picolm_silu_mul<<<(unsigned)((S*I+255)/256),256,0,ctx->stream>>>(ctx->gate, ctx->up, S*I);
    if (!gpu_ok(gpuGetLastError(), "expert silu (dev)")) return 0;

    /* Quantize F32 hidden [I*S] to Q8_0 for down */
    picolm_quantize_q8_0<<<dim3((unsigned)i_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->gate, I, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize hidden (dev)")) return 0;

    /* y = q8_q8 matmul(hidden_q8, down_weights) */
    int ys = y_stride > 0 ? y_stride : D;
    if (ctx->has_imma && S >= 16 && D >= 8) {
        dim3 grid((unsigned)((D + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    } else if (S > 1 && d->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)D, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)d->row_bytes,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)D,(unsigned)S),256,0,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    }
    return gpu_ok(gpuGetLastError(), "expert MLP dev");
}

extern "C"
/* Row-split factor for ssm_chunk_matvec_kernel. Grid becomes
 * (n_v_heads, MV_SPLIT) so a 48-head model fills more than 48 SMs, and each
 * block does cs/MV_SPLIT rows instead of all cs. Cheap to raise: every row is
 * independent. */
#define MV_SPLIT 8

int picolm_gpu_ssm_chunked_recurrence_dev(const float *conv_dev, const float *alpha_dev,
    const float *beta_dev, float *state_dev, float *xb2_dev,
    int n_tokens, int value_dim, int xb2_stride,
    int d_state, int n_k_heads, int n_v_heads,
    int head_v_dim, int repeat, int conv_dim, int cs, int device) {
    (void)conv_dim; /* conv_dim is used for offset calculations, xb2_stride for striding */
    /* Profile the SSM chunked recurrence: this file had NO prof slots at all,
     * so the whole SSM GPU prefill path was invisible while it accounted for
     * almost all of prefill's wall time on the 27B (48 SSM layers x ~29 ms,
     * against 16 attention layers costing ~50 ms in total). */
    int _ssmps = picolm_gpu_prof_slot("ssm_chunk_rec");
    picolm_gpu_prof_begin(device);
    /* This function returns early from ~16 places inside the open region.
     * Each of those would leave the profiler's open-region stack unbalanced,
     * and prof_flush_ring() then DROPS later pairs by design -- which is how
     * pf_matmul_q6_K silently stopped appearing in the profile at all while
     * matmul_q6_imma_mt_dev was demonstrably still running.
     * Every return now goes through this macro. */
    #define SSMCR_RET(_v) do { picolm_gpu_prof_end(_ssmps); return (_v); } while (0)
#ifndef PICOLM_SSM_CHUNKED_GPU_VALIDATED
    (void)conv_dev;(void)alpha_dev;(void)beta_dev;(void)state_dev;(void)xb2_dev;
    (void)n_tokens;(void)value_dim;(void)xb2_stride;(void)d_state;(void)n_k_heads;
    (void)n_v_heads;(void)head_v_dim;(void)repeat;(void)cs;(void)device;
    picolm_gpu_prof_end(_ssmps);
    SSMCR_RET(0);
#else
    if(d_state!=head_v_dim) SSMCR_RET(0);
    if(cs<=0) cs=64; if(cs>n_tokens) cs=n_tokens;
    int nc=(n_tokens+cs-1)/cs; if(nc<1) nc=1;
    int d=d_state, qk_dim=d_state*n_k_heads;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if(!ctx||!select_ctx(ctx)) SSMCR_RET(0);
    int n_threads=256;
    /* Persistent scratch buffers: must be static so they survive across
     * calls (one per SSM layer per layer loop). ssm_batch_scratch_ensure
     * is a grow-only allocator that checks *cap >= need before allocating.
     * If these were local variables, every call would re-allocate and leak. */
    static float *d_cq=NULL,*d_ck=NULL,*d_cv=NULL,*d_cb=NULL;
    static float *d_gl=NULL,*d_cg=NULL,*d_qd=NULL,*d_dm=NULL;
    static float *d_M=NULL,*d_kq=NULL,*d_ve=NULL,*d_vh=NULL;
    static float *d_sk=NULL,*d_sq=NULL,*d_co=NULL;
    static size_t caps[15]={0};
    size_t szs[15]={
        (size_t)n_k_heads*cs*d*sizeof(float),  /* cq */
        (size_t)n_k_heads*cs*d*sizeof(float),  /* ck */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cv */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cb */
        (size_t)n_v_heads*cs*sizeof(float),    /* gl */
        (size_t)n_v_heads*cs*sizeof(float),    /* cg */
        (size_t)n_v_heads*cs*sizeof(float),    /* qd */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* dm */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* M */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* kq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* ve */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* vh */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sk */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* co */
    };
    float **ptrs[15]={&d_cq,&d_ck,&d_cv,&d_cb,&d_gl,&d_cg,&d_qd,&d_dm,
        &d_M,&d_kq,&d_ve,&d_vh,&d_sk,&d_sq,&d_co};
    int ok=1;
    for(int i=0;i<15&&ok;i++){
        ok&=ssm_batch_scratch_ensure((void**)ptrs[i],&caps[i],szs[i]);
        if(ok) ok&=gpu_ok(gpuMemsetAsync(*(void**)ptrs[i],0,szs[i],ctx->stream),"scratch zero");
    }
    if(!ok) SSMCR_RET(0);
    for(int ci=0;ci<nc&&ok;ci++){
      int ca=(ci==nc-1)?(n_tokens-ci*cs):cs; if(ca<=0)break;
      int co2=ci*cs;
      { int _pk = picolm_gpu_prof_slot("gather_qk"); picolm_gpu_prof_begin(device);
              ssm_chunk_gather_qk_kernel<<<n_k_heads,n_threads,0,ctx->stream>>>((float*)d_cq,(float*)d_ck,conv_dev,co2,ca,xb2_stride,qk_dim,d_state,n_k_heads);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"gather_qk"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("gather_v"); picolm_gpu_prof_begin(device);
              ssm_chunk_gather_v_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cv,(float*)d_cb,(float*)d_gl,conv_dev,alpha_dev,beta_dev,co2,ca,xb2_stride,qk_dim,head_v_dim,n_v_heads,xb2_stride);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"gather_v"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("decay"); picolm_gpu_prof_begin(device);
              ssm_chunk_decay_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cg,(float*)d_qd,(float*)d_dm,(float*)d_gl,n_v_heads,ca,ca);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"decay"))SSMCR_RET(0);
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H decay_mask and gate_log scratch */
          float _dm[16], _gl[40], _cg[40];
          gpuDeviceSynchronize();
          gpuMemcpy(_dm, d_dm, 64, gpuMemcpyDeviceToHost);
          gpuMemcpy(_gl, d_gl, 160, gpuMemcpyDeviceToHost);
          gpuMemcpy(_cg, d_cg, 160, gpuMemcpyDeviceToHost);
          float _dm_full[1300];
          gpuMemcpy(_dm_full, d_dm, 5200, gpuMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] dm row0=[%.6f %.6f %.6f %.6f] row1=[%.6f %.6f %.6f %.6f] dm35_0=%.6f dm35_35=%.6f cg0=%.6f cg35=%.6f\n",
              _dm_full[0],_dm_full[1],_dm_full[2],_dm_full[3],
              _dm_full[36],_dm_full[37],_dm_full[38],_dm_full[39],
              _dm_full[35*ca+0], _dm_full[35*ca+35], _cg[0], _cg[35]);
      }
#endif
      { int _pk = picolm_gpu_prof_slot("masked_gemm"); picolm_gpu_prof_begin(device);
              ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_M,(const float*)d_ck,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"M_gemm"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("matvec"); picolm_gpu_prof_begin(device);
              ssm_chunk_matvec_kernel<<<dim3(n_v_heads,MV_SPLIT),n_threads,0,ctx->stream>>>((float*)d_sk,state_dev,(const float*)d_ck,n_v_heads,repeat,ca,d);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"sk_matvec"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("veff"); picolm_gpu_prof_begin(device);
              ssm_chunk_veff_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_ve,(const float*)d_cv,(const float*)d_sk,(const float*)d_qd,(const float*)d_cb,n_v_heads,ca,d);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"veff"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("trisolve"); picolm_gpu_prof_begin(device);
              {int tr=d<1024?d:1024; ssm_chunk_trisolve_kernel<<<n_v_heads,tr,0,ctx->stream>>>((float*)d_vh,(const float*)d_ve,(const float*)d_M,(const float*)d_cb,n_v_heads,ca,d);}
      if(!gpu_ok(gpuGetLastError(),"trisolve"))SSMCR_RET(0);
        picolm_gpu_prof_end(_pk); }
      ssm_chunk_matvec_kernel<<<dim3(n_v_heads,MV_SPLIT),n_threads,0,ctx->stream>>>((float*)d_sq,state_dev,(const float*)d_cq,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"sq_matvec"))SSMCR_RET(0);
      ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_kq,(const float*)d_cq,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"kq_gemm"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("chunk_out"); picolm_gpu_prof_begin(device);
              ssm_chunk_output_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_co,(const float*)d_sq,(const float*)d_qd,(const float*)d_kq,(const float*)d_vh,n_v_heads,ca,d);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"output"))SSMCR_RET(0);
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H v_eff[0][0..3] and v_hat[0][0..3] for head 0 */
          float _ve[4], _vh[4], _co[4];
          gpuDeviceSynchronize();
          gpuMemcpy(_ve, d_ve, 16, gpuMemcpyDeviceToHost);
          gpuMemcpy(_vh, d_vh, 16, gpuMemcpyDeviceToHost);
          gpuMemcpy(_co, d_co, 16, gpuMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] GPU ve[0][0..3]={%.6f,%.6f,%.6f,%.6f} vh[0][0..3]={%.6f,%.6f,%.6f,%.6f} co[0][0..3]={%.6f,%.6f,%.6f,%.6f}\n",
              _ve[0],_ve[1],_ve[2],_ve[3], _vh[0],_vh[1],_vh[2],_vh[3], _co[0],_co[1],_co[2],_co[3]);
      }
#endif
      { int _pk = picolm_gpu_prof_slot("scatter"); picolm_gpu_prof_begin(device);
              ssm_chunk_scatter_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>(xb2_dev,(const float*)d_co,co2,ca,xb2_stride,head_v_dim,n_v_heads);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"scatter"))SSMCR_RET(0);
      { int _pk = picolm_gpu_prof_slot("state_up"); picolm_gpu_prof_begin(device);
              ssm_chunk_state_update_kernel<<<dim3(n_v_heads,MV_SPLIT),n_threads,(unsigned)(((ca<1024?ca:1024)+4)*sizeof(float)),ctx->stream>>>(state_dev,(const float*)d_vh,(const float*)d_ck,(const float*)d_cg,n_v_heads,repeat,ca,d);
        picolm_gpu_prof_end(_pk); }
      if(!gpu_ok(gpuGetLastError(),"state_up"))SSMCR_RET(0);
    }
    picolm_gpu_prof_end(_ssmps);
    return ok;
#endif
}

/* ================================================================
 * PICOLM_GPU_PROF: per-kernel-class timing for the GPU pipeline
 * ================================================================
 *
 * Everything in the _dev pipeline runs on ctx->stream with no internal sync,
 * so a pair of CUDA events around each launch class and reading the deltas at
 * the single end-of-pass sync gives correct attribution with NO extra
 * synchronisation on the hot path (a per-launch gpuEventSynchronize would
 * serialise the pipeline and measure something else entirely).
 *
 * Usage: PICOLM_GPU_PROF=1 PICOLM_GPU=1 ./picolm.exe ... ; the report is
 * printed at exit. Costs two event records per launch when enabled, nothing
 * when not.
 */
#define PROF_MAX_SLOTS 48
#define PROF_MAX_EVENTS 262144

typedef struct {
    const char *name;
    int    slot_of_event_start;   /* index into the start-event ring */
    double total_ms;
    long   calls;
} prof_slot_t;

static prof_slot_t g_prof[PROF_MAX_SLOTS];
static int   g_prof_n = 0;
static int   g_prof_on = -1;
static gpuEvent_t g_prof_ev[PROF_MAX_EVENTS];
static int   g_prof_ev_n = 0;
static int   g_prof_slot_log[PROF_MAX_EVENTS];   /* slot for event pair i */
static int   g_prof_pair_start[PROF_MAX_EVENTS]; /* START event index for that pair */
static int   g_prof_depth = 0;
static gpuStream_t g_prof_stream = (gpuStream_t)0;

static void gpu_event_rec_stop(void) {
    gpuEventRecord(g_prof_ev[g_prof_ev_n], g_prof_stream);
}

static int prof_enabled(void) {
    if (g_prof_on < 0) {
        g_prof_on = getenv("PICOLM_GPU_PROF") ? 1 : 0;
        if (g_prof_on) {
            for (int i = 0; i < PROF_MAX_EVENTS; i++) {
                gpuEventCreate(&g_prof_ev[i]);
                /* Both pairing arrays are read by prof_flush_ring() before a
                 * slot is ever written, so an uninitialised entry can win the
                 * `st < e` sanity check and get paired with a random start
                 * event. That produced a bogus "gemv_q8_0, 2165 ms/call". */
                g_prof_slot_log[i]    = -1;
                g_prof_pair_start[i]  = -1;
            }
        }
    }
    return g_prof_on;
}

extern "C" int picolm_gpu_prof_slot(const char *name) {
    if (!prof_enabled()) return -1;
    for (int i = 0; i < g_prof_n; i++)
        if (g_prof[i].name == name) return i;   /* interned string literals */
    if (g_prof_n >= PROF_MAX_SLOTS) return -1;
    g_prof[g_prof_n].name = name;
    g_prof[g_prof_n].total_ms = 0.0;
    g_prof[g_prof_n].calls = 0;
    return g_prof_n++;
}

/* Record a start event. Nested regions are not supported; the GPU pipeline is
 * a flat sequence of launches, so a simple depth counter is enough to keep
 * begin/end paired. */
/* Nesting: regions DO nest here -- the whole-pass pair wraps every per-class
 * pair. The original flat i/i+1 pairing silently mis-attributed everything
 * (it reported the outer region's total against the inner slots and zero for
 * all of them). Each open region now records its start event index and the
 * matching end closes exactly that pair, so an outer region's delta naturally
 * includes its children. */
#define PROF_MAX_DEPTH 8
static int g_prof_open[PROF_MAX_DEPTH];
static int g_prof_open_n = 0;

/* Fold every completed pair in the ring into g_prof[] and reset the ring.
 * Called when the ring fills and once at exit. */
static void prof_flush_ring(void) {
    for (int e = 0; e < g_prof_ev_n; e++) {
        int slot = g_prof_slot_log[e];
        if (slot < 0 || slot >= g_prof_n) continue;
        int st = g_prof_pair_start[e];
        if (st < 0 || st >= e) continue;
        float ms = 0.0f;
        if (gpuEventElapsedTime(&ms, g_prof_ev[st], g_prof_ev[e]) != gpuSuccess) continue;
        g_prof[slot].total_ms += (double)ms;
        g_prof[slot].calls++;
    }
    for (int e = 0; e < g_prof_ev_n; e++) { g_prof_slot_log[e] = -1; g_prof_pair_start[e] = -1; }
    g_prof_ev_n = 0;
    g_prof_open_n = 0;
}

extern "C" void picolm_gpu_prof_begin(int device) {
    /* NOTE the API is begin(device)/end(slot): the slot is only known at END.
     * Rather than guess, this pushes a placeholder and trusts the caller to
     * pair. Callers that return early between begin and end leak an entry,
     * so the report SKIPS any pair whose start is not strictly after the
     * previous end (i.e. out-of-order / leaked) instead of mis-attributing. */
    if (!prof_enabled()) return;
    if (g_prof_ev_n + 1 >= PROF_MAX_EVENTS) prof_flush_ring();
    if (g_prof_ev_n + 1 >= PROF_MAX_EVENTS) return;
    if (g_prof_open_n >= PROF_MAX_DEPTH) return;
    /* MUST record on ctx->stream, not the null stream: every kernel in this
     * backend is enqueued on ctx->stream, and an event on the null stream is
     * not ordered against them -- it would time nothing. */
    gpu_device_ctx_t *c = find_ctx(device);
    gpuStream_t st = (c && c->stream) ? c->stream : (gpuStream_t)0;
    g_prof_stream = st;
    gpuEventRecord(g_prof_ev[g_prof_ev_n], st);
    g_prof_depth = g_prof_ev_n;
    g_prof_open[g_prof_open_n++] = g_prof_ev_n;   /* index of this START */
    g_prof_ev_n++;
}

extern "C" void picolm_gpu_prof_end(int slot) {
    if (slot < 0 || !prof_enabled()) return;
    if (g_prof_ev_n >= PROF_MAX_EVENTS) return;
    int start = (g_prof_open_n > 0) ? g_prof_open[--g_prof_open_n] : -1;
    gpu_event_rec_stop();
    /* Store {start_index, end_index, slot} so the report pairs the right two. */
    g_prof_slot_log[g_prof_ev_n] = slot;
    g_prof_pair_start[g_prof_ev_n] = start;
    g_prof_ev_n++;
    (void)g_prof_depth;
}

/* Called once at exit, after the final sync, to resolve the deltas. */
extern "C" void picolm_gpu_prof_report(void) {
    if (!g_prof_on || g_prof_n == 0) return;
    gpuDeviceSynchronize();
    prof_flush_ring();
    double grand = 0.0;
    for (int k = 0; k < g_prof_n; k++) grand += g_prof[k].total_ms;
    fprintf(stderr, "\n[GPROF] per-kernel-class totals (%.1f ms measured)\n", grand);
    for (int k = 0; k < g_prof_n; k++) {
        double pct = grand > 0 ? 100.0 * g_prof[k].total_ms / grand : 0.0;
        fprintf(stderr, "[GPROF] %-24s %9.1f ms  %5.1f%%  n=%-7ld %.4f ms/call\n",
                g_prof[k].name, g_prof[k].total_ms, pct, g_prof[k].calls,
                g_prof[k].calls ? g_prof[k].total_ms / g_prof[k].calls : 0.0);
    }
}


/* ================================================================
 * GLM-5 KDA tail (prefill): conv1d + SiLU, L2 norm, gated delta-rule
 * recurrence, RMSNorm and sigmoid gate, for n_tokens in order.
 * ================================================================
 *
 * GPU port of glm5_kda_head_task() (model_glm5.c), the sequential part of a
 * KDA layer's prefill after its batched projections. One block per head, one
 * thread per state row / channel (KD = 128); the head's KD x KD state
 * (column-major, Sh[col*KD + row], as on the CPU) lives in shared memory for
 * the whole chunk. Inputs and outputs are host arrays: the state is read and
 * written back so the CPU decode continues from it. Not bit-identical to the
 * CPU (summation order, FMA); within ~1e-5, make test-glm5-kda-gpu. */
#define G5K_KD 128

__device__ static inline float g5k_block_sum(float v, float *red) {
    for (int o = 16; o > 0; o >>= 1) v += gpuShflXorSync(0xFFFFFFFFu, v, o);
    const int lane = gpuThreadIdx_x & 31, w = gpuThreadIdx_x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < G5K_KD / 32; i++) t += red[i];
    return t;
}

__global__ void __launch_bounds__(G5K_KD)
picolm_glm5_kda_tail_kernel(const float *q, const float *k, const float *v,
                            const float *z, const float *raw, const float *beta_raw,
                            const float *wq, const float *wk, const float *wv,
                            const float *dt_bias, const float *a, const float *norm_w,
                            float *conv_state, float *state, float *out,
                            int N, int H, int CK, float lower_bound, float eps) {
    extern __shared__ float g5k_sh[];
    float *S = g5k_sh;                       /* [KD*KD] */
    float *kd = S + G5K_KD * G5K_KD, *qd = kd + G5K_KD, *kk = qd + G5K_KD, *dd = kk + G5K_KD;
    float *red = dd + G5K_KD;                /* [KD/32] */
    const int h = gpuBlockIdx_x, i = gpuThreadIdx_x, KD = G5K_KD, DI = H * KD, CD = 3 * DI;
    const int ch = h * KD + i;
    float *Sh = state + (size_t)h * KD * KD;
    for (int c = 0; c < KD; c++) S[c * KD + i] = Sh[c * KD + i];
    float cs[3][3];                          /* conv state, CK <= 4 */
    for (int p = 0; p < 3; p++)
        for (int t = 0; t < CK - 1; t++) cs[p][t] = conv_state[(size_t)t * CD + p * DI + ch];
    const float scale = 1.0f / sqrtf((float)KD), a_h = a[h], dtb = dt_bias[ch], nw = norm_w[i];
    const float *w3[3] = { wq, wk, wv };
    for (int t = 0; t < N; t++) {
        const size_t o = (size_t)t * DI + ch;
        const float beta = 1.0f / (1.0f + expf(-beta_raw[(size_t)t * H + h]));
        const float g = lower_bound * (1.0f / (1.0f + expf(a_h * (raw[o] + dtb))));
        float xv[3] = { q[o], k[o], v[o] }, co[3];
        for (int p = 0; p < 3; p++) {
            float acc = 0.0f;
            for (int tt = 0; tt < CK - 1; tt++) acc += cs[p][tt] * w3[p][tt + ch * CK];
            acc += xv[p] * w3[p][(CK - 1) + ch * CK];
            co[p] = acc / (1.0f + expf(-acc));
            for (int tt = 0; tt < CK - 2; tt++) cs[p][tt] = cs[p][tt + 1];
            cs[p][CK - 2] = xv[p];
        }
        const float sq = g5k_block_sum(co[0] * co[0], red);
        const float sk = g5k_block_sum(co[1] * co[1], red);
        const float qi = co[0] * (1.0f / fmaxf(sqrtf(sq), 1e-5f));
        const float ki = co[1] * (1.0f / fmaxf(sqrtf(sk), 1e-5f));
        const float vi = co[2];
        const float attn = g5k_block_sum(ki * qi, red) * scale;
        const float dec = expf(fminf(g, 50.0f));
        kd[i] = ki * dec; qd[i] = qi * dec; kk[i] = ki; dd[i] = dec;
        __syncthreads();
        float vp = 0.0f, ov = 0.0f;
        for (int c = 0; c < KD; c++) {
            const float st = S[c * KD + i];
            vp += st * kd[c];
            ov += st * qd[c];
        }
        const float vn = beta * (vi - vp);
        ov = ov * scale + vn * attn;
        for (int c = 0; c < KD; c++) {
            const float st = dd[c] * S[c * KD + i] + vn * kk[c];
            S[c * KD + i] = fminf(fmaxf(st, -1e6f), 1e6f);
        }
        const float ss = g5k_block_sum(ov * ov, red);
        const float inv = 1.0f / sqrtf(ss / KD + eps);
        out[o] = ov * inv * nw * (1.0f / (1.0f + expf(-z[o])));
        __syncthreads();   /* kd/qd/kk/dd are rewritten next token */
    }
    for (int c = 0; c < KD; c++) Sh[c * KD + i] = S[c * KD + i];
    for (int p = 0; p < 3; p++)
        for (int t = 0; t < CK - 1; t++) conv_state[(size_t)t * CD + p * DI + ch] = cs[p][t];
}

/* Host wrapper: all pointers are host memory. Returns 0 (caller keeps the CPU
 * path) if the shape is unsupported or a CUDA call fails. */
extern "C" int
picolm_gpu_glm5_kda_tail(int device, int N, int H, int KD, int CK,
                         const float *q, const float *k, const float *v,
                         const float *z, const float *raw, const float *beta_raw,
                         const float *wq, const float *wk, const float *wv,
                         const float *dt_bias, const float *a, const float *norm_w,
                         float lower_bound, float eps,
                         float *conv_state, float *state, float *out) {
    if (KD != G5K_KD || CK < 2 || CK > 4 || N < 1 || H < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    const size_t DI = (size_t)H * KD, act = (size_t)N * DI, cw = DI * CK;
    const size_t cst = (size_t)(CK - 1) * 3 * DI, st = (size_t)H * KD * KD;
    const size_t total = 6 * act + (size_t)N * H + 3 * cw + DI + H + KD + cst + st;
    static float *buf[PICOLM_GPU_MAX_DEVICES]; static size_t cap[PICOLM_GPU_MAX_DEVICES];
    const int di = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
    if (cap[di] < total) {
        if (buf[di]) gpuFree(buf[di]);
        buf[di] = NULL; cap[di] = 0;
        if (!gpu_ok(gpuMalloc((void **)&buf[di], total * sizeof(float)), "glm5 kda tail alloc")) return 0;
        cap[di] = total;
    }
    float *p = buf[di];
    float *dq = p; p += act; float *dk = p; p += act; float *dv = p; p += act;
    float *dz = p; p += act; float *draw = p; p += act; float *dout = p; p += act;
    float *dbeta = p; p += (size_t)N * H;
    float *dwq = p; p += cw; float *dwk = p; p += cw; float *dwv = p; p += cw;
    float *ddt = p; p += DI; float *da = p; p += H; float *dnw = p; p += KD;
    float *dcs = p; p += cst; float *dst = p;
    const struct { float *d; const float *h; size_t n; } up[] = {
        { dq, q, act }, { dk, k, act }, { dv, v, act }, { dz, z, act }, { draw, raw, act },
        { dbeta, beta_raw, (size_t)N * H }, { dwq, wq, cw }, { dwk, wk, cw }, { dwv, wv, cw },
        { ddt, dt_bias, DI }, { da, a, (size_t)H }, { dnw, norm_w, (size_t)KD },
        { dcs, conv_state, cst }, { dst, state, st } };
    for (size_t j = 0; j < sizeof up / sizeof up[0]; j++)
        if (!gpu_ok(gpuMemcpyAsync(up[j].d, up[j].h, up[j].n * sizeof(float), gpuMemcpyHostToDevice, ctx->stream), "glm5 kda tail H2D")) return 0;
    const size_t smem = ((size_t)KD * KD + 4 * KD + KD / 32) * sizeof(float);
    static int attr_set[PICOLM_GPU_MAX_DEVICES];
    if (!attr_set[di]) {
        if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_kda_tail_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "glm5 kda tail smem")) return 0;
        attr_set[di] = 1;
    }
    picolm_glm5_kda_tail_kernel<<<H, KD, smem, ctx->stream>>>(dq, dk, dv, dz, draw, dbeta, dwq, dwk, dwv,
        ddt, da, dnw, dcs, dst, dout, N, H, CK, lower_bound, eps);
    if (!gpu_ok(gpuGetLastError(), "glm5 kda tail kernel")) return 0;
    if (!gpu_ok(gpuMemcpyAsync(out, dout, act * sizeof(float), gpuMemcpyDeviceToHost, ctx->stream), "glm5 kda tail D2H") ||
        !gpu_ok(gpuMemcpyAsync(state, dst, st * sizeof(float), gpuMemcpyDeviceToHost, ctx->stream), "glm5 kda tail D2H") ||
        !gpu_ok(gpuMemcpyAsync(conv_state, dcs, cst * sizeof(float), gpuMemcpyDeviceToHost, ctx->stream), "glm5 kda tail D2H"))
        return 0;
    return gpu_ok(gpuStreamSynchronize(ctx->stream), "glm5 kda tail sync");
}

/* ================================================================
 * GLM-5 decode on the GPU (model_glm5.c: glm5_gpu_forward)
 * ================================================================
 *
 * One token per call, everything on device pointers and on the device stream,
 * no sync: the hyper-connection pre/post mixes, the MLA attention over an F16
 * latent cache, and the small element-wise steps between the GEMVs. The
 * projections are picolm_gpu_matmul_dev() and the KDA step is the prefill tail
 * kernel above with N = 1. Same formulas and order as the CPU decode in
 * model_glm5.c; not bit-identical (summation order, FMA, expf). */

/* flat = x / sqrt(mean(x^2) + eps) over n values, sum of squares in double
 * like glm5_hc_pre(). One block. */
__global__ void __launch_bounds__(1024)
picolm_glm5_hc_norm_kernel(const float *x, float *flat, int n, float eps) {
    x += (size_t)gpuBlockIdx_x * n; flat += (size_t)gpuBlockIdx_x * n;   /* token */
    __shared__ double red[1024];
    __shared__ float inv_s;
    double ss = 0.0;
    for (int i = gpuThreadIdx_x; i < n; i += gpuBlockDim_x) ss += (double)x[i] * (double)x[i];
    red[gpuThreadIdx_x] = ss;
    __syncthreads();
    for (int o = gpuBlockDim_x / 2; o > 0; o >>= 1) {
        if ((int)gpuThreadIdx_x < o) red[gpuThreadIdx_x] += red[gpuThreadIdx_x + o];
        __syncthreads();
    }
    if (gpuThreadIdx_x == 0) inv_s = 1.0f / (float)sqrt(red[0] / (double)n + (double)eps);
    __syncthreads();
    const float inv = inv_s;
    for (int i = gpuThreadIdx_x; i < n; i += gpuBlockDim_x) flat[i] = x[i] * inv;
}

/* From the M = (2+S)*S mixes: pre (sigmoid + eps), post (2*sigmoid) and the
 * Sinkhorn-normalised S x S comb, then cur[d] = sum_s pre[s] * x[s][d].
 * Every block derives pre itself (S sigmoids); warp 0 of block 0 also runs
 * the Sinkhorn with one lane per comb element (row and column sums by
 * shuffles, in the CPU's order) and stores pc = [post S][comb S*S]. */
#define G5_HC_SMAX 5    /* S*S <= 32 lanes */
__global__ void picolm_glm5_hc_mix_kernel(const float *mm, const float *base, const float *scale,
                                          const float *x, float *cur, float *pc,
                                          int S, int D, float hc_eps, int iters) {
    __shared__ float pre[G5_HC_SMAX];
    const int tid = gpuThreadIdx_x, tok = gpuBlockIdx_y;
    mm += (size_t)tok * (2 + S) * S; x += (size_t)tok * S * D; cur += (size_t)tok * D;
    pc += (size_t)tok * (S + S * S);
    if (tid < S) {
        const float v = mm[tid] * scale[0] + base[tid];
        pre[tid] = 1.0f / (1.0f + expf(-v)) + hc_eps;
    }
    if (gpuBlockIdx_x == 0 && tid < 32) {
        const int i = tid, r = i / S, cc = i % S, on = i < S * S;
        const unsigned F = 0xFFFFFFFFu;
        if (i < S) {
            const float v = mm[S + i] * scale[1] + base[S + i];
            pc[i] = 2.0f * (1.0f / (1.0f + expf(-v)));
        }
        float v = on ? mm[2 * S + i] * scale[2] + base[2 * S + i] : 0.0f;
        float mx = gpuShflSync(F, v, r * S);
        for (int c = 1; c < S; c++) { const float o = gpuShflSync(F, v, r * S + c); if (o > mx) mx = o; }
        const float e = expf(v - mx);
        float sum = 0.f;
        for (int c = 0; c < S; c++) sum += gpuShflSync(F, e, r * S + c);
        v = e / sum + hc_eps;
        sum = hc_eps;
        for (int q = 0; q < S; q++) sum += gpuShflSync(F, v, q * S + cc);
        v /= sum;
        for (int it = 1; it < iters; it++) {
            sum = hc_eps;
            for (int c = 0; c < S; c++) sum += gpuShflSync(F, v, r * S + c);
            v /= sum;
            sum = hc_eps;
            for (int q = 0; q < S; q++) sum += gpuShflSync(F, v, q * S + cc);
            v /= sum;
        }
        if (on) pc[S + i] = v;
    }
    __syncthreads();
    const int d = gpuBlockIdx_x * gpuBlockDim_x + tid;
    if (d >= D) return;
    float acc = 0.f;
    for (int s = 0; s < S; s++) acc += pre[s] * x[(size_t)s * D + d];
    cur[d] = acc;
}

/* out[i][d] = a[d]*post[i] + sum_j comb[j*S+i] * res[j][d]  (glm5_hc_post) */
__global__ void picolm_glm5_hc_post_kernel(const float *a, const float *res, const float *pc,
                                           float *out, int S, int D) {
    const int tok = gpuBlockIdx_y;
    const int idx = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (idx >= S * D) return;
    a += (size_t)tok * D; res += (size_t)tok * S * D; pc += (size_t)tok * (S + S * S);
    out += (size_t)tok * S * D;
    const int i = idx / D, d = idx - i * D;
    float o = a[d] * pc[i];
    for (int j = 0; j < S; j++) o += pc[S + j * S + i] * res[(size_t)j * D + d];
    out[idx] = o;
}

/* Final norm input: mean over the S streams, then RMSNorm with w. One block. */
__global__ void __launch_bounds__(1024)
picolm_glm5_head_in_kernel(const float *x, const float *w, float *cur, int S, int D, float eps) {
    __shared__ float red[32];
    __shared__ float inv_s;
    const float inv_S = 1.0f / (float)S;
    float ss = 0.f;
    for (int d = gpuThreadIdx_x; d < D; d += gpuBlockDim_x) {
        float acc = 0.f;
        for (int i = 0; i < S; i++) acc += x[(size_t)i * D + d];
        acc *= inv_S;
        cur[d] = acc;
        ss += acc * acc;
    }
    for (int o = 16; o > 0; o >>= 1) ss += gpuShflXorSync(0xFFFFFFFFu, ss, o);
    if ((gpuThreadIdx_x & 31) == 0) red[gpuThreadIdx_x >> 5] = ss;
    __syncthreads();
    if (gpuThreadIdx_x == 0) {
        float t = 0.f;
        for (int k = 0; k < (int)(gpuBlockDim_x >> 5); k++) t += red[k];
        inv_s = 1.0f / sqrtf(t / D + eps);
    }
    __syncthreads();
    for (int d = gpuThreadIdx_x; d < D; d += gpuBlockDim_x) cur[d] *= inv_s * w[d];
}

__global__ void picolm_glm5_add_kernel(float *y, const float *x, int n) {
    const int i = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) y[i] += x[i];
}

/* This token's latent into the F16 cache row. */
__global__ void picolm_glm5_kv16_kernel(const float *kv, uint16_t *row, int n) {
    const int i = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) row[i] = gpu_fp32_to_fp16(kv[i]);
}

/* MLA decode attention, K == V == the F16 latent (KVL = 512), NH query heads.
 *
 * Split over the cache: block (c, g) takes keys [c*T, c*T+T) for heads
 * [g*8, g*8+8), one warp per key for the scores (lane holds 16 of the 512
 * latent values, kept for all 8 dots), one warp per head for the chunk's
 * softmax, and one thread per two latent columns for the weighted sum.
 * Partials (m, l, o) go to part; picolm_glm5_mla_merge_kernel combines them
 * per head. */
#define G5A_KVL 512
#define G5A_HG  8
#define G5A_TMAX 512
__global__ void __launch_bounds__(256)
picolm_glm5_mla_part_kernel(const uint16_t *cache, const float *qabs, float *part,
                            int NH, int n_kv, int T, float scale) {
    extern __shared__ float g5a_sh[];
    float *q = g5a_sh;                       /* [HG][KVL] */
    float *sc = q + G5A_HG * G5A_KVL;        /* [HG][T]   */
    float *ml = sc + G5A_HG * T;             /* [HG][2]   */
    const int c = gpuBlockIdx_x, g = gpuBlockIdx_y, tid = gpuThreadIdx_x;
    const int lane = tid & 31, w = tid >> 5;
    const int t0 = c * T, nt = min(T, n_kv - t0);
    for (int i = tid; i < G5A_HG * G5A_KVL; i += 256)
        q[i] = qabs[(size_t)g * G5A_HG * G5A_KVL + i];
    __syncthreads();
    for (int t = w; t < nt; t += 8) {
        const uint16_t *row = cache + (size_t)(t0 + t) * G5A_KVL;
        float kv[16];
        const uint4 a = *(const uint4 *)(row + lane * 8);
        const uint4 b = *(const uint4 *)(row + 256 + lane * 8);
        const uint16_t *ah = (const uint16_t *)&a, *bh = (const uint16_t *)&b;
        for (int j = 0; j < 8; j++) { kv[j] = gpu_fp16_to_fp32(ah[j]); kv[8 + j] = gpu_fp16_to_fp32(bh[j]); }
        for (int h = 0; h < G5A_HG; h++) {
            const float *qh = q + h * G5A_KVL;
            float s = 0.f;
            for (int j = 0; j < 8; j++) s += kv[j] * qh[lane * 8 + j];
            for (int j = 0; j < 8; j++) s += kv[8 + j] * qh[256 + lane * 8 + j];
            for (int o = 16; o > 0; o >>= 1) s += gpuShflXorSync(0xFFFFFFFFu, s, o);
            if (lane == 0) sc[h * T + t] = s * scale;
        }
    }
    __syncthreads();
    {   /* warp w: softmax numerators of head w over this chunk */
        float *sh = sc + w * T;
        float mx = -INFINITY;
        for (int t = lane; t < nt; t += 32) mx = fmaxf(mx, sh[t]);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, gpuShflXorSync(0xFFFFFFFFu, mx, o));
        float l = 0.f;
        for (int t = lane; t < nt; t += 32) { const float p = expf(sh[t] - mx); sh[t] = p; l += p; }
        for (int o = 16; o > 0; o >>= 1) l += gpuShflXorSync(0xFFFFFFFFu, l, o);
        if (lane == 0) { ml[w * 2] = mx; ml[w * 2 + 1] = l; }
    }
    __syncthreads();
    float acc[G5A_HG][2];
    for (int h = 0; h < G5A_HG; h++) acc[h][0] = acc[h][1] = 0.f;
    for (int t = 0; t < nt; t++) {
        const uint16_t *row = cache + (size_t)(t0 + t) * G5A_KVL;
        const float x0 = gpu_fp16_to_fp32(row[tid]), x1 = gpu_fp16_to_fp32(row[256 + tid]);
        for (int h = 0; h < G5A_HG; h++) {
            const float p = sc[h * T + t];
            acc[h][0] += p * x0;
            acc[h][1] += p * x1;
        }
    }
    const size_t stride = 2 + G5A_KVL;     /* per (chunk, head): m, l, o[KVL] */
    for (int h = 0; h < G5A_HG; h++) {
        float *pp = part + ((size_t)c * NH + g * G5A_HG + h) * stride;
        if (tid == 0) { pp[0] = ml[h * 2]; pp[1] = ml[h * 2 + 1]; }
        pp[2 + tid] = acc[h][0];
        pp[2 + 256 + tid] = acc[h][1];
    }
}

__global__ void __launch_bounds__(256)
picolm_glm5_mla_merge_kernel(const float *part, float *ctx, int NH, int nc) {
    const int h = gpuBlockIdx_x, tid = gpuThreadIdx_x;
    const size_t stride = 2 + G5A_KVL;
    float M = -INFINITY;
    for (int c = 0; c < nc; c++) M = fmaxf(M, part[((size_t)c * NH + h) * stride]);
    float L = 0.f, o0 = 0.f, o1 = 0.f;
    for (int c = 0; c < nc; c++) {
        const float *pp = part + ((size_t)c * NH + h) * stride;
        const float f = expf(pp[0] - M);
        L += pp[1] * f;
        o0 += pp[2 + tid] * f;
        o1 += pp[2 + 256 + tid] * f;
    }
    const float inv = 1.0f / L;
    ctx[(size_t)h * G5A_KVL + tid] = o0 * inv;
    ctx[(size_t)h * G5A_KVL + 256 + tid] = o1 * inv;
}

extern "C" int
picolm_gpu_glm5_hc_pre(int device, int S, int D, int N, const float *x, float *flat, float *mm,
                       picolm_gpu_tensor_t *fn, const float *base, const float *scale,
                       float eps, float hc_eps, int iters, float *cur, float *pc) {
    if (S < 1 || S > G5_HC_SMAX || N < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_hc_norm_kernel<<<N, 1024, 0, ctx->stream>>>(x, flat, S * D, eps);
    if (!gpu_ok(gpuGetLastError(), "glm5 hc norm")) return 0;
    if (!picolm_gpu_matmul_dev(fn, mm, flat, N, device, 0, 0)) return 0;
    picolm_glm5_hc_mix_kernel<<<dim3((D + 255) / 256, N), 256, 0, ctx->stream>>>(mm, base, scale, x, cur, pc,
                                                                                S, D, hc_eps, iters);
    return gpu_ok(gpuGetLastError(), "glm5 hc mix");
}

extern "C" int
picolm_gpu_glm5_hc_post(int device, int S, int D, int N, const float *a, const float *res,
                        const float *pc, float *out) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_hc_post_kernel<<<dim3((S * D + 255) / 256, N), 256, 0, ctx->stream>>>(a, res, pc, out, S, D);
    return gpu_ok(gpuGetLastError(), "glm5 hc post");
}

extern "C" int
picolm_gpu_glm5_head_in(int device, int S, int D, const float *x, const float *w, float eps, float *cur) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_head_in_kernel<<<1, 1024, 0, ctx->stream>>>(x, w, cur, S, D, eps);
    return gpu_ok(gpuGetLastError(), "glm5 head in");
}

extern "C" int picolm_gpu_glm5_add(int device, float *y, const float *x, int n) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_add_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(y, x, n);
    return gpu_ok(gpuGetLastError(), "glm5 add");
}

__global__ void picolm_glm5_swiglu_kernel(float *g, const float *u, size_t n, float clamp_v);
extern "C" int picolm_gpu_glm5_swiglu(int device, float *g, const float *u, int n, float clamp_v) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_swiglu_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(g, u, (size_t)n, clamp_v);
    return gpu_ok(gpuGetLastError(), "glm5 swiglu");
}

extern "C" int picolm_gpu_glm5_kv16(int device, const float *kv, uint16_t *row, int n) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_glm5_kv16_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(kv, row, n);
    return gpu_ok(gpuGetLastError(), "glm5 kv16");
}

/* MLA prefill attention on tensor cores (WMMA m16n16k16, f16 in, f32 acc).
 *
 * Block (t, g): 32 query rows = heads [g*32, g*32+32) of prefill token t, all
 * with the causal bound L = start + t + 1 over the F16 latent cache (K == V,
 * KVL = 512). Two passes over 32-key tiles: the first finds each row's max
 * and sum, the second accumulates softmax(S) . V with the O tile held in
 * fragments (no per-row rescale, which WMMA accumulators cannot do). The
 * extra QK pass costs 1.5x the flops of one-pass flash attention.
 * qabs and ctx are [n_tokens][NH][KVL] F32 on the device. */
#if !defined(__HIP__)
#include <mma.h>
#endif
#define G5P_BM 32
#define G5P_BN 32
#define G5P_LD (G5A_KVL + 8)       /* half stride of the Q and K tiles */
#define G5P_LS (G5P_BN + 4)        /* float stride of the score tiles */
#define G5P_LP (G5P_BN + 8)        /* half stride of the P tile */
#define G5P_SMEM ((size_t)2 * G5P_BM * G5P_LD * 2 + (size_t)2 * G5P_BM * G5P_LS * 4 \
                  + (size_t)G5P_BM * G5P_LP * 2 + (size_t)2 * G5P_BM * 4)

__global__ void __launch_bounds__(256)
picolm_glm5_mla_prefill_kernel(const uint16_t *cache, const float *qabs, float *ctx,
                               float *part_ml, int NH, int start, float scale, int kchunk) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIP__)
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char g5p_sh[];
    __half *Qs = (__half *)g5p_sh;                         /* [BM][LD] */
    __half *Ks = Qs + G5P_BM * G5P_LD;                     /* [BN][LD] */
    float *Sp = (float *)(Ks + G5P_BN * G5P_LD);           /* [2][BM][LS] split-K halves */
    __half *Ps = (__half *)(Sp + 2 * G5P_BM * G5P_LS);     /* [BM][LP] */
    float *rm = (float *)(Ps + G5P_BM * G5P_LP), *rl = rm + G5P_BM;
    /* part_ml == NULL: prefill, block x = token, all keys of its causal range,
     * normalised ctx. Else decode split over keys: block x = key chunk of
     * kchunk keys of the single query token (pos = start), unnormalised
     * partial o to ctx[x][NH][KVL] and (max, sum) to part_ml[x][NH][2]. */
    const int g = gpuBlockIdx_y, tid = gpuThreadIdx_x;
    const int w = tid >> 5, lane = tid & 31, KVL = G5A_KVL;
    const int split = part_ml != NULL;
    const int t = split ? 0 : gpuBlockIdx_x;
    const int kb = split ? gpuBlockIdx_x * kchunk : 0;
    const int L = split ? min(start + 1, kb + kchunk) : start + t + 1;
    const int ntile = (L - kb + G5P_BN - 1) / G5P_BN;
    const size_t row0 = (size_t)t * NH + (size_t)g * G5P_BM;
    const size_t orow0 = split ? (size_t)gpuBlockIdx_x * NH + (size_t)g * G5P_BM : row0;
    for (int i = tid; i < G5P_BM * KVL; i += 256) {
        const int r = i / KVL, d = i - r * KVL;
        Qs[r * G5P_LD + d] = __float2half(qabs[(row0 + r) * KVL + d]);
    }
    if (tid < G5P_BM) { rm[tid] = -INFINITY; rl[tid] = 0.f; }
    const int fr = (w & 3) >> 1, fc = w & 1, kh = w >> 2;  /* QK: S fragment, k half */
    const int rb = w >> 2, cb0 = (w & 3) * 8;              /* PV: row block, 8 col blocks */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[8];
    for (int j = 0; j < 8; j++) wmma::fill_fragment(o[j], 0.0f);
    for (int pass = 0; pass < 2; pass++) {
        for (int kt = 0; kt < ntile; kt++) {
            const int k0 = kb + kt * G5P_BN;
            __syncthreads();   /* previous tile's K/P reads are done */
            for (int i = tid; i < G5P_BN * (KVL / 8); i += 256) {
                const int r = i / (KVL / 8), c8 = (i - r * (KVL / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (k0 + r < L) v = *(const uint4 *)(cache + (size_t)(k0 + r) * KVL + c8);
                *(uint4 *)(Ks + r * G5P_LD + c8) = v;
            }
            __syncthreads();
            {   /* S = Q K^T, warp (fr, fc) over half of the 512 dims */
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> sa;
                wmma::fill_fragment(sa, 0.0f);
                for (int kk = kh * 16; kk < kh * 16 + 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(a, Qs + fr * 16 * G5P_LD + kk * 16, G5P_LD);
                    wmma::load_matrix_sync(b, Ks + fc * 16 * G5P_LD + kk * 16, G5P_LD);
                    wmma::mma_sync(sa, a, b, sa);
                }
                wmma::store_matrix_sync(Sp + kh * G5P_BM * G5P_LS + fr * 16 * G5P_LS + fc * 16,
                                        sa, G5P_LS, wmma::mem_row_major);
            }
            __syncthreads();
            for (int r = w * 4; r < w * 4 + 4; r++) {   /* warp: 4 rows, lane: key */
                const int key = k0 + lane;
                const float sv = key < L ? (Sp[r * G5P_LS + lane] + Sp[G5P_BM * G5P_LS + r * G5P_LS + lane]) * scale
                                         : -INFINITY;
                if (pass == 0) {
                    float mx = sv;
                    for (int of = 16; of > 0; of >>= 1) mx = fmaxf(mx, gpuShflXorSync(0xFFFFFFFFu, mx, of));
                    const float m0 = rm[r], l0 = rl[r], mn = fmaxf(m0, mx);
                    float e = sv == -INFINITY ? 0.f : expf(sv - mn);
                    for (int of = 16; of > 0; of >>= 1) e += gpuShflXorSync(0xFFFFFFFFu, e, of);
                    __syncwarp();   /* every lane has read rm/rl before lane 0 updates them */
                    if (lane == 0) {
                        rl[r] = (m0 == -INFINITY ? 0.f : l0 * expf(m0 - mn)) + e;
                        rm[r] = mn;
                    }
                } else {
                    float p = sv == -INFINITY ? 0.f : expf(sv - rm[r]);
                    if (!split) p /= rl[r];
                    Ps[r * G5P_LP + lane] = __float2half(p);
                }
            }
            if (pass == 0) continue;
            __syncthreads();
            for (int ks = 0; ks < G5P_BN / 16; ks++) {   /* O += P V */
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::load_matrix_sync(a, Ps + rb * 16 * G5P_LP + ks * 16, G5P_LP);
                for (int j = 0; j < 8; j++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::load_matrix_sync(b, Ks + ks * 16 * G5P_LD + (cb0 + j) * 16, G5P_LD);
                    wmma::mma_sync(o[j], a, b, o[j]);
                }
            }
        }
    }
    for (int j = 0; j < 8; j++)
        wmma::store_matrix_sync(ctx + (orow0 + rb * 16) * KVL + (cb0 + j) * 16, o[j], KVL, wmma::mem_row_major);
    if (split && tid < G5P_BM) {
        part_ml[(orow0 + tid) * 2] = rm[tid];
        part_ml[(orow0 + tid) * 2 + 1] = rl[tid];
    }
#else
    (void)cache; (void)qabs; (void)ctx; (void)part_ml; (void)NH; (void)start; (void)scale; (void)kchunk;
#endif
}

/* Merge of the split decode partials above: per head, rescale each chunk's
 * (max, sum, o) to the global max and normalise. */
__global__ void __launch_bounds__(256)
picolm_glm5_mla_merge2_kernel(const float *ml, const float *po, float *ctx, int NH, int nc) {
    const int h = gpuBlockIdx_x, tid = gpuThreadIdx_x;
    float M = -INFINITY;
    for (int c = 0; c < nc; c++) M = fmaxf(M, ml[((size_t)c * NH + h) * 2]);
    float Lsum = 0.f, o0 = 0.f, o1 = 0.f;
    for (int c = 0; c < nc; c++) {
        const size_t r = (size_t)c * NH + h;
        const float f = expf(ml[r * 2] - M);
        Lsum += ml[r * 2 + 1] * f;
        o0 += po[r * G5A_KVL + tid] * f;
        o1 += po[r * G5A_KVL + 256 + tid] * f;
    }
    const float inv = 1.0f / Lsum;
    ctx[(size_t)h * G5A_KVL + tid] = o0 * inv;
    ctx[(size_t)h * G5A_KVL + 256 + tid] = o1 * inv;
}

/* ctx[NH][KVL] = softmax(scale * cache . qabs[h]) . cache over n_kv rows.
 * part is scratch from picolm_gpu_glm5_mla_part_size(). 0 if the shape is
 * unsupported (KVL != 512 or NH not a multiple of 8). */
static int g5a_chunk(int n_kv, int *nc) {
    int T = 64;
    if ((n_kv + T - 1) / T > 256) T = ((n_kv + 255) / 256 + 31) & ~31;
    if (T > G5A_TMAX) T = G5A_TMAX;
    *nc = (n_kv + T - 1) / T;
    return T;
}

extern "C" size_t picolm_gpu_glm5_mla_part_size(int NH, int n_kv) {
    int nc; g5a_chunk(n_kv, &nc);
    const int nc2 = (n_kv + 255) / 256;     /* the tensor-core split, 256 keys a chunk */
    return (size_t)(nc > nc2 ? nc : nc2) * NH * (2 + G5A_KVL) * sizeof(float);
}

extern "C" int
picolm_gpu_glm5_mla_attn(int device, int NH, int KVL, int n_kv, float scale,
                         const uint16_t *cache, const float *qabs, float *part, float *ctx_out) {
    if (KVL != G5A_KVL || NH % G5A_HG || n_kv < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
#if !defined(__HIP__)
    /* Long contexts: the tensor-core kernel split over 256-key chunks reads the
     * cache once per 32 heads instead of once per 8, at f16 q precision
     * (~3e-4 instead of ~1e-6). Partials fit in the scratch: fewer chunks. */
    static int wmma_min = -1;
    if (wmma_min < 0) { const char *e = getenv("PICOLM_GLM5_MLA_WMMA_MIN"); wmma_min = e ? atoi(e) : 4096; }
    if (wmma_min > 0 && n_kv >= wmma_min && ctx->compute_major >= 7 && NH % G5P_BM == 0) {
        const int kc = 256, nc2 = (n_kv + kc - 1) / kc;
        static int attr2[PICOLM_GPU_MAX_DEVICES];
        const int d2 = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
        if (!attr2[d2]) {
            if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_mla_prefill_kernel,
                        gpuFuncAttributeMaxDynamicSharedMemorySize, (int)G5P_SMEM), "glm5 mla split smem")) return 0;
            attr2[d2] = 1;
        }
        float *po = part, *pml = part + (size_t)nc2 * NH * G5A_KVL;
        dim3 grid2((unsigned)nc2, (unsigned)(NH / G5P_BM));
        picolm_glm5_mla_prefill_kernel<<<grid2, 256, G5P_SMEM, ctx->stream>>>(cache, qabs, po, pml, NH,
                                                                             n_kv - 1, scale, kc);
        if (!gpu_ok(gpuGetLastError(), "glm5 mla split")) return 0;
        picolm_glm5_mla_merge2_kernel<<<NH, 256, 0, ctx->stream>>>(pml, po, ctx_out, NH, nc2);
        return gpu_ok(gpuGetLastError(), "glm5 mla merge2");
    }
#endif
    int nc; const int T = g5a_chunk(n_kv, &nc);
    const size_t smem = ((size_t)G5A_HG * G5A_KVL + (size_t)G5A_HG * T + 2 * G5A_HG) * sizeof(float);
    static int attr_set[PICOLM_GPU_MAX_DEVICES];
    const int di = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
    if (!attr_set[di]) {
        const size_t mx = ((size_t)G5A_HG * G5A_KVL + (size_t)G5A_HG * G5A_TMAX + 2 * G5A_HG) * sizeof(float);
        if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_mla_part_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize, (int)mx), "glm5 mla smem")) return 0;
        attr_set[di] = 1;
    }
    dim3 grid((unsigned)nc, (unsigned)(NH / G5A_HG));
    picolm_glm5_mla_part_kernel<<<grid, 256, smem, ctx->stream>>>(cache, qabs, part, NH, n_kv, T, scale);
    if (!gpu_ok(gpuGetLastError(), "glm5 mla part")) return 0;
    picolm_glm5_mla_merge_kernel<<<NH, 256, 0, ctx->stream>>>(part, ctx_out, NH, nc);
    return gpu_ok(gpuGetLastError(), "glm5 mla merge");
}

/* Device-pointer form for the device-resident prefill; no sync. */
extern "C" int
picolm_gpu_glm5_mla_prefill_dev(int device, int NH, int KVL, int n_tokens, int start, float scale,
                                const uint16_t *cache, const float *qabs, float *ctx_out) {
#if defined(__HIP__)
    (void)device; (void)NH; (void)KVL; (void)n_tokens; (void)start; (void)scale;
    (void)cache; (void)qabs; (void)ctx_out;
    return 0;
#else
    if (KVL != G5A_KVL || NH % G5P_BM || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx) || ctx->compute_major < 7) return 0;
    static int attr_set[PICOLM_GPU_MAX_DEVICES];
    const int di = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
    if (!attr_set[di]) {
        if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_mla_prefill_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize, (int)G5P_SMEM), "glm5 mla prefill smem")) return 0;
        attr_set[di] = 1;
    }
    dim3 grid((unsigned)n_tokens, (unsigned)(NH / G5P_BM));
    picolm_glm5_mla_prefill_kernel<<<grid, 256, G5P_SMEM, ctx->stream>>>(cache, qabs, ctx_out, NULL, NH, start, scale, 0);
    return gpu_ok(gpuGetLastError(), "glm5 mla prefill dev");
#endif
}

/* ctx[t][h] = softmax over keys [0, start + t] of the F16 latent cache, for
 * the n_tokens prefill rows of qabs; host arrays in and out, cache on the
 * device. 0 if unsupported (HIP, pre-Volta, KVL != 512, NH % 32). */
extern "C" int
picolm_gpu_glm5_mla_prefill(int device, int NH, int KVL, int n_tokens, int start, float scale,
                            const uint16_t *cache, const float *qabs, float *ctx_out) {
#if defined(__HIP__)
    (void)device; (void)NH; (void)KVL; (void)n_tokens; (void)start; (void)scale;
    (void)cache; (void)qabs; (void)ctx_out;
    return 0;
#else
    if (KVL != G5A_KVL || NH % G5P_BM || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx) || ctx->compute_major < 7) return 0;
    const size_t n = (size_t)n_tokens * NH * KVL;
    static float *buf[PICOLM_GPU_MAX_DEVICES]; static size_t cap[PICOLM_GPU_MAX_DEVICES];
    static int attr_set[PICOLM_GPU_MAX_DEVICES];
    const int di = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
    if (cap[di] < 2 * n) {
        if (buf[di]) gpuFree(buf[di]);
        buf[di] = NULL; cap[di] = 0;
        if (!gpu_ok(gpuMalloc((void **)&buf[di], 2 * n * sizeof(float)), "glm5 mla prefill alloc")) return 0;
        cap[di] = 2 * n;
    }
    if (!attr_set[di]) {
        if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_mla_prefill_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize, (int)G5P_SMEM), "glm5 mla prefill smem")) return 0;
        attr_set[di] = 1;
    }
    float *dq = buf[di], *dc = dq + n;
    if (!gpu_ok(gpuMemcpyAsync(dq, qabs, n * sizeof(float), gpuMemcpyHostToDevice, ctx->stream), "glm5 mla prefill H2D"))
        return 0;
    dim3 grid((unsigned)n_tokens, (unsigned)(NH / G5P_BM));
    picolm_glm5_mla_prefill_kernel<<<grid, 256, G5P_SMEM, ctx->stream>>>(cache, dq, dc, NULL, NH, start, scale, 0);
    if (!gpu_ok(gpuGetLastError(), "glm5 mla prefill kernel")) return 0;
    if (!gpu_ok(gpuMemcpyAsync(ctx_out, dc, n * sizeof(float), gpuMemcpyDeviceToHost, ctx->stream), "glm5 mla prefill D2H"))
        return 0;
    return gpu_ok(gpuStreamSynchronize(ctx->stream), "glm5 mla prefill sync");
#endif
}

/* N tokens of a KDA layer on device pointers (the prefill tail kernel);
 * conv_state and state stay on the device. */
extern "C" int
picolm_gpu_glm5_kda_step(int device, int N, int H, int KD, int CK,
                         const float *q, const float *k, const float *v,
                         const float *z, const float *raw, const float *beta_raw,
                         const float *wq, const float *wk, const float *wv,
                         const float *dt_bias, const float *a, const float *norm_w,
                         float lower_bound, float eps, float *conv_state, float *state, float *out) {
    if (KD != G5K_KD || CK < 2 || CK > 4 || H < 1 || N < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    const size_t smem = ((size_t)KD * KD + 4 * KD + KD / 32) * sizeof(float);
    static int attr_set[PICOLM_GPU_MAX_DEVICES];
    const int di = device < PICOLM_GPU_MAX_DEVICES ? device : 0;
    if (!attr_set[di]) {
        if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_glm5_kda_tail_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "glm5 kda step smem")) return 0;
        attr_set[di] = 1;
    }
    picolm_glm5_kda_tail_kernel<<<H, KD, smem, ctx->stream>>>(q, k, v, z, raw, beta_raw, wq, wk, wv,
        dt_bias, a, norm_w, conv_state, state, out, N, H, CK, lower_bound, eps);
    return gpu_ok(gpuGetLastError(), "glm5 kda step");
}

/* Device-to-host copy ordered on the device stream, then wait for it. */
extern "C" int picolm_gpu_glm5_d2h(int device, void *dst, const void *src, size_t bytes) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!gpu_ok(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyDeviceToHost, ctx->stream), "glm5 D2H")) return 0;
    return gpu_ok(gpuStreamSynchronize(ctx->stream), "glm5 D2H sync");
}
