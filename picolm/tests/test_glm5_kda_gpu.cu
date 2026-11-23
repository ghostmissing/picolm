/* test_glm5_kda_gpu: picolm_gpu_glm5_kda_tail (the GLM-5 KDA prefill tail on
 * the GPU) against a C copy of glm5_kda_head_task()'s arithmetic, over N
 * tokens with carried-in conv and recurrent state. Checks outputs, the
 * updated state and the conv state (relative to their max magnitude).
 * Also the decode MLA attention (picolm_gpu_glm5_mla_attn) against a plain
 * softmax over the F16 latent cache, across the chunk-size steps.
 *
 *   make test-glm5-kda-gpu && ./test_glm5_kda_gpu */
#include "backend_gpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The GPU host objects reference these for weight upload paths this test
 * never takes; stubs keep the link to the CUDA objects alone. */
extern "C" {
#define STUB_DQ(n) void n(const void *src, float *dst, int k) { (void)src; (void)dst; (void)k; abort(); }
STUB_DQ(dequantize_row_q2_K) STUB_DQ(dequantize_row_q3_K) STUB_DQ(dequantize_row_q4_K)
STUB_DQ(dequantize_row_q5_K) STUB_DQ(dequantize_row_q6_K) STUB_DQ(dequantize_row_iq4_nl)
void quantize_row_q8_0(const float *x, void *dst, int n) { (void)x; (void)dst; (void)n; abort(); }
}

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

/* Reference: one token, all heads (glm5_kda_head_task per head). */
static void ref_token(int H, int KD, int CK, const float *q, const float *k, const float *v,
                      const float *z, const float *raw, const float *beta_raw,
                      const float *wq, const float *wk, const float *wv,
                      const float *dt_bias, const float *a, const float *norm_w,
                      float lb, float eps, float *cs, float *S, float *out) {
    const int DI = H * KD, CD = 3 * DI;
    float *co = (float *)malloc(sizeof(float) * CD);
    for (int h = 0; h < H; h++) {
        const float beta = 1.0f / (1.0f + expf(-beta_raw[h]));
        float g[256], dec[256], vp[256], ov[256];
        for (int c = 0; c < KD; c++) g[c] = lb * (1.0f / (1.0f + expf(a[h] * (raw[h*KD+c] + dt_bias[h*KD+c]))));
        for (int p = 0; p < 3; p++) {
            const float *src = p == 0 ? q : p == 1 ? k : v, *w = p == 0 ? wq : p == 1 ? wk : wv;
            for (int i = 0; i < KD; i++) {
                const int off = h*KD + i, ch = p*DI + off;
                float xs[4];
                for (int t = 0; t < CK - 1; t++) xs[t] = cs[t*CD + ch];
                xs[CK-1] = src[off];
                float acc = 0.f;
                for (int t = 0; t < CK; t++) acc += xs[t] * w[t + off*CK];
                co[ch] = acc / (1.0f + expf(-acc));
                for (int t = 0; t < CK - 2; t++) cs[t*CD + ch] = cs[(t+1)*CD + ch];
                cs[(CK-2)*CD + ch] = src[off];
            }
        }
        float *qh = co + h*KD, *kh = co + DI + h*KD, *vh = co + 2*DI + h*KD;
        float sq = 0, sk = 0;
        for (int c = 0; c < KD; c++) { sq += qh[c]*qh[c]; sk += kh[c]*kh[c]; }
        const float iq = 1.0f / fmaxf(sqrtf(sq), 1e-5f), ik = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int c = 0; c < KD; c++) { qh[c] *= iq; kh[c] *= ik; }
        const float scale = 1.0f / sqrtf((float)KD);
        float attn = 0; for (int c = 0; c < KD; c++) attn += kh[c]*qh[c]; attn *= scale;
        for (int c = 0; c < KD; c++) dec[c] = expf(fminf(g[c], 50.0f));
        float *Sh = S + (size_t)h*KD*KD;
        for (int r = 0; r < KD; r++) { vp[r] = 0; ov[r] = 0; }
        for (int c = 0; c < KD; c++) for (int r = 0; r < KD; r++) {
            vp[r] += Sh[c*KD+r] * (kh[c]*dec[c]); ov[r] += Sh[c*KD+r] * (qh[c]*dec[c]); }
        float vn[256];
        for (int r = 0; r < KD; r++) { vn[r] = beta * (vh[r] - vp[r]); ov[r] = ov[r]*scale + vn[r]*attn; }
        for (int c = 0; c < KD; c++) for (int r = 0; r < KD; r++) {
            float st = dec[c]*Sh[c*KD+r] + vn[r]*kh[c]; Sh[c*KD+r] = fminf(fmaxf(st, -1e6f), 1e6f); }
        float ss = 0; for (int r = 0; r < KD; r++) ss += ov[r]*ov[r];
        const float inv = 1.0f / sqrtf(ss / KD + eps);
        for (int r = 0; r < KD; r++) out[h*KD+r] = ov[r]*inv*norm_w[r] * (1.0f / (1.0f + expf(-z[h*KD+r])));
    }
    free(co);
}

static double maxrel(const float *a, const float *b, size_t n) {
    double mx = 0, mg = 0;
    for (size_t i = 0; i < n; i++) { mg = fmax(mg, fabs(b[i])); }
    for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs(a[i] - b[i]));
    return mg > 0 ? mx / mg : mx;
}

static int run(int N, int H, int CK) {
    const int KD = 128, DI = H * KD;
    const size_t act = (size_t)N * DI, cst = (size_t)(CK - 1) * 3 * DI, st = (size_t)H * KD * KD;
    float *q = (float *)malloc(act*4), *k = (float *)malloc(act*4), *v = (float *)malloc(act*4);
    float *z = (float *)malloc(act*4), *raw = (float *)malloc(act*4), *br = (float *)malloc((size_t)N*H*4);
    float *wq = (float *)malloc((size_t)DI*CK*4), *wk = (float *)malloc((size_t)DI*CK*4), *wv = (float *)malloc((size_t)DI*CK*4);
    float *dtb = (float *)malloc((size_t)DI*4), *a = (float *)malloc((size_t)H*4), *nw = (float *)malloc((size_t)KD*4);
    float *cs0 = (float *)malloc(cst*4), *S0 = (float *)malloc(st*4);
    for (size_t i = 0; i < act; i++) { q[i] = frand(); k[i] = frand(); v[i] = frand(); z[i] = frand(); raw[i] = frand(); }
    for (int i = 0; i < N*H; i++) br[i] = frand();
    for (int i = 0; i < DI*CK; i++) { wq[i] = frand()*0.5f; wk[i] = frand()*0.5f; wv[i] = frand()*0.5f; }
    for (int i = 0; i < DI; i++) dtb[i] = frand();
    for (int i = 0; i < H; i++) a[i] = frand();
    for (int i = 0; i < KD; i++) nw[i] = 1.0f + 0.1f*frand();
    for (size_t i = 0; i < cst; i++) cs0[i] = frand();
    for (size_t i = 0; i < st; i++) S0[i] = 0.1f*frand();
    float *csr = (float *)malloc(cst*4), *Sr = (float *)malloc(st*4), *outr = (float *)malloc(act*4);
    float *csg = (float *)malloc(cst*4), *Sg = (float *)malloc(st*4), *outg = (float *)malloc(act*4);
    memcpy(csr, cs0, cst*4); memcpy(Sr, S0, st*4); memcpy(csg, cs0, cst*4); memcpy(Sg, S0, st*4);
    for (int t = 0; t < N; t++)
        ref_token(H, KD, CK, q + (size_t)t*DI, k + (size_t)t*DI, v + (size_t)t*DI, z + (size_t)t*DI,
                  raw + (size_t)t*DI, br + (size_t)t*H, wq, wk, wv, dtb, a, nw, -5.0f, 1e-6f, csr, Sr, outr + (size_t)t*DI);
    int ok = picolm_gpu_glm5_kda_tail(0, N, H, KD, CK, q, k, v, z, raw, br, wq, wk, wv, dtb, a, nw,
                                      -5.0f, 1e-6f, csg, Sg, outg);
    double eo = maxrel(outg, outr, act), es = maxrel(Sg, Sr, st), ec = maxrel(csg, csr, cst);
    int fail = !ok || !(eo < 1e-4) || !(es < 1e-4) || !(ec < 1e-6);
    printf("N=%-4d H=%-3d CK=%d  out %.1e  state %.1e  conv %.1e  %s\n", N, H, CK, eo, es, ec, !ok ? "GPU declined" : fail ? "FAIL" : "ok");
    free(q); free(k); free(v); free(z); free(raw); free(br); free(wq); free(wk); free(wv); free(dtb); free(a); free(nw);
    free(cs0); free(S0); free(csr); free(Sr); free(outr); free(csg); free(Sg); free(outg);
    return fail;
}

/* ctx[h] = softmax(scale * cache . q[h]) . cache, cache rows F16 (K == V). */
static int run_mla(int NH, int n_kv) {
    const int KVL = 512;
    const float scale = 1.0f / sqrtf(256.0f);
    uint16_t *c16 = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n_kv * KVL);
    float *cf = (float *)malloc(sizeof(float) * (size_t)n_kv * KVL);
    float *q = (float *)malloc(sizeof(float) * NH * KVL), *ref = (float *)calloc((size_t)NH * KVL, sizeof(float));
    float *got = (float *)malloc(sizeof(float) * NH * KVL), *sc = (float *)malloc(sizeof(float) * n_kv);
    for (size_t i = 0; i < (size_t)n_kv * KVL; i++) {
        __half hv = __float2half(frand());
        c16[i] = __half_as_ushort(hv); cf[i] = __half2float(hv);
    }
    for (int i = 0; i < NH * KVL; i++) q[i] = frand() * 2.0f;
    for (int h = 0; h < NH; h++) {
        float mx = -1e30f; double l = 0;
        for (int t = 0; t < n_kv; t++) {
            double d = 0;
            for (int j = 0; j < KVL; j++) d += (double)cf[(size_t)t * KVL + j] * q[h * KVL + j];
            sc[t] = (float)d * scale; if (sc[t] > mx) mx = sc[t];
        }
        for (int t = 0; t < n_kv; t++) { sc[t] = expf(sc[t] - mx); l += sc[t]; }
        for (int t = 0; t < n_kv; t++)
            for (int j = 0; j < KVL; j++) ref[h * KVL + j] += sc[t] / (float)l * cf[(size_t)t * KVL + j];
    }
    const size_t pb = picolm_gpu_glm5_mla_part_size(NH, n_kv);
    uint16_t *dc = (uint16_t *)picolm_gpu_alloc_device(sizeof(uint16_t) * (size_t)n_kv * KVL, 0);
    float *dq = (float *)picolm_gpu_alloc_device(sizeof(float) * NH * KVL, 0);
    float *dctx = (float *)picolm_gpu_alloc_device(sizeof(float) * NH * KVL, 0);
    float *dp = (float *)picolm_gpu_alloc_device(pb, 0);
    int ok = dc && dq && dctx && dp &&
             picolm_gpu_memcpy(dc, c16, sizeof(uint16_t) * (size_t)n_kv * KVL, 1, 0) &&
             picolm_gpu_memcpy(dq, q, sizeof(float) * NH * KVL, 1, 0) &&
             picolm_gpu_glm5_mla_attn(0, NH, KVL, n_kv, scale, dc, dq, dp, dctx) &&
             picolm_gpu_glm5_d2h(0, got, dctx, sizeof(float) * NH * KVL);
    double md = 0, mr = 0;
    for (int i = 0; ok && i < NH * KVL; i++) {
        double d = fabs((double)got[i] - ref[i]);
        if (d > md) md = d;
        if (fabs(ref[i]) > mr) mr = fabs(ref[i]);
    }
    const double rel = mr > 0 ? md / mr : md;
    ok = ok && rel < 1e-4;
    printf("mla NH=%-3d n_kv=%-6d ctx %.1e  %s\n", NH, n_kv, rel, ok ? "ok" : "FAIL");
    cudaFree(dc); cudaFree(dq); cudaFree(dctx); cudaFree(dp);
    free(c16); free(cf); free(q); free(ref); free(got); free(sc);
    return !ok;
}

int main(void) {
    int dev0 = 0;
    if (!picolm_gpu_init(&dev0, 1)) { printf("no GPU: skipped\n"); return 0; }
    int f = 0;
    f |= run(1, 1, 4);
    f |= run(37, 4, 4);
    f |= run(512, 32, 4);
    f |= run(64, 8, 3);
    f |= run_mla(8, 1);
    f |= run_mla(16, 63);
    f |= run_mla(64, 700);
    f |= run_mla(32, 17000);     /* > 256 chunks of 64: larger chunks */
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
