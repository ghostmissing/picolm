/* test_kq_gemm_gpu: on Volta, picolm_kq_q8_K_matmul_hmma (Q4_K/Q5_K/Q6_K x
 * Q8_K) must be within 1e-6 (of the sum of |terms|) of a double-precision
 * reference of the same dequantised weights, and prints its speed.
 *
 *   make test-kq-gemm-gpu && ./test_kq_gemm_gpu */
#include "backend_gpu_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static uint32_t rng = 11;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float h2f(uint16_t h) { uint32_t u = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(((h >> 10) & 0x1F) - 15 + 127) << 23) | ((uint32_t)(h & 0x3FF) << 13); float f; memcpy(&f, &u, 4); return f; }
static void smk4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
/* reference dequant of one 256-block to doubles */
static void deq(int qt, const uint8_t *b, double *y) {
    if (qt == 14) {
        double d = h2f(b[208] | b[209] << 8); const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        for (int n = 0; n < 2; n++) { for (int l = 0; l < 32; l++) { int is = l / 16;
            y[l]      = d * sc[is + 0] * (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32); }
          y += 128; ql += 64; qh += 32; sc += 8; }
    } else {
        double d = h2f(b[0] | b[1] << 8), dm = h2f(b[2] | b[3] << 8); const uint8_t *qs = b + (qt == 13 ? 48 : 16), *qh = b + 16;
        for (int j = 0; j < 8; j++) { uint8_t sc, mn; smk4(j, b + 4, &sc, &mn);
            for (int l = 0; l < 32; l++) { int q = (j & 1) ? (qs[(j / 2) * 32 + l] >> 4) : (qs[(j / 2) * 32 + l] & 0xF);
                if (qt == 13) q |= ((qh[l] >> j) & 1) << 4;
                y[j * 32 + l] = d * sc * q - dm * mn; } }
    }
}
static int run(int qt, int S, int I, int O) {
    int bsz = qt == 14 ? 210 : qt == 13 ? 176 : 144, ns = I / 256, rb = ns * bsz;
    uint8_t *w = (uint8_t *)malloc((size_t)O * rb); for (size_t i = 0; i < (size_t)O * rb; i++) w[i] = rnd();
    for (size_t k = 0; k < (size_t)O * ns; k++) { uint8_t *b = w + k * bsz;
        if (qt == 14) { b[208] = rnd(); b[209] = 0x10 + rnd() % 0x10; } else { b[0] = rnd(); b[1] = 0x10 + rnd() % 0x10; b[2] = rnd(); b[3] = 0x10 + rnd() % 0x10; } }
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *xd = (float *)malloc((size_t)S * ns * 4); int16_t *bs = (int16_t *)malloc((size_t)S * ns * 32);
    for (size_t i = 0; i < (size_t)S * I; i++) xq[i] = rnd() % 255 - 127;
    for (size_t k = 0; k < (size_t)S * ns; k++) { xd[k] = (rnd() % 1000 + 1) / 1e5f; for (int j = 0; j < 16; j++) { int s = 0; for (int t = 0; t < 16; t++) s += xq[k * 256 + j * 16 + t]; bs[k * 16 + j] = s; } }
    void *dw; int8_t *dq; float *dd, *dy; int16_t *db;
    cudaMalloc(&dw, (size_t)O * rb); cudaMalloc(&dq, (size_t)S * I); cudaMalloc(&dd, (size_t)S * ns * 4); cudaMalloc(&db, (size_t)S * ns * 32); cudaMalloc(&dy, (size_t)S * O * 4);
    cudaMemcpy(dw, w, (size_t)O * rb, cudaMemcpyHostToDevice); cudaMemcpy(dq, xq, (size_t)S * I, cudaMemcpyHostToDevice); cudaMemcpy(dd, xd, (size_t)S * ns * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, bs, (size_t)S * ns * 32, cudaMemcpyHostToDevice);
    dim3 g((O + 63) / 64, (S + 127) / 128);
    picolm_kq_q8_K_matmul_hmma_launch(qt, g, 0, dy, dq, dd, db, dw, S, I, O, rb, O);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int r = 0; r < 3; r++) picolm_kq_q8_K_matmul_hmma_launch(qt, g, 0, dy, dq, dd, db, dw, S, I, O, rb, O);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 3;
    float *hy = (float *)malloc((size_t)S * O * 4); cudaMemcpy(hy, dy, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    double mx = 0, *wr = (double *)malloc(256 * 8); int n = 0;
    for (int o = 0; o < O; o += (O > 300 ? 97 : 1))
        for (int s = 0; s < S; s += (S > 64 ? 37 : 1)) {
            double ref = 0, mag = 0;
            for (int k = 0; k < ns; k++) { deq(qt, w + (size_t)o * rb + (size_t)k * bsz, wr);
                for (int i = 0; i < 256; i++) { double t = wr[i] * xq[(size_t)s * I + k * 256 + i] * xd[(size_t)s * ns + k]; ref += t; mag += fabs(t); } }
            double e = fabs(hy[(size_t)s * O + o] - ref) / mag; if (!(e <= mx)) mx = e; n++;
        }
    int fail = err != cudaSuccess || !(mx < 1e-6);
    printf("%s S=%-4d I=%-6d O=%-5d %7.2f ms %5.1f TOPS  max_err %.1e (%d) %s\n", qt == 14 ? "Q6_K" : qt == 13 ? "Q5_K" : "Q4_K", S, I, O, ms, 2.0 * S * I * O / ms / 1e9, mx, n,
           err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dq); cudaFree(dd); cudaFree(db); cudaFree(dy); free(w); free(xq); free(xd); free(bs); free(hy); free(wr);
    return fail;
}
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!(prop.major == 7 && prop.minor == 0)) { printf("not a Volta GPU: skipped\n"); return 0; }
    int f = 0;
    for (int qt = 12; qt <= 14; qt++) { f |= run(qt, 5, 256, 7); f |= run(qt, 37, 1024, 129); f |= run(qt, 1900, 5120, 4096); }
    printf(f ? "FAILED\n" : "all ok\n"); return f;
}
