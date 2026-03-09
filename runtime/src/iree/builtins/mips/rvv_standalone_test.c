// Standalone QEMU test for rvv_matmul_core.
//
// Build for QEMU (no libc — RV-only syscall wrappers):
//   clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
//         -O2 -static -nostdlib -ffreestanding \
//         rvv_standalone_test.c -o rvv_test
//   qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 ./rvv_test
//
// Build for host validation (x86, scalar fallback, with libc):
//   clang rvv_standalone_test.c -O2 -o rvv_test_host && ./rvv_test_host

#include <stddef.h>
#include <stdint.h>

// ── I/O and exit ──────────────────────────────────────────────────────────────
// RISC-V target: raw ecall (no libc dependency for -nostdlib build).
// Host (x86) target: libc stdio — so the --host build also works.

#ifdef __riscv

static long _rv_syscall1(long nr, long a0) {
  register long _a7 __asm__("a7") = nr;
  register long _a0 __asm__("a0") = a0;
  __asm__ volatile("ecall" : "+r"(_a0) : "r"(_a7) : "memory");
  return _a0;
}

static long _rv_syscall3(long nr, long a0, long a1, long a2) {
  register long _a7 __asm__("a7") = nr;
  register long _a0 __asm__("a0") = a0;
  register long _a1 __asm__("a1") = a1;
  register long _a2 __asm__("a2") = a2;
  __asm__ volatile("ecall" : "+r"(_a0) : "r"(_a7), "r"(_a1), "r"(_a2)
                   : "memory");
  return _a0;
}

static void sys_write(const char *buf, size_t len) {
  _rv_syscall3(64 /*SYS_write*/, 1 /*stdout*/, (long)buf, (long)len);
}

__attribute__((noreturn)) static void sys_exit(int code) {
  _rv_syscall1(94 /*SYS_exit_group*/, (long)code);
  __builtin_unreachable();
}

void _start(void);  // forward-declare; entry point at bottom

#else // host x86 build

#include <stdio.h>
#include <stdlib.h>

static void sys_write(const char *buf, size_t len) {
  fwrite(buf, 1, len, stdout);
}

__attribute__((noreturn)) static void sys_exit(int code) {
  exit(code);
}

#endif // __riscv

// ── Minimal print helpers (no sprintf / printf dependency) ────────────────────

static void print(const char *s) {
  size_t n = 0;
  while (s[n]) ++n;
  sys_write(s, n);
}

static void print_float(float v) {
  if (v < 0.0f) { print("-"); v = -v; }
  int whole = (int)v;
  int frac  = (int)((v - (float)whole) * 10000.0f + 0.5f);
  char buf[20];
  int i = 19;
  buf[i--] = '\0';
  for (int j = 0; j < 4; ++j) { buf[i--] = (char)('0' + frac % 10); frac /= 10; }
  buf[i--] = '.';
  if (whole == 0) { buf[i--] = '0'; }
  else { while (whole > 0) { buf[i--] = (char)('0' + whole % 10); whole /= 10; } }
  print(&buf[i + 1]);
}

// ── Inline RVV matmul core (no IREE headers required) ─────────────────────────
// Guards on __riscv_vector (set by clang when -march=rv64gcv is active) rather
// than __riscv so the RVV path only compiles when vector intrinsics are present.

#ifdef __riscv_vector
#include <riscv_vector.h>

static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  A += A_off; B += B_off; C += C_off;
  const int b_unit = (B_s1 == 1);
  const int c_unit = (C_s1 == 1);
  for (int64_t m = 0; m < M; ++m) {
    int64_t n = 0;
    while (n < N) {
      size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n));
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      for (int64_t k = 0; k < K; ++k) {
        float a_val = A[m * A_s0 + k * A_s1];
        vfloat32m4_t b_vec = b_unit
            ? __riscv_vle32_v_f32m4(&B[k * B_s0 + n], vl)
            : __riscv_vlse32_v_f32m4(&B[k * B_s0 + n * B_s1],
                                      B_s1 * (int64_t)sizeof(float), vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a_val, b_vec, vl);
      }
      if (c_unit)
        __riscv_vse32_v_f32m4(&C[m * C_s0 + n], acc, vl);
      else
        __riscv_vsse32_v_f32m4(&C[m * C_s0 + n * C_s1],
                                C_s1 * (int64_t)sizeof(float), acc, vl);
      n += (int64_t)vl;
    }
  }
}

#else // scalar fallback

static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  A += A_off; B += B_off; C += C_off;
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += A[m * A_s0 + k * A_s1] * B[k * B_s0 + n * B_s1];
      C[m * C_s0 + n * C_s1] = acc;
    }
}

#endif // __riscv_vector

// ── Tests ─────────────────────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

static float _fabsf(float x) { return x < 0.0f ? -x : x; }

static void check(const char *name, float got, float expected) {
  if (_fabsf(got - expected) < 1e-4f) {
    ++tests_passed;
  } else {
    ++tests_failed;
    print("  FAIL "); print(name);
    print(" got="); print_float(got);
    print(" expected="); print_float(expected); print("\n");
  }
}

// Test 1: A * I = A  (4×4, row-major)
static void test_identity(void) {
  print("[1] A * I = A  (4x4 row-major)\n");
  float A[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16 };
  float I[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
  float C[16] = {0};
  rvv_matmul_core(A, 0,4,1, I, 0,4,1, C, 0,4,1, 4,4,4);
  for (int i = 0; i < 16; ++i) check("A*I", C[i], A[i]);
}

// Test 2: 2x3 * 3x2 = 2x2
//   [[58,64],[139,154]]
static void test_2x3x2(void) {
  print("[2] 2x3 * 3x2 = 2x2\n");
  float A[6] = {1,2,3, 4,5,6};
  float B[6] = {7,8, 9,10, 11,12};
  float C[4] = {0};
  rvv_matmul_core(A, 0,3,1, B, 0,2,1, C, 0,2,1, 2,2,3);
  check("C[0,0]", C[0],  58.0f);
  check("C[0,1]", C[1],  64.0f);
  check("C[1,0]", C[2], 139.0f);
  check("C[1,1]", C[3], 154.0f);
}

// Test 3: column-major strides
static void test_col_major(void) {
  print("[3] col-major strides\n");
  // A[2×3] col-major: stored [1,4, 2,5, 3,6], s0=1, s1=2
  float A[6] = {1,4, 2,5, 3,6};
  // B[3×2] col-major: stored [7,9,11, 8,10,12], s0=1, s1=3
  float B[6] = {7,9,11, 8,10,12};
  float C[4] = {0};
  rvv_matmul_core(A, 0,1,2, B, 0,1,3, C, 0,1,2, 2,2,3);
  // C[m,n] = C[m + n*2]
  check("C[0,0]", C[0],  58.0f);
  check("C[1,0]", C[1], 139.0f);
  check("C[0,1]", C[2],  64.0f);
  check("C[1,1]", C[3], 154.0f);
}

// Test 4: non-zero base offset
static void test_offset(void) {
  print("[4] non-zero base offset\n");
  float A[8] = {99,99,99,99, 1,0, 0,1};
  float B[8] = {99,99,99,99, 3,0, 0,5};
  float C[8] = {0};
  rvv_matmul_core(A, 4,2,1, B, 4,2,1, C, 4,2,1, 2,2,2);
  check("C[0,0]", C[4], 3.0f);
  check("C[0,1]", C[5], 0.0f);
  check("C[1,0]", C[6], 0.0f);
  check("C[1,1]", C[7], 5.0f);
}

// ── Entry point ───────────────────────────────────────────────────────────────

// On RISC-V nostdlib builds the linker expects _start.
// On host builds we emit main() so it links normally.

#ifdef __riscv
void _start(void) {
#else
int main(void) {
#endif
  print("=== rvv_matmul standalone test");
#ifdef __riscv_vector
  print(" [RVV]\n");
#else
  print(" [scalar]\n");
#endif

  test_identity();
  test_2x3x2();
  test_col_major();
  test_offset();

  print("\n");
  print(tests_failed == 0 ? "PASSED" : "FAILED");
  print(" (");
  char b[4] = {'0' + (char)tests_passed, ' ', '\0', '\0'};
  b[1] = '\0'; print(b);
  print(" passed, ");
  b[0] = '0' + (char)tests_failed;
  print(b);
  print(" failed)\n");

#ifdef __riscv
  sys_exit(tests_failed == 0 ? 0 : 1);
#else
  sys_exit(tests_failed == 0 ? 0 : 1);
  return 0;
#endif
}
