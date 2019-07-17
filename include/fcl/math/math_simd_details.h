/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011-2014, Willow Garage, Inc.
 *  Copyright (c) 2014-2016, Open Source Robotics Foundation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Open Source Robotics Foundation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Jia Pan */


#ifndef FCL_MATH_SIMD_DETAILS_H
#define FCL_MATH_SIMD_DETAILS_H

// #include "fcl/common/types.h"

// NOTE: some compilers might not define __SSE4__ macro even if SSE4 is supported.
// Check with following command:
// $ cc -march=native -dM -E - < /dev/null | egrep "SSE|AVX"
#if not defined(__SSE4__)
#if defined(__SSE4_1__) or defined(__SSE4_2__)
#define __SSE4__
#endif
#endif

#include <xmmintrin.h>
#if defined (__SSE3__)
#include <pmmintrin.h>
#endif
#if defined (__SSE4__)
#include <smmintrin.h>
#endif
#if defined (__AVX__)
#include <immintrin.h>
#endif


namespace fcl
{

namespace details
{

#define vec_splat_ps(a, e) _mm_shuffle_ps((a), (a), _MM_SHUFFLE((e), (e), (e), (e)))
#define vec_splat_pd(a, e) _mm256_shuffle_pd((a), (a), _MM_SHUFFLE((e), (e), (e), (e)))


//==============================================================================
// Some of the following macros / matrix functions are adapted from MathGeoLib:
// https://github.com/juj/MathGeoLib

inline __m128 abs_ps(const __m128& x)
{
  static const __m128 sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
  return _mm_andnot_ps(sign_mask, x);
}
#if defined (__AVX__)
inline __m256d abs_pd(const __m256d& x)
{
  static const __m256d sign_mask = _mm256_set1_pd(-0); // -0 = 1 << 63
  return _mm256_andnot_pd(sign_mask, x);
}
#endif

#define shuffle1_ps(reg, shuffle) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128((reg)), (shuffle)))
// #define shuffle1_pd(reg, shuffle) _mm256_castsi256_pd(_mm256_shuffle_epi32(_mm256_castpd_si256((reg)), (shuffle)))

#define allzero_ps(x) _mm_testz_si128(_mm_castps_si128((x)), _mm_castps_si128((x)))
#define allzero_pd(x) _mm256_testz_si256(_mm256_castpd_si256((x)), _mm256_castpd_si256((x)))

// fused multiply–accumulate operation
#define fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define fmsub_ps(a, b, c) _mm_sub_ps(_mm_mul_ps((a), (b)), (c))

#if defined (__AVX__)
#define fmadd_pd(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
#define fmsub_pd(a, b, c) _mm256_sub_pd(_mm256_mul_pd((a), (b)), (c))
#endif


//==============================================================================
/// @brief
/// Compute the transpose of M, where M is a 3x4 matrix denoted by an array of 4 __m128's
/// and all of its last column is zero.
///
/// @param[in]  src
/// [ x0 y0 z0 0 ]
/// [ x1 y1 z1 0 ]
/// [ x2 y2 z2 0 ]
///
/// @param[out] dst
/// [ x0 x1 x2 0 ]
/// [ y0 y1 y2 0 ]
/// [ z0 z1 z2 0 ]
///
/// TODO: compare with _MM_TRANSPOSE4_PS
#if 1
static inline
void mat3x4_transpose(const __m128* src_r0, const __m128* src_r1, const __m128* src_r2, __m128* dst_r0, __m128* dst_r1, __m128* dst_r2)
{
  __m128 src3 = _mm_setzero_ps();
  __m128 tmp0 = _mm_unpacklo_ps(*src_r0, *src_r1);  // [ x0 x1 y0 y1 ]
  __m128 tmp2 = _mm_unpacklo_ps(*src_r2, src3);     // [ x2  0 y2  0 ]
  __m128 tmp1 = _mm_unpackhi_ps(*src_r0, *src_r1);  // [ z0 z1  0  0 ]
  __m128 tmp3 = _mm_unpackhi_ps(*src_r2, src3);     // [ z2  0  0  0 ]
  *dst_r0 = _mm_movelh_ps(tmp0, tmp2);              // [ x0 x1 x2  0 ]
  *dst_r1 = _mm_movehl_ps(tmp2, tmp0);              // [ y0 y1 y2  0 ]
  *dst_r2 = _mm_movelh_ps(tmp1, tmp3);              // [ z0 z1 z2  0 ]
}
#else
// Another implementation
// NOTE: the CPI of _mm_unpacklo_ps / _mm_unpackhi_ps and _mm_movelh_ps / _mm_movehl_ps is the same

static inline
void mat3x4_transpose(const __m128* src_r0, const __m128* src_r1, const __m128* src_r2, __m128* dst_r0, __m128* dst_r1, __m128* dst_r2)
{
  __m128 src3 = _mm_setzero_ps();
  __m128 tmp0 = _mm_unpacklo_ps(*src_r0, *src_r2);  // [ x0 x2 y0 y2 ]
  __m128 tmp1 = _mm_unpacklo_ps(*src_r1, src3);     // [ x1  0 y1  0 ]
  __m128 tmp2 = _mm_unpackhi_ps(*src_r0, *src_r2);  // [ z0 z2  0  0 ]
  __m128 tmp3 = _mm_unpackhi_ps(*src_r1, src3);     // [ z1  0  0  0 ]
  *dst_r0 = _mm_unpacklo_ps(tmp0, tmp1);            // [ x0 x1 x2  0 ]
  *dst_r1 = _mm_unpackhi_ps(tmp0, tmp1);            // [ y0 y1 y2  0 ]
  *dst_r2 = _mm_unpacklo_ps(tmp2, tmp3);            // [ z0 z1 z2  0 ]
}
#endif

static inline
void mat3x4_transpose(const __m128 src[3], __m128 dst[3])
{
  mat3x4_transpose(&src[0], &src[1], &src[2],
      &dst[0], &dst[1], &dst[2]);
}

#if defined (__AVX__)
static inline
void mat3x4_transpose(const __m256d* src_r0, const __m256d* src_r1, const __m256d* src_r2, __m256d* dst_r0, __m256d* dst_r1, __m256d* dst_r2)
{
  __m256d src3 = _mm256_setzero_pd();
  __m256d tmp0 = _mm256_unpacklo_pd(*src_r0, *src_r2);  // [ x0 x2 y0 y2 ]
  __m256d tmp1 = _mm256_unpacklo_pd(*src_r1, src3);     // [ x1  0 y1  0 ]
  __m256d tmp2 = _mm256_unpackhi_pd(*src_r0, *src_r2);  // [ z0 z2  0  0 ]
  __m256d tmp3 = _mm256_unpackhi_pd(*src_r1, src3);     // [ z1  0  0  0 ]
  *dst_r0 = _mm256_unpacklo_pd(tmp0, tmp1);             // [ x0 x1 x2  0 ]
  *dst_r1 = _mm256_unpackhi_pd(tmp0, tmp1);             // [ y0 y1 y2  0 ]
  *dst_r2 = _mm256_unpacklo_pd(tmp2, tmp3);             // [ z0 z1 z2  0 ]
}

static inline
void mat3x4_transpose(const __m256d src[3], __m256d dst[3])
{
  mat3x4_transpose(&src[0], &src[1], &src[2],
      &dst[0], &dst[1], &dst[2]);
}
#endif


//==============================================================================
/// @brief
/// Compute the product M*v,
/// where M is a 3x4 matrix denoted by an array of 4 __m128's,
/// and v is a 4x1 vector denoted by a __m128.
///
/// \param[in] matrix of 3x4
/// [ x0 x1 x2 0 ]
/// [ y0 y1 y2 0 ]
/// [ z0 z1 z2 0 ]
///
/// \param[in] vector of 4x1
/// [ v0 v1 v2 0 ]
///
/// \return a __m128 which represents the result 3x1 vector and a padding element which we don't care
/// [ x0v0+x1v1+x2v2 y0v0+y1v1+y2v2 z0v0+z1v1+z2v2 0 ]
#if defined (__SSE4__)
static inline
__m128 mat3x4_mul_vec4(const __m128* matrix_r0, const __m128* matrix_r1, const __m128* matrix_r2, const __m128* vector)
{
  // NOTE: Opinions vary on whether dpps is more efficient.
  // Further micro-benchmarking is needed.
  // Please refer to the following debates:
  // https://codereview.stackexchange.com/questions/101144/simd-matrix-multiplication
  // https://stackoverflow.com/questions/37879678/dot-product-performance-with-sse-instructions
  __m128 prod0 = _mm_dp_ps(*matrix_r0, *vector, 0xFF);  // [ x0v0+x1v1+x2v2 x0v0+x1v1+x2v2 x0v0+x1v1+x2v2 x0v0+x1v1+x2v2 ]
  __m128 prod1 = _mm_dp_ps(*matrix_r1, *vector, 0xFF);  // [ y0v0+y1v1+y2v2 y0v0+y1v1+y2v2 y0v0+y1v1+y2v2 y0v0+y1v1+y2v2 ]
  __m128 prod2 = _mm_dp_ps(*matrix_r2, *vector, 0xFF);  // [ z0v0+z1v1+z2v2 z0v0+z1v1+z2v2 z0v0+z1v1+z2v2 z0v0+z1v1+z2v2 ]
  __m128 prod3 = _mm_set1_ps(0.f);
  // shuffle lhs: [ x0v0+x1v1+x2v2 x0v0+x1v1+x2v2 y0v0+y1v1+y2v2 y0v0+y1v1+y2v2 ]
  // shuffle rhs: [ z0v0+z1v1+z2v2 z0v0+z1v1+z2v2 y0v0+y1v1+y2v2 y0v0+y1v1+y2v2 ]
  // shuffle result: [ x0v0+x1v1+x2v2 y0v0+y1v1+y2v2 z0v0+z1v1+z2v2 y0v0+y1v1+y2v2 ]
  return _mm_shuffle_ps(_mm_movelh_ps(prod0, prod1), _mm_movelh_ps(prod2, prod3), _MM_SHUFFLE(2, 0, 2, 0));
}
#else
#if 1
static inline
__m128 mat3x4_mul_vec4(const __m128* matrix_r0, const __m128* matrix_r1, const __m128* matrix_r2, const __m128* vector)
{
  __m128 x = _mm_mul_ps(*matrix_r0, *vector);     // [ x0v0 x1v1 x2v2 0 ]
  __m128 y = _mm_mul_ps(*matrix_r1, *vector);     // [ y0v0 y1v1 y2v2 0 ]
  __m128 t0 = _mm_unpacklo_ps(x, y);              // [ x0v0 y0v0 x1v1 y1v1 ]
  __m128 t1 = _mm_unpackhi_ps(x, y);              // [ x2v2 y2v2 0 0 ]
  t0 = _mm_add_ps(t0, t1);                        // [ x0v0+x2v2 y0v0+y2v2 x1v1 y1v1 ]
  __m128 z = _mm_mul_ps(*matrix_r2, *vector);     // [ z0v0 z1v1 z2v2 0 ]
  __m128 w = _mm_set1_ps(0.f);
  __m128 t2 = _mm_unpacklo_ps(z, w);              // [ z0v0 0 z1v1 0 ]
  __m128 t3 = _mm_unpackhi_ps(z, w);              // [ z2v2 0 0 0 ]
  t2 = _mm_add_ps(t2, t3);                        // [ z0v0+z2v2 0 z1v1 0 ]
  // [ x0v0+x2v2 y0v0+y2v2 z0v0+z2v2 0 ] + [ x1v1 y1y1 z1v1 0 ]
  return _mm_add_ps(_mm_movelh_ps(t0, t2), _mm_movehl_ps(t2, t0));
}
#else
// A more general version
static inline
__m128 mat3x4_mul_vec4(const __m128* matrix_r0, const __m128* matrix_r1, const __m128* matrix_r2, const __m128* vector)
{
  __m128 x = _mm_mul_ps(*matrix_r0, *vector);     // [ x0v0 x1v1 x2v2 0 ]
  __m128 y = _mm_mul_ps(*matrix_r1, *vector);     // [ y0v0 y1v1 y2v2 0 ]
  __m128 t0 = _mm_unpacklo_ps(x, y);              // [ x0v0 y0v0 x1v1 y1v1 ]
  __m128 t1 = _mm_unpackhi_ps(x, y);              // [ x2v2 y2v2 0 0 ]
  t0 = _mm_add_ps(t0, t1);                        // [ x0v0+x2v2 y0v0+y2v2 x1v1 y1v1 ]
  __m128 z = _mm_mul_ps(*matrix_r2, *vector);     // [ z0v0 z1v1 z2v2 0 ]
  __m128 w = _mm_mul_ps(_mm_setr_ps(0.f, 0.f, 0.f, 1.f), *vector);  // [ 0 0 0 v2 ]
  __m128 t2 = _mm_unpacklo_ps(z, w);              // [ z0v0 v0 z1v1 0 ]
  __m128 t3 = _mm_unpackhi_ps(z, w);              // [ z2v2 0 0 0 ]
  t2 = _mm_add_ps(t2, t3);                        // [ z0v0+z2v2 v0 z1v1 0 ]
  // [ x0v0+x2v2 y0v0+y2v2 z0v0+z2v2 v0 ] + [ x1v1 y1y1 z1v1 0 ]
  return _mm_add_ps(_mm_movelh_ps(t0, t2), _mm_movehl_ps(t2, t0));
}
#endif
#endif

static inline
__m128 mat3x4_mul_vec4(const __m128 matrix[3], const __m128 vector)
{
  return mat3x4_mul_vec4(&matrix[0], &matrix[1], &matrix[2], &vector);
}

#if defined (__AVX__)
static inline
__m256d mat3x4_mul_vec4(const __m256d* matrix_r0, const __m256d* matrix_r1, const __m256d* matrix_r2, const __m256d* vector)
{
  __m256d x = _mm256_mul_pd(*matrix_r0, *vector);     // [ x0v0 x1v1 x2v2 0 ]
  __m256d y = _mm256_mul_pd(*matrix_r1, *vector);     // [ y0v0 y1v1 y2v2 0 ]
  __m256d t0 = _mm256_unpacklo_pd(x, y);              // [ x0v0 y0v0 x1v1 y1v1 ]
  __m256d t1 = _mm256_unpackhi_pd(x, y);              // [ x2v2 y2v2 0 0 ]
  t0 = _mm256_add_pd(t0, t1);                         // [ x0v0+x2v2 y0v0+y2v2 x1v1 y1v1 ]
  __m256d z = _mm256_mul_pd(*matrix_r2, *vector);     // [ z0v0 z1v1 z2v2 0 ]
  __m256d w = _mm256_set1_pd(0);
  __m256d t2 = _mm256_unpacklo_pd(z, w);              // [ z0v0 0 z1v1 0 ]
  __m256d t3 = _mm256_unpackhi_pd(z, w);              // [ z2v2 0 0 0 ]
  t2 = _mm256_add_pd(t2, t3);                         // [ z0v0+z2v2 0 z1v1 0 ]
  // [ x0v0+x2v2 y0v0+y2v2 z1v1 0 ] + [ x1v1 y1y1 z1v1 0 ]
  return _mm256_add_pd(_mm256_blend_pd(t0, t2, 0b1100), _mm256_permute2f128_pd(t0, t2, 0x21));
}

static inline
__m256d mat3x4_mul_vec4(const __m256d matrix[3], const __m256d vector)
{
  return mat3x4_mul_vec4(&matrix[0], &matrix[1], &matrix[2], &vector);
}
#endif

//==============================================================================
/// @brief
/// Compute the product (M)^T*v,
/// where M is a 3x4 matrix denoted by an array of 4 __m128's,
/// and v is a 4x1 vector denoted by a __m128.
///
/// \param[in] matrix of 3x4
/// [ x0 x1 x2 0 ]
/// [ y0 y1 y2 0 ]
/// [ z0 z1 z2 0 ]
///
/// \param[in] vector of 4x1
/// [ v0 v1 v2 0 ]
///
/// \return a __m128 which represents the result 3x1 vector and a padding element which we don't care
/// [ x0v0+y0v1+z0v2 x1v0+y1v1+z1v2 x2v0+y2v1+z2v2 0 ]
static inline
__m128 transp_mat3x4_mul_vec4(const __m128* matrix_r0, const __m128* matrix_r1, const __m128* matrix_r2, const __m128* vector)
{
  __m128 r0 = _mm_mul_ps(vec_splat_ps(*vector, 0), *matrix_r0);
  __m128 r1 = _mm_mul_ps(vec_splat_ps(*vector, 1), *matrix_r1);
  __m128 r2 = _mm_mul_ps(vec_splat_ps(*vector, 2), *matrix_r2);
  return _mm_add_ps(_mm_add_ps(r0, r1), r2);
}

static inline
__m128 transp_mat3x4_mul_vec4(const __m128 matrix[3], const __m128 vector)
{
  return transp_mat3x4_mul_vec4(&matrix[0], &matrix[1], &matrix[2], &vector);
}

#if defined (__AVX__)
static inline
__m256d transp_mat3x4_mul_vec4(const __m256d* matrix_r0, const __m256d* matrix_r1, const __m256d* matrix_r2, const __m256d* vector)
{
  __m256d r0 = _mm256_mul_pd(vec_splat_pd(*vector, 0), *matrix_r0);
  __m256d r1 = _mm256_mul_pd(vec_splat_pd(*vector, 1), *matrix_r1);
  __m256d r2 = _mm256_mul_pd(vec_splat_pd(*vector, 2), *matrix_r2);
  return _mm256_add_pd(_mm256_add_pd(r0, r1), r2);
}

static inline
__m256d transp_mat3x4_mul_vec4(const __m256d matrix[3], const __m256d vector)
{
  return transp_mat3x4_mul_vec4(&matrix[0], &matrix[1], &matrix[2], &vector);
}
#endif


//==============================================================================
/// @brief
/// Compute the product M1*M2,
/// where M1 is a 3x4 matrix denoted by an array of 4 __m128's,
/// and M2 is a 3x4 matrix denoted by an array of 4 __m128's.
/// The last columns of both M1 and M2 are zero vectors.
///
/// \param[in] matrix M1 of 3x4 (we only use 3x3 submatrix)
/// [ a00 a01 a02 0 ]
/// [ a10 a11 a12 0 ]
/// [ a20 a21 a22 0 ]
///
/// \param[in] matrix M2 of 3x4 (we only use 3x3 submatrix)
/// [ b00 b01 b02 0 ]
/// [ b10 b11 b12 0 ]
/// [ b20 b21 b22 0 ]
///
/// \param[out] matrix out of 3x4 (we only care about the 3x3 submatrix)
/// [ a00b00+a01b10+a02b20 a00b01+a01b11+a02b21 a00b02+a01b12+a02b22 0 ]
/// [ a10b00+a11b10+a12b20 a10b01+a11b11+a12b21 a10b02+a11b12+a12b22 0 ]
/// [ a20b00+a21b10+a22b20 a20b01+a21b11+a22b21 a20b02+a21b12+a22b22 0 ]
static inline
void mat3x3_mul_mat3x3(__m128* out_r0, __m128* out_r1, __m128* out_r2,
    const __m128* m1_r0, const __m128* m1_r1, const __m128* m1_r2,
    const __m128* m2_r0, const __m128* m2_r1, const __m128* m2_r2)
{
  const __m128 m2_3 = _mm_setr_ps(0.f, 0.f, 0.f, 1.f);

  __m128 r0 = _mm_mul_ps(vec_splat_ps(*m1_r0, 0), *m2_r0);
  __m128 r1 = _mm_mul_ps(vec_splat_ps(*m1_r0, 1), *m2_r1);
  __m128 r2 = _mm_mul_ps(vec_splat_ps(*m1_r0, 2), *m2_r2);
  __m128 r3 = _mm_mul_ps(*m1_r0, m2_3);
  *out_r0 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));

  r0 = _mm_mul_ps(vec_splat_ps(*m1_r1, 0), *m2_r0);
  r1 = _mm_mul_ps(vec_splat_ps(*m1_r1, 1), *m2_r1);
  r2 = _mm_mul_ps(vec_splat_ps(*m1_r1, 2), *m2_r2);
  r3 = _mm_mul_ps(*m1_r1, m2_3);
  *out_r1 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));

  r0 = _mm_mul_ps(vec_splat_ps(*m1_r2, 0), *m2_r0);
  r1 = _mm_mul_ps(vec_splat_ps(*m1_r2, 1), *m2_r1);
  r2 = _mm_mul_ps(vec_splat_ps(*m1_r2, 2), *m2_r2);
  r3 = _mm_mul_ps(*m1_r2, m2_3);
  *out_r2 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
}

static inline
void mat3x3_mul_mat3x3(__m128 out[3], const __m128 m1[3], const __m128 m2[3])
{
  mat3x3_mul_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}

#if defined (__AVX__)
static inline
void mat3x3_mul_mat3x3(__m256d* out_r0, __m256d* out_r1, __m256d* out_r2,
    const __m256d* m1_r0, const __m256d* m1_r1, const __m256d* m1_r2,
    const __m256d* m2_r0, const __m256d* m2_r1, const __m256d* m2_r2)
{
  const __m256d m2_3 = _mm256_setr_pd(0, 0, 0, 1);

  __m256d r0 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 0), *m2_r0);
  __m256d r1 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 1), *m2_r1);
  __m256d r2 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 2), *m2_r2);
  __m256d r3 = _mm256_mul_pd(*m1_r0, m2_3);
  *out_r0 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));

  r0 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 0), *m2_r0);
  r1 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 1), *m2_r1);
  r2 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 2), *m2_r2);
  r3 = _mm256_mul_pd(*m1_r1, m2_3);
  *out_r1 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));

  r0 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 0), *m2_r0);
  r1 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 1), *m2_r1);
  r2 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 2), *m2_r2);
  r3 = _mm256_mul_pd(*m1_r2, m2_3);
  *out_r2 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));
}

static inline
void mat3x3_mul_mat3x3(__m256d out[3], const __m256d m1[3], const __m256d m2[3])
{
  mat3x3_mul_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}
#endif


//==============================================================================
/// @brief
/// Compute the product (M1)^T*M2,
/// where M1 is a 3x4 matrix denoted by an array of 4 __m128's,
/// and M2 is a 3x4 matrix denoted by an array of 4 __m128's.
/// The last columns of both M1 and M2 are zero vectors.
///
/// \param[in] matrix M1 of 3x4 (we only use 3x3 submatrix)
/// [ a00 a01 a02 0 ]
/// [ a10 a11 a12 0 ]
/// [ a20 a21 a22 0 ]
///
/// \param[in] matrix M2 of 3x4 (we only use 3x3 submatrix)
/// [ b00 b01 b02 0 ]
/// [ b10 b11 b12 0 ]
/// [ b20 b21 b22 0 ]
///
/// \param[out] matrix out of 3x4 (we only care about the 3x3 submatrix)
/// [ a00b00+a10b10+a20b20 a00b01+a10b11+a20b21 a00b02+a10b12+a20b22 0 ]
/// [ a01b00+a11b10+a21b20 a01b01+a11b11+a21b21 a01b02+a11b12+a21b22 0 ]
/// [ a02b00+a12b10+a22b20 a02b01+a12b11+a22b21 a02b02+a12b12+a22b22 0 ]
static inline
void transp_mat3x3_mul_mat3x3(__m128* out_r0, __m128* out_r1, __m128* out_r2,
    const __m128* m1_r0, const __m128* m1_r1, const __m128* m1_r2,
    const __m128* m2_r0, const __m128* m2_r1, const __m128* m2_r2)
{
  const __m128 padding = _mm_setzero_ps();

  __m128 r0 = _mm_mul_ps(vec_splat_ps(*m1_r0, 0), *m2_r0);
  __m128 r1 = _mm_mul_ps(vec_splat_ps(*m1_r1, 0), *m2_r1);
  __m128 r2 = _mm_mul_ps(vec_splat_ps(*m1_r2, 0), *m2_r2);
  *out_r0 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, padding));

  r0 = _mm_mul_ps(vec_splat_ps(*m1_r0, 1), *m2_r0);
  r1 = _mm_mul_ps(vec_splat_ps(*m1_r1, 1), *m2_r1);
  r2 = _mm_mul_ps(vec_splat_ps(*m1_r2, 1), *m2_r2);
  *out_r1 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, padding));

  r0 = _mm_mul_ps(vec_splat_ps(*m1_r0, 2), *m2_r0);
  r1 = _mm_mul_ps(vec_splat_ps(*m1_r1, 2), *m2_r1);
  r2 = _mm_mul_ps(vec_splat_ps(*m1_r2, 2), *m2_r2);
  *out_r2 = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, padding));
}

static inline
void transp_mat3x3_mul_mat3x3(__m128 out[3], const __m128 m1[3], const __m128 m2[3])
{
  transp_mat3x3_mul_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}

#if defined (__AVX__)
static inline
void transp_mat3x3_mul_mat3x3(__m256d* out_r0, __m256d* out_r1, __m256d* out_r2,
    const __m256d* m1_r0, const __m256d* m1_r1, const __m256d* m1_r2,
    const __m256d* m2_r0, const __m256d* m2_r1, const __m256d* m2_r2)
{
  const __m256d padding = _mm256_setzero_pd();

  __m256d r0 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 0), *m2_r0);
  __m256d r1 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 0), *m2_r1);
  __m256d r2 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 0), *m2_r2);
  *out_r0 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, padding));

  r0 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 1), *m2_r0);
  r1 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 1), *m2_r1);
  r2 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 1), *m2_r2);
  *out_r1 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, padding));

  r0 = _mm256_mul_pd(vec_splat_pd(*m1_r0, 2), *m2_r0);
  r1 = _mm256_mul_pd(vec_splat_pd(*m1_r1, 2), *m2_r1);
  r2 = _mm256_mul_pd(vec_splat_pd(*m1_r2, 2), *m2_r2);
  *out_r2 = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, padding));
}

static inline
void transp_mat3x3_mul_mat3x3(__m256d out[3], const __m256d m1[3], const __m256d m2[3])
{
  transp_mat3x3_mul_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}
#endif


//==============================================================================
/// @brief
/// Compute the product M1*(M2)^T,
/// where M1 is a 3x4 matrix denoted by an array of 4 __m128's,
/// and M2 is a 3x4 matrix denoted by an array of 4 __m128's.
/// The last columns of both M1 and M2 are zero vectors.
///
/// \param[in] matrix M1 of 3x4 (we only use 3x3 submatrix)
/// [ a00 a01 a02 0 ]
/// [ a10 a11 a12 0 ]
/// [ a20 a21 a22 0 ]
///
/// \param[in] matrix M2 of 3x4 (we only use 3x3 submatrix)
/// [ b00 b01 b02 0 ]
/// [ b10 b11 b12 0 ]
/// [ b20 b21 b22 0 ]
///
/// \param[out] matrix out of 3x4 (we only care about the 3x3 submatrix)
/// [ a00b00+a01b01+a02b02 a00b10+a01b11+a02b12 a00b20+a01b21+a02b22 0 ]
/// [ a10b00+a11b01+a12b02 a10b10+a11b11+a12b12 a10b20+a11b21+a12b22 0 ]
/// [ a20b00+a21b01+a22b02 a20b10+a21b11+a22b12 a20b20+a21b21+a22b22 0 ]
#if defined (__SSE4__)
static inline
void mat3x3_mul_transp_mat3x3(__m128* out_r0, __m128* out_r1, __m128* out_r2,
    const __m128* m1_r0, const __m128* m1_r1, const __m128* m1_r2,
    const __m128* m2_r0, const __m128* m2_r1, const __m128* m2_r2)
{
  __m128 prod0 = _mm_dp_ps(*m1_r0, *m2_r0, 0xFF);  // [ a00b00+a01b01+a02b02 a00b00+a01b01+a02b02 a00b00+a01b01+a02b02 a00b00+a01b01+a02b02 ]
  __m128 prod1 = _mm_dp_ps(*m1_r0, *m2_r1, 0xFF);  // [ a00b10+a01b11+a02b12 a00b10+a01b11+a02b12 a00b10+a01b11+a02b12 a00b10+a01b11+a02b12 ]
  __m128 prod2 = _mm_dp_ps(*m1_r0, *m2_r2, 0xFF);  // [ a00b20+a01b21+a02b22 a00b20+a01b21+a02b22 a00b20+a01b21+a02b22 a00b20+a01b21+a02b22 ]
  *out_r0 = _mm_setr_ps(prod0[0], prod1[0], prod2[0], 0.f);

  prod0 = _mm_dp_ps(*m1_r1, *m2_r0, 0xFF);         // [ a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 ]
  prod1 = _mm_dp_ps(*m1_r1, *m2_r1, 0xFF);         // [ a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 ]
  prod2 = _mm_dp_ps(*m1_r1, *m2_r2, 0xFF);         // [ a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 ]
  *out_r1 = _mm_setr_ps(prod0[0], prod1[0], prod2[0], 0.f);

  prod0 = _mm_dp_ps(*m1_r2, *m2_r0, 0xFF);         // [ a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 a10b00+a11b01+a12b02 ]
  prod1 = _mm_dp_ps(*m1_r2, *m2_r1, 0xFF);         // [ a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 a10b10+a11b11+a12b12 ]
  prod2 = _mm_dp_ps(*m1_r2, *m2_r2, 0xFF);         // [ a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 a10b20+a11b21+a12b22 ]
  *out_r2 = _mm_setr_ps(prod0[0], prod1[0], prod2[0], 0.f);
}

static inline
void mat3x3_mul_transp_mat3x3(__m128 out[3], const __m128 m1[3], const __m128 m2[3])
{
  mat3x3_mul_transp_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}
#endif

#if defined (__AVX__)
static inline
void mat3x3_mul_transp_mat3x3(__m256d* out_r0, __m256d* out_r1, __m256d* out_r2,
    const __m256d* m1_r0, const __m256d* m1_r1, const __m256d* m1_r2,
    const __m256d* m2_r0, const __m256d* m2_r1, const __m256d* m2_r2)
{
  *out_r0 = mat3x4_mul_vec4(m2_r0, m2_r1, m2_r2, m1_r0);
  *out_r1 = mat3x4_mul_vec4(m2_r0, m2_r1, m2_r2, m1_r1);
  *out_r2 = mat3x4_mul_vec4(m2_r0, m2_r1, m2_r2, m1_r2);
}

static inline
void mat3x3_mul_transp_mat3x3(__m256d out[3], const __m256d m1[3], const __m256d m2[3])
{
  mat3x3_mul_transp_mat3x3(&out[0], &out[1], &out[2],
      &m1[0], &m1[1], &m1[2],
      &m2[0], &m2[1], &m2[2]);
}
#endif


} // details
} // fcl


#endif
