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

/** @author Jia Pan */

#include "fcl/math/bv/OBB-inl.h"

#include "fcl/math/math_simd_details.h"

namespace fcl
{

#if defined (__SSE4__) or defined(__AVX__)
  using namespace fcl::details;
#endif

//==============================================================================
template
class OBB<double>;

//==============================================================================
template
void computeVertices(const OBB<double>& b, Vector3<double> vertices[8]);

//==============================================================================
template
OBB<double> merge_largedist(const OBB<double>& b1, const OBB<double>& b2);

//==============================================================================
template
OBB<double> merge_smalldist(const OBB<double>& b1, const OBB<double>& b2);

//==============================================================================
#if defined (__AVX__)
inline bool obbDisjoint(const __m256d* R, const __m256d& t, const __m256d& r1, const __m256d& r2)
{
  const double reps = 1e-6;
  const __m256d epsilonxyz = _mm256_setr_pd(reps, reps, reps, 0);
  __m256d AbsR[3];
  AbsR[0] = _mm256_add_pd(abs_pd(R[0]), epsilonxyz);
  AbsR[1] = _mm256_add_pd(abs_pd(R[1]), epsilonxyz);
  AbsR[2] = _mm256_add_pd(abs_pd(R[2]), epsilonxyz);

  // Test the three major axes of this OBB.
  __m256d res = _mm256_cmp_pd(abs_pd(t), _mm256_add_pd(r1, mat3x4_mul_vec4(AbsR, r2)), _CMP_GT_OQ);
  if (!allzero_pd(res)) return true;

  // Test the three major axes of the OBB b.
  __m256d l = abs_pd(transp_mat3x4_mul_vec4(R, t));
  __m256d s = transp_mat3x4_mul_vec4(AbsR, r1);
  res = _mm256_cmp_pd(l, _mm256_add_pd(s, r2), _CMP_GT_OQ);
  if (!allzero_pd(res)) return true;

  // Test the 9 different cross-axes.
  __m256d symmetric_matrix[3] = {
    _mm256_setr_pd(    0, r2[2], r2[1], 0),
    _mm256_setr_pd(r2[2],     0, r2[0], 0),
    _mm256_setr_pd(r2[1], r2[0],     0, 0),
  };

  // A.x <cross> B.x
  // A.x <cross> B.y
  // A.x <cross> B.z
  // __m256d ra = fmadd_pd(yyyw_pd(r1), AbsR[2], _mm256_mul_pd(zzzw_pd(r1), AbsR[1]));
  __m256d ra = fmadd_pd(vec_splat_pd(r1, 1), AbsR[2], _mm256_mul_pd(vec_splat_pd(r1, 2), AbsR[1]));
  __m256d rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[0]);
  // __m256d lhs = fmsub_pd(zzzw_pd(t), R[1], _mm256_mul_pd(yyyw_pd(t), R[2]));
  __m256d lhs = fmsub_pd(vec_splat_pd(t, 2), R[1], _mm256_mul_pd(vec_splat_pd(t, 1), R[2]));
  res = _mm256_cmp_pd(abs_pd(lhs), _mm256_add_pd(ra, rb), _CMP_GT_OQ);
  if (!allzero_pd(res)) return true;

  // A.y <cross> B.x
  // A.y <cross> B.y
  // A.y <cross> B.z
  // ra = fmadd_pd(xxxw_pd(r1), AbsR[2], _mm256_mul_pd(zzzw_pd(r1), AbsR[0]));
  ra = fmadd_pd(vec_splat_pd(r1, 0), AbsR[2], _mm256_mul_pd(vec_splat_pd(r1, 2), AbsR[0]));
  rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[1]);
  // lhs = fmsub_pd(xxxw_pd(t), R[2], _mm256_mul_pd(zzzw_pd(t), R[0]));
  lhs = fmsub_pd(vec_splat_pd(t, 0), R[2], _mm256_mul_pd(vec_splat_pd(t, 2), R[0]));
  res = _mm256_cmp_pd(abs_pd(lhs), _mm256_add_pd(ra, rb), _CMP_GT_OQ);
  if (!allzero_pd(res)) return true;

  // A.z <cross> B.x
  // A.z <cross> B.y
  // A.z <cross> B.z
  // ra = fmadd_pd(xxxw_pd(r1), AbsR[1], _mm256_mul_pd(yyyw_pd(r1), AbsR[0]));
  ra = fmadd_pd(vec_splat_pd(r1, 0), AbsR[1], _mm256_mul_pd(vec_splat_pd(r1, 1), AbsR[0]));
  rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[2]);
  // lhs = fmsub_pd(yyyw_pd(t), R[0], _mm256_mul_pd(xxxw_pd(t), R[1]));
  lhs = fmsub_pd(vec_splat_pd(t, 1), R[0], _mm256_mul_pd(vec_splat_pd(t, 0), R[1]));
  res = _mm256_cmp_pd(abs_pd(lhs), _mm256_add_pd(ra, rb), _CMP_GT_OQ);
  return (!allzero_pd(res));
}
#endif

//==============================================================================
#if defined (__SSE4__)
inline bool obbDisjoint(const __m128* R, const __m128& t, const __m128& r1, const __m128& r2)
{
  std::cout << "obbDisjoint SSE4" << std::endl;
  const float reps = 1e-6;
  const __m128 epsilonxyz = _mm_setr_ps(reps, reps, reps, 0.f);
  __m128 AbsR[3];
  AbsR[0] = _mm_add_ps(abs_ps(R[0]), epsilonxyz);
  AbsR[1] = _mm_add_ps(abs_ps(R[1]), epsilonxyz);
  AbsR[2] = _mm_add_ps(abs_ps(R[2]), epsilonxyz);

  // Test the three major axes of this OBB.
  __m128 res = _mm_cmpgt_ps(abs_ps(t), _mm_add_ps(r1, mat3x4_mul_vec4(AbsR, r2)));
  if (!allzero_ps(res)) return true;

  // Test the three major axes of the OBB b.
  __m128 l = abs_ps(transp_mat3x4_mul_vec4(R, t));
  __m128 s = transp_mat3x4_mul_vec4(AbsR, r1);
  res = _mm_cmpgt_ps(l, _mm_add_ps(s, r2));
  if (!allzero_ps(res)) return true;

  // Test the 9 different cross-axes.
  __m128 symmetric_matrix[3] = {
    _mm_setr_ps(  0.f, r2[2], r2[1], 0.f),
    _mm_setr_ps(r2[2],   0.f, r2[0], 0.f),
    _mm_setr_ps(r2[1], r2[0],   0.f, 0.f),
  };

  // A.x <cross> B.x
  // A.x <cross> B.y
  // A.x <cross> B.z
  // __m128 ra = fmadd_ps(yyyw_ps(r1), AbsR[2], _mm_mul_ps(zzzw_ps(r1), AbsR[1]));
  __m128 ra = fmadd_ps(vec_splat_ps(r1, 1), AbsR[2], _mm_mul_ps(vec_splat_ps(r1, 2), AbsR[1]));
  __m128 rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[0]);
  // __m128 lhs = fmsub_ps(zzzw_ps(t), R[1], _mm_mul_ps(yyyw_ps(t), R[2]));
  __m128 lhs = fmsub_ps(vec_splat_ps(t, 2), R[1], _mm_mul_ps(vec_splat_ps(t, 1), R[2]));
  res = _mm_cmpgt_ps(abs_ps(lhs), _mm_add_ps(ra, rb));
  if (!allzero_ps(res)) return true;

  // A.y <cross> B.x
  // A.y <cross> B.y
  // A.y <cross> B.z
  // ra = fmadd_ps(xxxw_ps(r1), AbsR[2], _mm_mul_ps(zzzw_ps(r1), AbsR[0]));
  ra = fmadd_ps(vec_splat_ps(r1, 0), AbsR[2], _mm_mul_ps(vec_splat_ps(r1, 2), AbsR[0]));
  rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[1]);
  // lhs = fmsub_ps(xxxw_ps(t), R[2], _mm_mul_ps(zzzw_ps(t), R[0]));
  lhs = fmsub_ps(vec_splat_ps(t, 0), R[2], _mm_mul_ps(vec_splat_ps(t, 2), R[0]));
  res = _mm_cmpgt_ps(abs_ps(lhs), _mm_add_ps(ra, rb));
  if (!allzero_ps(res)) return true;

  // A.z <cross> B.x
  // A.z <cross> B.y
  // A.z <cross> B.z
  // ra = fmadd_ps(xxxw_ps(r1), AbsR[1], _mm_mul_ps(yyyw_ps(r1), AbsR[0]));
  ra = fmadd_ps(vec_splat_ps(r1, 0), AbsR[1], _mm_mul_ps(vec_splat_ps(r1, 1), AbsR[0]));
  rb = mat3x4_mul_vec4(symmetric_matrix, AbsR[2]);
  // lhs = fmsub_ps(yyyw_ps(t), R[0], _mm_mul_ps(xxxw_ps(t), R[1]));
  lhs = fmsub_ps(vec_splat_ps(t, 1), R[0], _mm_mul_ps(vec_splat_ps(t, 0), R[1]));
  res = _mm_cmpgt_ps(abs_ps(lhs), _mm_add_ps(ra, rb));
  return (!allzero_ps(res));
}
#endif

//==============================================================================
#if defined (__AVX__)
template <>
bool obbDisjoint(const Matrix3<double>& B, const Vector3<double>& T,
                 const Vector3<double>& a, const Vector3<double>& b)
{
  std::cout << "obbDisjoint AVX" << std::endl;
  __m256d B_avx[3] = {
    _mm256_setr_pd(B(0, 0), B(0, 1), B(0, 2), 0),
    _mm256_setr_pd(B(1, 0), B(1, 1), B(1, 2), 0),
    _mm256_setr_pd(B(2, 0), B(2, 1), B(2, 2), 0),
  };
  __m256d T_avx = _mm256_setr_pd(T[0], T[1], T[2], 0);
  __m256d a_avx = _mm256_setr_pd(a[0], a[1], a[2], 0);
  __m256d b_avx = _mm256_setr_pd(b[0], b[1], b[2], 0);
  return obbDisjoint(B_avx, T_avx, a_avx, b_avx);
}
#else
template
bool obbDisjoint(
    const Matrix3<double>& B,
    const Vector3<double>& T,
    const Vector3<double>& a,
    const Vector3<double>& b);

#endif

//==============================================================================
#if defined (__SSE4__)
template <>
bool obbDisjoint(const Matrix3<float>& B, const Vector3<float>& T,
                 const Vector3<float>& a, const Vector3<float>& b)
{
  std::cout << "obbDisjoint SSE4" << std::endl;
  __m128 B_sse[3] = {
    _mm_setr_ps(B(0, 0), B(0, 1), B(0, 2), 0.f),
    _mm_setr_ps(B(1, 0), B(1, 1), B(1, 2), 0.f),
    _mm_setr_ps(B(2, 0), B(2, 1), B(2, 2), 0.f),
  };
  __m128 T_sse = _mm_setr_ps(T[0], T[1], T[2], 0.f);
  __m128 a_sse = _mm_setr_ps(a[0], a[1], a[2], 0.f);
  __m128 b_sse = _mm_setr_ps(b[0], b[1], b[2], 0.f);
  return obbDisjoint(B_sse, T_sse, a_sse, b_sse);
}
#else
template
bool obbDisjoint(
    const Matrix3<float>& B,
    const Vector3<float>& T,
    const Vector3<float>& a,
    const Vector3<float>& b);
#endif

//==============================================================================
template
bool obbDisjoint(
    const Transform3<double>& tf,
    const Vector3<double>& a,
    const Vector3<double>& b);

} // namespace fcl
