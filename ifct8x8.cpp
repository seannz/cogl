#include "mex.h"
#include "matrix.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <immintrin.h>

__m256 operator+(const __m256& a, const __m256& b)
{
  return _mm256_add_ps(a, b);
}

__m256 operator-(const __m256& a, const __m256& b)
{
  return _mm256_sub_ps(a, b);
}

__m256 operator*(const __m256& a, const __m256& b)
{
  return _mm256_mul_ps(a, b);
}

void transform(float *img, size_t stride, float* buf)
{
    //DCT on columns
    //This code is adapted from libjpeg with constants added as an after thought
    __m256 tmp0 = _mm256_loadu_ps(img + 0*stride) * _mm256_set1_ps(0.353553390593274f);
    __m256 tmp1 = _mm256_loadu_ps(img + 2*stride) * _mm256_set1_ps(0.461939766255643f);
    __m256 tmp2 = _mm256_loadu_ps(img + 4*stride) * _mm256_set1_ps(0.353553390593274f);
    __m256 tmp3 = _mm256_loadu_ps(img + 6*stride) * _mm256_set1_ps(0.191341716182545f);

    __m256 tmp10 = tmp0 + tmp2;
/* phase 3 */
    __m256 tmp11 = tmp0 - tmp2;

    __m256 tmp13 = tmp1 + tmp3;
/* phases 5-3 */
    __m256 tmp12 = (tmp1 - tmp3) * _mm256_set1_ps(float(M_SQRT2)) - tmp13; /* 2*c4 */

    tmp0 = tmp10 + tmp13;
/* phase 2 */
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;

    /* Odd part */

    __m256 tmp4 = _mm256_loadu_ps(img + 1*stride) * _mm256_set1_ps(0.490392640201615f);
    __m256 tmp5 = _mm256_loadu_ps(img + 3*stride) * _mm256_set1_ps(0.415734806151273f);
    __m256 tmp6 = _mm256_loadu_ps(img + 5*stride) * _mm256_set1_ps(0.277785116509801f);
    __m256 tmp7 = _mm256_loadu_ps(img + 7*stride) * _mm256_set1_ps(0.097545161008064f);

    __m256 z13 = tmp6 + tmp5;
/* phase 6 */
    __m256 z10 = tmp6 - tmp5;
    __m256 z11 = tmp4 + tmp7;
    __m256 z12 = tmp4 - tmp7;

    tmp7 = z11 + z13;
/* phase 5 */
    tmp11 = (z11 - z13) * _mm256_set1_ps(M_SQRT2); /* 2*c4 */

    __m256 z5 = (z10 + z12) * _mm256_set1_ps(1.847759065f); /* 2*c2 */
    tmp10 = z5 - z12 * _mm256_set1_ps(1.082392200f); /* 2*(c2-c6) */
    tmp12 = z5 - z10 * _mm256_set1_ps(2.613125930f); /* 2*(c2+c6) */

    tmp6 = tmp12 - tmp7;
/* phase 2 */
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 - tmp5;

    __m256 o0 = tmp0 + tmp7;
    __m256 o7 = tmp0 - tmp7;
    __m256 o1 = tmp1 + tmp6;
    __m256 o6 = tmp1 - tmp6;
    __m256 o2 = tmp2 + tmp5;
    __m256 o5 = tmp2 - tmp5;
    __m256 o3 = tmp3 + tmp4;
    __m256 o4 = tmp3 - tmp4;

    __m256 i0, i1, i2, i3, i4, i5, i6, i7;
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;

    // transpose the coefficients to perform rows DCT
    i0 = _mm256_unpacklo_ps(o0, o1);//00 10 01 11 04 14 05 15
    i1 = _mm256_unpackhi_ps(o0, o1);//02 12 03 13 06 16 07 17
    i2 = _mm256_unpacklo_ps(o2, o3);//20 30 21 31 24 34 25 35
    i3 = _mm256_unpackhi_ps(o2, o3);//22 32 23 33 26 36 27 37
    i4 = _mm256_unpacklo_ps(o4, o5);//40 50 41 51 44 54 45 55
    i5 = _mm256_unpackhi_ps(o4, o5);//42 52 43 53 46 56 47 57
    i6 = _mm256_unpacklo_ps(o6, o7);//60 70 61 71 64 74 65 75
    i7 = _mm256_unpackhi_ps(o6, o7);//62 72 63 73 66 76 67 77
    t0 = _mm256_shuffle_ps(i0,i2,0x44);//00 10 20 30 04 14 24 34
    t1 = _mm256_shuffle_ps(i0,i2,0xee);//01 11 21 31 05 15 25 35
    t2 = _mm256_shuffle_ps(i1,i3,0x44);//02 12 22 32 06 16 26 36
    t3 = _mm256_shuffle_ps(i1,i3,0xee);//03 13 23 33 07 17 27 37
    t4 = _mm256_shuffle_ps(i4,i6,0x44);//40 50 60 70 44 54 64 74
    t5 = _mm256_shuffle_ps(i4,i6,0xee);//41 51 61 71 45 55 65 75
    t6 = _mm256_shuffle_ps(i5,i7,0x44);//42 52 62 72 46 56 66 76
    t7 = _mm256_shuffle_ps(i5,i7,0xee);//43 53 63 73 47 57 67 77
    i0 = _mm256_permute2f128_ps(t0, t4, 0x20);//00 10 20 30 40 50 60 70
    i1 = _mm256_permute2f128_ps(t1, t5, 0x20);//01 11 21 31 41 51 61 71
    i2 = _mm256_permute2f128_ps(t2, t6, 0x20);//02 12 22 32 42 52 62 72
    i3 = _mm256_permute2f128_ps(t3, t7, 0x20);//03 13 23 33 47 57 67 77
    i4 = _mm256_permute2f128_ps(t0, t4, 0x31);//04 14 24 34 44 54 64 74
    i5 = _mm256_permute2f128_ps(t1, t5, 0x31);//05 15 25 35 45 55 65 75
    i6 = _mm256_permute2f128_ps(t2, t6, 0x31);//06 16 26 36 46 56 66 76
    i7 = _mm256_permute2f128_ps(t3, t7, 0x31);//07 17 27 37 47 57 67 77
    
    //DCT on rows
    //This code is adapted from libjpeg
    tmp0 = i0 * _mm256_set1_ps(0.353553390593274f);
    tmp1 = i2 * _mm256_set1_ps(0.461939766255643f);
    tmp2 = i4 * _mm256_set1_ps(0.353553390593274f);
    tmp3 = i6 * _mm256_set1_ps(0.191341716182545f);

    tmp10 = tmp0 + tmp2;
/* phase 3 */
    tmp11 = tmp0 - tmp2;

    tmp13 = tmp1 + tmp3;
/* phases 5-3 */
    tmp12 = (tmp1 - tmp3) * _mm256_set1_ps(float(M_SQRT2)) - tmp13; /* 2*c4 */

    tmp0 = tmp10 + tmp13;
/* phase 2 */
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;

    /* Odd part */

    tmp4 = i1 * _mm256_set1_ps(0.490392640201615f);
    tmp5 = i3 * _mm256_set1_ps(0.415734806151273f);
    tmp6 = i5 * _mm256_set1_ps(0.277785116509801f);
    tmp7 = i7 * _mm256_set1_ps(0.097545161008064f);

    z13 = tmp6 + tmp5;
/* phase 6 */
    z10 = tmp6 - tmp5;
    z11 = tmp4 + tmp7;
    z12 = tmp4 - tmp7;

    tmp7 = z11 + z13;
/* phase 5 */
    tmp11 = (z11 - z13) * _mm256_set1_ps(M_SQRT2); /* 2*c4 */

    z5 = (z10 + z12) * _mm256_set1_ps(1.847759065f); /* 2*c2 */
    tmp10 = z5 - z12 * _mm256_set1_ps(1.082392200f); /* 2*(c2-c6) */
    tmp12 = z5 - z10 * _mm256_set1_ps(2.613125930f); /* 2*(c2+c6) */

    tmp6 = tmp12 - tmp7;
/* phase 2 */
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 - tmp5;

    o0 = tmp0 + tmp7;
    o7 = tmp0 - tmp7;
    o1 = tmp1 + tmp6;
    o6 = tmp1 - tmp6;
    o2 = tmp2 + tmp5;
    o5 = tmp2 - tmp5;
    o3 = tmp3 + tmp4;
    o4 = tmp3 - tmp4;

    //desirable arrangement to change from
    //00 10 20 30 40 50 60 70
    //01 11 21 31 41 51 61 71
    //02 12 22 32 42 52 62 72
    //03 13 23 33 47 57 67 77
    //04 14 24 34 44 54 64 74
    //05 15 25 35 45 55 65 75
    //06 16 26 36 46 56 66 76
    //07 17 27 37 47 57 67 77
    // to zigzag order
//     0  1  8 16  9  2  3 10
//    17 24 32 25 18 11  4  5
//    12 19 26 33 40 48 41 34
//    27 20 13  6  7 14 21 28
//    35 42 49 56 57 50 43 36
//    29 22 15 23 30 37 44 51
//    58 59 52 45 38 31 39 46
//    53 60 61 54 47 55 62 63
//  which is
    //00 01 10 20 11 02 03 12
    //21 30 40 31 22 13 04 05
    //14 23 32 41 50 60 51 42
    //33 24 15 06 07 16 25 34
    //43 52 61 70 71 62 53 44
    //35 26 17 27 36 45 54 63
    //72 73 64 55 46 37 47 56
    //65 74 75 66 57 67 76 77
    
    
    //transpose back
    i0 = _mm256_unpacklo_ps(o0, o1);//00 10 01 11 04 14 05 15
    i1 = _mm256_unpackhi_ps(o0, o1);//02 12 03 13 06 16 07 17
    i2 = _mm256_unpacklo_ps(o2, o3);//20 30 21 31 24 34 25 35
    i3 = _mm256_unpackhi_ps(o2, o3);//22 32 23 33 26 36 27 37
    i4 = _mm256_unpacklo_ps(o4, o5);//40 50 41 51 44 54 45 55
    i5 = _mm256_unpackhi_ps(o4, o5);//42 52 43 53 46 56 47 57
    i6 = _mm256_unpacklo_ps(o6, o7);//60 70 61 71 64 74 65 75
    i7 = _mm256_unpackhi_ps(o6, o7);//62 72 63 73 66 76 67 77
    t0 = _mm256_shuffle_ps(i0, i2, 0x44);//00 10 20 30 04 14 24 34
    t1 = _mm256_shuffle_ps(i0, i2, 0xee);//01 11 21 31 05 15 25 35
    t2 = _mm256_shuffle_ps(i1, i3, 0x44);//02 12 22 32 06 16 26 36
    t3 = _mm256_shuffle_ps(i1, i3, 0xee);//03 13 23 33 07 17 27 37
    t4 = _mm256_shuffle_ps(i4, i6, 0x44);//40 50 60 70 44 54 64 74
    t5 = _mm256_shuffle_ps(i4, i6, 0xee);//41 51 61 71 45 55 65 75
    t6 = _mm256_shuffle_ps(i5, i7, 0x44);//42 52 62 72 46 56 66 76
    t7 = _mm256_shuffle_ps(i5, i7, 0xee);//43 53 63 73 47 57 67 77
    i0 = _mm256_permute2f128_ps(t0, t4, 0x20);//00 10 20 30 40 50 60 70
    i1 = _mm256_permute2f128_ps(t1, t5, 0x20);//01 11 21 31 41 51 61 71
    i2 = _mm256_permute2f128_ps(t2, t6, 0x20);//02 12 22 32 42 52 62 72
    i3 = _mm256_permute2f128_ps(t3, t7, 0x20);//03 13 23 33 47 57 67 77
    i4 = _mm256_permute2f128_ps(t0, t4, 0x31);//04 14 24 34 44 54 64 74
    i5 = _mm256_permute2f128_ps(t1, t5, 0x31);//05 15 25 35 45 55 65 75
    i6 = _mm256_permute2f128_ps(t2, t6, 0x31);//06 16 26 36 46 56 66 76
    i7 = _mm256_permute2f128_ps(t3, t7, 0x31);//07 17 27 37 47 57 67 77
    
    _mm256_storeu_ps(buf + 0*stride, i0);
    _mm256_storeu_ps(buf + 1*stride, i1);
    _mm256_storeu_ps(buf + 2*stride, i2);
    _mm256_storeu_ps(buf + 3*stride, i3);
    _mm256_storeu_ps(buf + 4*stride, i4);
    _mm256_storeu_ps(buf + 5*stride, i5);
    _mm256_storeu_ps(buf + 6*stride, i6);
    _mm256_storeu_ps(buf + 7*stride, i7);

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float*     vin1;
    float*     vout;
    int        pics = 1;
    const mwSize*    dims;

    /* Check for proper number of arguments */

    if (nrhs != 2 || mxGetClassID(prhs[0]) != mxSINGLE_CLASS 
	|| mxGetClassID(prhs[1]) != mxSINGLE_CLASS)
    {
	mexErrMsgIdAndTxt("MATLAB:fct2:nargin", "ifct2 requires two float arguments.");
    }

    dims = mxGetDimensions(prhs[0]);
    /* populate the real part of the created array */
    vin1 = (float *) mxGetData(prhs[0]);
    vout = (float *) mxGetData(prhs[1]);

    if (mxGetNumberOfDimensions(prhs[0]) == 3)
    {
	pics = dims[2];
    }

    for (int k = 0; k < pics; k++)
    {
        #pragma omp parallel for
	for (int j = 0; j < dims[1]-7; j+=8)
	{
	    for (int i = 0; i < dims[0]-7; i+=8)
	    {
		transform(vin1+(k*dims[1]+j)*dims[0]+i,dims[0],vout+(k*dims[1]+j)*dims[0]+i);
	    }
	}
    }
    return;
}
