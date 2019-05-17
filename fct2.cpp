#include "mex.h"
#include "matrix.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <immintrin.h>

const float c1  = 0.490392640201615f;
const float c3  = 0.415734806151273f;
const float c5  = 0.277785116509801f; 
const float c7  = 0.097545161008064f;
const float cs1 = 0.191341716182545f; 
const float cs2 = 0.270598050073098f;
const float cs3 =-0.653281482438188f;
const float ca = - c1 + c3 + c5 - c7;
const float cb =   c1 + c3 - c5 + c7;
const float cc =   c1 + c3 + c5 - c7;
const float cd =   c1 + c3 - c5 - c7;
const float ce =      - c3      + c7;
const float cf = - c1 - c3;
const float cg =      - c3 - c5;
const float ch =      - c3 + c5;
const float ci =        c3;

#if (defined WIN32)||(defined WIN64)||(defined _WIN32)||(defined _WIN64)
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
#endif

void transform(float *img, size_t stride, float* buf)
{
    __m256 i0 = _mm256_loadu_ps(img + 0*stride);
    __m256 i1 = _mm256_loadu_ps(img + 1*stride);
    __m256 i2 = _mm256_loadu_ps(img + 2*stride);
    __m256 i3 = _mm256_loadu_ps(img + 3*stride);
    __m256 i4 = _mm256_loadu_ps(img + 4*stride);
    __m256 i5 = _mm256_loadu_ps(img + 5*stride);
    __m256 i6 = _mm256_loadu_ps(img + 6*stride);
    __m256 i7 = _mm256_loadu_ps(img + 7*stride);

    //DCT on columns
    // stage 1
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    t0 = i0 + i7; t1 = i1 + i6; t2 = i2 + i5; t3 = i3 + i4;
    t4 = i3 - i4; t5 = i2 - i5; t6 = i1 - i6; t7 = i0 - i7;
    
    // stage 2 even
    __m256 u0, u1, u2, u3;
    u0 = t0 + t3; u1 = t1 + t2; u2 = t1 - t2; u3 = t0 - t3;

    // stage 3 even even
    __m256 o0 = (u0 + u1) * _mm256_set1_ps(float(M_SQRT1_2) / 2);
    __m256 o4 = (u0 - u1) * _mm256_set1_ps(float(M_SQRT1_2) / 2);
    
    // stage 4 even odd
    __m256 tt = _mm256_mul_ps(u2 + u3, _mm256_set1_ps(cs1));
    __m256 o2 = tt + _mm256_set1_ps(cs2) * u3;
    __m256 o6 = tt + _mm256_set1_ps(cs3) * u2;

    // stages 3,4,5 odd
    __m256 z1 = (t4 + t7) * _mm256_set1_ps(ce);
    __m256 z2 = (t5 + t6) * _mm256_set1_ps(cf);
    __m256 z3 = t4 + t6;
    __m256 z4 = t5 + t7;
    __m256 z5 = (z3 + z4) * _mm256_set1_ps(ci);
    z3 = z3 * _mm256_set1_ps(cg) + z5;
    z4 = z4 * _mm256_set1_ps(ch) + z5;
    
    t4 = t4 * _mm256_set1_ps(ca);
    t5 = t5 * _mm256_set1_ps(cb);
    t6 = t6 * _mm256_set1_ps(cc);
    t7 = t7 * _mm256_set1_ps(cd);
    
    __m256 o1 = t7 + z1 + z4;
    __m256 o3 = t6 + z2 + z3;
    __m256 o5 = t5 + z2 + z4;
    __m256 o7 = t4 + z1 + z3;

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
    // stage 1
    t0 = i0 + i7; t1 = i1 + i6; t2 = i2 + i5; t3 = i3 + i4;
    t4 = i3 - i4; t5 = i2 - i5; t6 = i1 - i6; t7 = i0 - i7;
    
    // stage 2 even
    u0 = t0 + t3; u1 = t1 + t2; u2 = t1 - t2; u3 = t0 - t3;
    
    // stage 3 even even
    o0 = (u0 + u1) * _mm256_set1_ps(float(M_SQRT1_2) / 2);
    o4 = (u0 - u1) * _mm256_set1_ps(float(M_SQRT1_2) / 2);
    
    // stage 4 even odd
    tt = _mm256_mul_ps(u2 + u3, _mm256_set1_ps(cs1));
    o2 = tt + _mm256_set1_ps(cs2) * u3;
    o6 = tt + _mm256_set1_ps(cs3) * u2;
    
    // stages 3,4,5 odd
    z1 = (t4 + t7) * _mm256_set1_ps(ce);
    z2 = (t5 + t6) * _mm256_set1_ps(cf);
    z3 = t4 + t6;
    z4 = t5 + t7;
    z5 = (z3 + z4) * _mm256_set1_ps(ci);
    z3 = z3 * _mm256_set1_ps(cg) + z5;
    z4 = z4 * _mm256_set1_ps(ch) + z5;
    
    t4 = t4 * _mm256_set1_ps(ca);
    t5 = t5 * _mm256_set1_ps(cb);
    t6 = t6 * _mm256_set1_ps(cc);
    t7 = t7 * _mm256_set1_ps(cd);
    
    o1 = t7 + z1 + z4;
    o3 = t6 + z2 + z3;
    o5 = t5 + z2 + z4;
    o7 = t4 + z1 + z3;
    
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
    
    _mm256_storeu_ps(buf +  0, i0);
    _mm256_storeu_ps(buf +  8, i1);
    _mm256_storeu_ps(buf + 16, i2);
    _mm256_storeu_ps(buf + 24, i3);
    _mm256_storeu_ps(buf + 32, i4);
    _mm256_storeu_ps(buf + 40, i5);
    _mm256_storeu_ps(buf + 48, i6);
    _mm256_storeu_ps(buf + 56, i7);
}
//    IACA_END

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float*     vin1;
    float*     vout;
    int        deci = 1;
    size_t        pics = 1;

    const mwSize*    dimi;
    const mwSize*    dimo;

    /* Check for proper number of arguments */

    if (nrhs < 2 || mxGetClassID(prhs[0]) != mxSINGLE_CLASS 
                 || mxGetClassID(prhs[1]) != mxSINGLE_CLASS)
    {
	mexErrMsgIdAndTxt("MATLAB:fct2:nargin", "fct2 requires two float arguments.");
    }

    dimi = mxGetDimensions(prhs[0]);
    dimo = mxGetDimensions(prhs[1]);
    /* populate the real part of the created array */
    vin1 = (float *) mxGetData(prhs[0]);
    vout = (float *) mxGetData(prhs[1]);

    if (nrhs == 3)
    {
	deci = (int) mxGetScalar(prhs[2]);
    }
    if (mxGetNumberOfDimensions(prhs[0]) == 3)
    {
	pics = dimi[2];
    }

    int istri = 1;
    size_t jstri = dimi[0];
    size_t kstri = dimi[0]*dimi[1];

    size_t istro = dimo[0]/deci;
    size_t jstro = dimo[0]*dimo[1]/deci;
    size_t kstro = dimo[0]*dimo[1]*dimo[2];

    for (int k = 0; k < pics; k++)
    {
        #pragma omp parallel for schedule(static, 16)
	for (int j = 0; j < dimi[1] - 7; j+=deci)
	{
	    for (int i = 0; i < dimi[0] - 7; i+=deci)
	    {
		transform(vin1+k*kstri+j*jstri+i,dimi[0],
			  vout+k*kstro+j*jstro+i*istro);
	    }
	}
    }
    return;
}
