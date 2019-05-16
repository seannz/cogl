#include <stdio.h>
#include "mex.h"
#include "matrix.h"
#include "macros.h"
#include "Image.h"
#include "jpg.h"
#include "gkdtree.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:bilateral:nargin", "Usage: bilateral(input file, output file, spatial standard deviation, color standard deviation)\n");
        return;
    }
    
    float invSpatialStdev = 1.0f/mxGetScalar(prhs[2]);
    float invColorStdev = 1.0f/mxGetScalar(prhs[3]);

    float* vimg = (float*) mxGetData(prhs[0]);
    float* vout = (float*) mxGetData(prhs[1]);
    // Load the input image
    const mwSize* img_s = mxGetDimensions(prhs[0]);
    const mwSize* out_s = mxGetDimensions(prhs[1]);
    Image input(1, (int) img_s[1], (int) img_s[2], 4, vimg);
    Image image(1, (int) out_s[1], (int) out_s[2], 4, vout);

    Image positions(1, input.width, input.height, 5);
    for (int y = 0; y < input.height; y++) {
	for (int x = 0; x < input.width; x++) {
	    positions(x, y)[0] = invSpatialStdev * x;
	    positions(x, y)[1] = invSpatialStdev * y;
	    positions(x, y)[2] = invColorStdev * input(x, y)[0];
	    positions(x, y)[3] = invColorStdev * input(x, y)[1];
	    positions(x, y)[4] = invColorStdev * input(x, y)[2];
	}
    }

    Times times = GKDTree::filter(input, image, positions, 0.5);

    plhs[0] = mxCreateNumericMatrix(4,1, mxSINGLE_CLASS, mxREAL);
    times.copyTo((float*) mxGetData(plhs[0]));
}
