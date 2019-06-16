#include "matrix.h"
#include "mex.h"
#include "macros.h"
#include "Image.h"
#include "gkdtree.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("MATLAB:fgt2:nargin", "Usage: fgt2(input, output, guide, accuracy)\n");
        return;
    }

    mexAtExit(GKDTree::release);

    float invSpatialStdev = 1.0f/mxGetScalar(prhs[2]);
    float invColorStdev = 1.0f/mxGetScalar(prhs[3]);

    float accuracy = 1.0f;
    bool replay = false;
    if (nrhs == 5) 
      replay = (bool) mxGetScalar(prhs[4]);

    // Load the input image
    const mwSize* img_s = mxGetDimensions(prhs[0]);
    const mwSize* gud_s = mxGetDimensions(prhs[1]);
    plhs[0] = mxCreateNumericArray(3, img_s, mxSINGLE_CLASS, mxREAL);

    float* vimg = (float*) mxGetData(prhs[0]);
    float* vgud = (float*) mxGetData(prhs[1]);
    float* vout = (float*) mxGetData(plhs[0]);

    Image input(1, (int) img_s[1], (int) img_s[2], (int) img_s[0], vimg);
    Image guide(1, (int) gud_s[1], (int) gud_s[2], (int) gud_s[0], vgud);
    Image image(1, (int) img_s[1], (int) img_s[2], (int) img_s[0], vout);

    Image positions(guide.frames, guide.width, guide.height, guide.channels + 2);
    for (int f = 0; f < guide.frames; f++) { 
	for (int x = 0; x < guide.width; x++) {
	    for (int y = 0; y < guide.height; y++) {
		positions(f, x, y)[0] = invSpatialStdev * x;
		positions(f, x, y)[1] = invSpatialStdev * y;
		for (int d = 0; d < guide.channels; d++)
		    positions(x, y)[d+2] = invColorStdev * guide(x, y)[d];
	    }
	}
    }

    Times times = GKDTree::filter(input, image, positions, accuracy, !replay);

    plhs[1] = mxCreateNumericMatrix(4, 1, mxSINGLE_CLASS, mxREAL);
    times.copyTo((float*) mxGetData(plhs[1]));
}
