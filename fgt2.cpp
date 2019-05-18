#include "matrix.h"
#include "mex.h"
#include "macros.h"
#include "fgt2_Image.h"
#include "fgt2_gkdtree.h"

int transform(float* inimg,float* refimg,float* outimg,float* degimg,
	      int rchannels,int dchannels,int width,int height, int frames, float* rand,
	       int* splatids, float* splatweights, int* splatresults,
	       int* sliceids, float* sliceweights, int* sliceresults,
	       int* blurids,  float* blurweights,  int* blurresults,
	       float* values, float* tmpValues, float accuracy, int numleaves) {
    Image in(frames,width,height,dchannels,inimg);
    Image ref(frames,width,height,rchannels,refimg);
    Image out(frames,width,height,dchannels,outimg);
    Image deg(frames,width,height,1,degimg);
    
    if (numleaves ==  0)
    {
	return GKDTree::filter(in, ref, out, deg, accuracy, rand, 
			       splatids, splatweights, splatresults,
			       sliceids, sliceweights, sliceresults,
			       blurids,  blurweights,  blurresults,
			       values, tmpValues, numleaves);
    }
    else
    {
	return GKDTree::cached(in, ref, out, deg, accuracy, rand, 
			       splatids, splatweights, splatresults,
			       sliceids, sliceweights, sliceresults,
			       blurids,  blurweights,  blurresults,
			       values, tmpValues, numleaves);
    }
}

int getleaves(float* refimg, int rchannels,int width,int height,int frames)
{
    Image ref(frames,width,height,rchannels,refimg);
    return GKDTree::nleaves(ref);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float*     vin;
    float*     vref;
    float*     vout;
    float*     vdeg;
    float*     rand;
    int*       splatids;
    float*     splatweights;
    int*       splatresults;
    int*       sliceids;
    float*     sliceweights;
    int*       sliceresults;
    int*       blurids;
    float*     blurweights;
    int*       blurresults;
    float      accuracy;
    int        numleaves;
    int        frames = 1;

    float*     values;
    float*     tmpValues;

    const mwSize*    ddims;
    const mwSize*    rdims;

    /* Check for proper number of arguments */
    if (nrhs == 1)
    {
	vref = (float *) mxGetData(prhs[0]);
	rdims = mxGetDimensions(prhs[0]);

	if (mxGetNumberOfDimensions(prhs[0]) == 4)
	{
	    frames = rdims[3];
	}

	numleaves = getleaves(vref,rdims[0],rdims[1],rdims[2],frames);
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	*mxGetPr(plhs[0]) = numleaves;

	return;

    }
    else if (nrhs == 18)
    {
	/* populate the real part of the created array */
	vref = (float *) mxGetData(prhs[0]);
	vin  = (float *) mxGetData(prhs[1]);
	vout = (float *) mxGetData(prhs[2]);
	vdeg = (float *) mxGetData(prhs[3]);
	rand = (float *) mxGetData(prhs[4]);
	//splat
	splatids =     (int *)   mxGetData(prhs[5]);
	splatweights = (float *) mxGetData(prhs[6]);
	splatresults = (int *)   mxGetData(prhs[7]);
	//slice
	sliceids =     (int *)   mxGetData(prhs[8]);
	sliceweights = (float *) mxGetData(prhs[9]);
	sliceresults = (int *)   mxGetData(prhs[10]);
	//blur
	blurids =     (int *)   mxGetData(prhs[11]);
	blurweights = (float *) mxGetData(prhs[12]);
	blurresults = (int *)   mxGetData(prhs[13]);
	
	//values
	values =      (float *) mxGetData(prhs[14]);
	tmpValues =   (float *) mxGetData(prhs[15]);
	//cached?
	accuracy  =   (float)   mxGetScalar(prhs[16]);
	numleaves =   (int)     mxGetScalar(prhs[17]);
	
	
	ddims = mxGetDimensions(prhs[1]);
	rdims = mxGetDimensions(prhs[0]);

	if (mxGetNumberOfDimensions(prhs[1]) == 4)
	{
	    frames = ddims[3];
	}

	numleaves = transform(vin, vref, vout, vdeg, rdims[0], ddims[0], ddims[1], ddims[2], frames, rand,
			      splatids, splatweights, splatresults,
			      sliceids, sliceweights, sliceresults,
			      blurids, blurweights, blurresults,
			      values, tmpValues, accuracy, numleaves);

	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	*mxGetPr(plhs[0]) = numleaves;

	return;

    }
    else
    {
	mexErrMsgIdAndTxt("MATLAB:fct2:nargin", "fgt2 requires 1 or 17 arguments.");

	return;
    }
}
