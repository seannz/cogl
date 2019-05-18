#ifndef GKDTREE_H 
#define GKDTREE_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include <limits>
#ifndef INF
#define INF (std::numeric_limits<float>::infinity())
#endif

//#define RAND_FLOAT ((double) (rand()) / (RAND_MAX+1.0))
#define RAND_FLOAT (rands[randi=((randi+1)%300000)])

#include "fgt2_gaussian.h"


static int randi = 0;
static float *rands = NULL;

void doNothing(...) {}

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

class GKDTree {
public:
    static int nleaves(Image ref) {
	float **points = new float *[ref.width*ref.height*ref.frames];
	int i = 0;
	for (int t = 0; t < ref.frames; t++) {
	    for (int x = 0; x < ref.width; x++) {
		for (int y = 0; y < ref.height; y++) {
		    points[i++] = ref(t, x, y);
		}
	    }
	}
	
	GKDTree tree(ref.channels, points, ref.frames*ref.width*ref.height, 0.15f);
	tree.finalize();

	delete[] points;
	return tree.getLeaves();
    }

    static int filter(Image im, Image ref, Image out, Image deg, float accuracy, float* rand,
		      int* splatids, float* splatweights, int* splatresults,
		      int* sliceids, float* sliceweights, int* sliceresults,
		      int* blurids, float* blurweights, int* blurresults,
		      float* values, float* tmpValues, int numleaves) {

	printf("Building...\n");

	rands = rand;

	float **points = new float *[ref.width*ref.height*ref.frames];
	int i = 0;
	for (int t = 0; t < ref.frames; t++) {
	    for (int x = 0; x < ref.width; x++) {
		for (int y = 0; y < ref.height; y++) {
		    points[i++] = ref(t, x, y);
		}
	    }
	}
	
	GKDTree tree(ref.channels, points, ref.frames*ref.width*ref.height, 0.15f);
	tree.finalize();
	numleaves = tree.getLeaves();

	printf("Splatting...\n");

	int SPLAT_ACCURACY = 4 + (int)(accuracy*64);
	int BLUR_ACCURACY = 16 + (int)(accuracy*128);
	int SLICE_ACCURACY = 12 + (int)(accuracy*64);

    const float SPLAT_STD_DEV = 0.30156f;
	const float BLUR_STD_DEV = 0.9045f;
	const float SLICE_STD_DEV = 0.30156f;
	
	int numdimens = im.channels;
	int numpixels = im.frames*im.height*im.width;
	memset(values, 0, sizeof(float)*numleaves*numdimens);
	memset(tmpValues, 0, sizeof(float)*numleaves*numdimens);
	float *refPtr = ref(0, 0, 0);

	for (int idx = 0; idx < numpixels; idx++) {
	    int& results = splatresults[idx];
	    int *indices = splatids + idx*SPLAT_ACCURACY;
	    float* weights = splatweights + idx*SPLAT_ACCURACY;
	    results = tree.gaussianLookup(refPtr, SPLAT_STD_DEV, indices, weights, SPLAT_ACCURACY);
	    for (int i = 0; i < results; i++) {
		float w = weights[i];
		float *vPtr = values + indices[i]*numdimens;
		vPtr[numdimens-1] += w;
	    }
	    refPtr += ref.channels;
	}
    
	printf("Blurring...\n");
	tree.blur(BLUR_STD_DEV, values, tmpValues, numdimens, blurids, blurweights, blurresults, BLUR_ACCURACY);

	float *tmp = tmpValues;
	tmpValues = values;
	values = tmp;
    
	printf("Slicing...\n");  

	float *degPtr = deg(0, 0, 0);
	refPtr = ref(0, 0, 0);
	for (int idx = 0; idx < numpixels; idx++) {
	    int& results = sliceresults[idx];
	    int *indices = sliceids + idx*SLICE_ACCURACY;
	    float *weights = sliceweights + idx*SLICE_ACCURACY;
	    results = tree.gaussianLookup(refPtr, SLICE_STD_DEV, indices, weights, SLICE_ACCURACY);
	    float totalWeight = 0;
	    for (int i = 0; i < results; i++) {
		float w = weights[i];
		float *vPtr = values + indices[i]*numdimens;
		totalWeight += w*vPtr[numdimens-1];
	    }
	    degPtr[0] = totalWeight;
	    
	    degPtr += deg.xstride;
	    refPtr += ref.xstride;
	}

	delete[] points;
	return numleaves;
    };


    static int cached(Image im, Image ref, Image out, Image deg, float accuracy, float* rand,
		      int* splatids, float* splatweights, int* splatresults,
		      int* sliceids, float* sliceweights, int* sliceresults,
		      int* blurids, float* blurweights, int* blurresults,
		      float* values, float* tmpValues, int numleaves) {

	rands = rand;

	int SPLAT_ACCURACY = 4 + (int)(accuracy*64);
	int BLUR_ACCURACY = 16 + (int)(accuracy*128);
	int SLICE_ACCURACY = 12 + (int)(accuracy*64);

    const float SPLAT_STD_DEV = 0.30156f;
	const float BLUR_STD_DEV = 0.9045f;
	const float SLICE_STD_DEV = 0.30156f;
	
	int numdimens = im.channels;
	int numpixels = im.frames*im.height*im.width;
	memset(values, 0, sizeof(float)*numleaves*numdimens);
	memset(tmpValues, 0, sizeof(float)*numleaves*numdimens);

	for (int idx = 0; idx < numpixels; idx++) {
	    float *imPtr = im(0, 0, 0) + idx*im.xstride;
	    int *indices = splatids + idx*SPLAT_ACCURACY;
	    float* weights = splatweights + idx*SPLAT_ACCURACY;

	    __m256 ac0,ac1,ac2,ac3,ac4,ac5,ac6,ac7;

	    for (int i = 0; i < splatresults[idx]; i++) {
		float *vPtr = values + indices[i]*numdimens;
		__m256 mul = _mm256_broadcast_ss(weights+i);

		ac0 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+ 0));
		ac1 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+ 8));
		ac2 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+16));
		ac3 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+24));
		ac4 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+32));
		ac5 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+40));
		ac6 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+48));
		ac7 = _mm256_mul_ps(mul,_mm256_loadu_ps(imPtr+56));

		_mm256_storeu_ps(vPtr+ 0,_mm256_loadu_ps(vPtr+ 0)+ac0);
		_mm256_storeu_ps(vPtr+ 8,_mm256_loadu_ps(vPtr+ 8)+ac1);
		_mm256_storeu_ps(vPtr+16,_mm256_loadu_ps(vPtr+16)+ac2);
		_mm256_storeu_ps(vPtr+24,_mm256_loadu_ps(vPtr+24)+ac3);
		_mm256_storeu_ps(vPtr+32,_mm256_loadu_ps(vPtr+32)+ac4);
		_mm256_storeu_ps(vPtr+40,_mm256_loadu_ps(vPtr+40)+ac5);
		_mm256_storeu_ps(vPtr+48,_mm256_loadu_ps(vPtr+48)+ac6);
		_mm256_storeu_ps(vPtr+56,_mm256_loadu_ps(vPtr+56)+ac7);
	    }
	}
    
        #pragma omp parallel for
	for (int id = 0; id < numleaves; id++)
	{
	    int* ids = blurids + id*BLUR_ACCURACY;
	    float* weights = blurweights + id*BLUR_ACCURACY;

	    __m256 ac0,ac1,ac2,ac3,ac4,ac5,ac6,ac7;
	    ac0=ac1=ac2=ac3=ac4=ac5=ac6=ac7=_mm256_setzero_ps();

	    float* newValues = tmpValues + id*numdimens;
	    for (int i = 0; i < blurresults[id]; i++) {
		float* oldValues = values + ids[i]*numdimens;
		__m256 mul = _mm256_broadcast_ss(weights+i);
		ac0 = ac0 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+ 0));
		ac1 = ac1 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+ 8));
		ac2 = ac2 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+16));
		ac3 = ac3 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+24));
		ac4 = ac4 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+32));
		ac5 = ac5 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+40));
		ac6 = ac6 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+48));
		ac7 = ac7 + _mm256_mul_ps(mul,_mm256_loadu_ps(oldValues+56));
	    }

	    _mm256_storeu_ps(newValues+ 0,ac0);
	    _mm256_storeu_ps(newValues+ 8,ac1);
	    _mm256_storeu_ps(newValues+16,ac2);
	    _mm256_storeu_ps(newValues+24,ac3);
	    _mm256_storeu_ps(newValues+32,ac4);
	    _mm256_storeu_ps(newValues+40,ac5);
	    _mm256_storeu_ps(newValues+48,ac6);
	    _mm256_storeu_ps(newValues+56,ac7);
	}

	float *tmp = tmpValues;
	tmpValues = values;
	values = tmp;
    
        #pragma omp parallel for
	for (int idx = 0; idx < numpixels; idx++) {
		    
	    float  *outPtr = out(0,0,0) + idx*out.xstride;
	    int   *indices = sliceids + idx*SLICE_ACCURACY;
	    float *weights = sliceweights + idx*SLICE_ACCURACY;
    	    __m256 ac0,ac1,ac2,ac3,ac4,ac5,ac6,ac7;
            ac0=ac1=ac2=ac3=ac4=ac5=ac6=ac7=_mm256_setzero_ps();

	    for (int i = 0; i < sliceresults[idx]; i++) {
		float *vPtr = values + indices[i]*numdimens;
		__m256 mul = _mm256_broadcast_ss(weights+i);
		ac0 = ac0 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+ 0));
		ac1 = ac1 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+ 8));
		ac2 = ac2 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+16));
		ac3 = ac3 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+24));
		ac4 = ac4 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+32));
		ac5 = ac5 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+40));
		ac6 = ac6 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+48));
		ac7 = ac7 + _mm256_mul_ps(mul,_mm256_loadu_ps(vPtr+56));
	    }

	    _mm256_storeu_ps(outPtr+ 0,ac0);
	    _mm256_storeu_ps(outPtr+ 8,ac1);
	    _mm256_storeu_ps(outPtr+16,ac2);
	    _mm256_storeu_ps(outPtr+24,ac3);
	    _mm256_storeu_ps(outPtr+32,ac4);
	    _mm256_storeu_ps(outPtr+40,ac5);
	    _mm256_storeu_ps(outPtr+48,ac6);
	    _mm256_storeu_ps(outPtr+56,ac7);
	}

	return numleaves;
    };

    
    // Build a gkdtree using the supplied array of points to control
    // the sampling.

    // sizeBound specifies the maximum allowable side length of a
    // kdtree leaf.

    // At least one point from data lies in any given leaf.

    GKDTree(int dims, float **data, int nData, float sBound) : 
    	dimensions(dims), sizeBound(sBound), leaves(0) {

    	// Allocate space to store a data bounding box while we build
    	// the tree 
    	dataMins = new float[dims];
    	dataMaxs = new float[dims];

	root = build(data, nData);	

    	gaussian = NULL;

    	delete[] dataMins;
    	delete[] dataMaxs;
    }       	
	
    ~GKDTree() {
	delete root;
	if (gaussian)
	    delete gaussian;
    }

    void include(float *value) {
	root = root->include(value, sizeBound, &leaves);
    }

    void finalize() {
	float *kdtreeMins = new float[dimensions];
	float *kdtreeMaxs = new float[dimensions];

	for (int i = 0; i < dimensions; i++) {
	    kdtreeMins[i] = -INF;
	    kdtreeMaxs[i] = +INF;
	}
	
	root->computeBounds(kdtreeMins, kdtreeMaxs);
	
	delete[] kdtreeMins;
	delete[] kdtreeMaxs;		

	printf("Constructed a gkdtree with %i leaves\n", leaves);	
    }

    int getLeaves() {
	return leaves;
    }

    int getDimensions() {
	return dimensions;
    }

    // Compute a gaussian spread of kdtree leaves around the given
    // point. This is the general case sampling strategy.
    int gaussianLookup(float *value, float sigma, int *ids, float *weights, int nSamples) {
	return root->gaussianLookup(value, makeGaussian(sigma), &ids, &weights, nSamples, 1);
    }

    // compute which kdtree cell a given point lies within. Equivalent
    // to a gaussian lookup of sigma zero
    int nearestLookup(float *value) {
	return root->nearestLookup(value);
    }

    // assign some data to each leaf node and do a gaussian blur
    void blur(float sigma, float *oldValues, float *newValues, int dataDimensions, int* blurids, float* blurweights,
	      int* blurresults, int nSamples) {
	//int ids[nSamples];
	//float weights[nSamples];	

	memset(newValues, 0, sizeof(float)*leaves*dataDimensions);
	root->blur(this, sigma, oldValues, newValues, dataDimensions, blurids, blurweights, blurresults, nSamples);

	//delete[] ids;
	//delete[] weights;
    }
	
	


private:

    Gaussian *makeGaussian(float sigma) {
	if (gaussian) {
	    if (gaussian->getSigma() != sigma) {
		//delete gaussian;
		//gaussian = new Gaussian(sigma);
		gaussian->setSigma(sigma);
	    }
	} else {
	    gaussian = new Gaussian(sigma);
	}
	return gaussian;
    }

    class Node {	
    public:
	virtual ~Node() {}

	// Returns a list of samples from the kdtree distributed
	// around value with std-dev sigma in all dimensions. Some
	// samples may be repeated. Returns how many entries in the
	// ids and weights arrays were used.
	virtual int gaussianLookup(float *value, Gaussian *gaussian, int **ids, float **weights, int nSamples, float p) = 0;

	// special case optimization of the above where nsamples = 1
	virtual int singleGaussianLookup(float *value, Gaussian *gaussian, int **ids, float **weights, float p) = 0;

	// Same as the above if the gaussian has std dev 0
	virtual int nearestLookup(float *value) = 0;

	virtual void blur(GKDTree *tree, float sigma, float *oldValues, float *newValues, int dataDimensions, int *blurids, float *blurweights, int *blurresults, int nSamples) = 0;

	virtual Node *include(float *value, float sizeBound, int *leaves) = 0;

	virtual void computeBounds(float *mins, float *maxs) = 0;

    };
    
    class Split : public Node {
    public:
	virtual ~Split() {
	    delete left;
	    delete right;
	}


	// for a given gaussian and a given value, the probability of splitting left at this node
	inline float pLeft(Gaussian *gaussian, float value) {	    
	    // Coarsely approximate the cumulative normal distribution 
	    float val = gaussian->sampleCDF(cut_val - value);
	    float minBound = gaussian->sampleCDF(min_val - value);
	    float maxBound = gaussian->sampleCDF(max_val - value);	    
	    return (val - minBound) / (maxBound - minBound);	    
	}

	int gaussianLookup(float *value, Gaussian *gaussian, int **ids, float **weights, int nSamples, float p) {
	    // Calculate how much of a gaussian ball of radius sigma,
	    // that has been trimmed by all the cuts so far, lies on
	    // each side of the split

	    // compute the probability of a sample splitting left
	    float val = pLeft(gaussian, value[cut_dim]);

	    // Send some samples to the left of the split
	    int leftSamples = (int)(val*nSamples);

	    // Send some samples to the right of the split
	    int rightSamples = (int)((1-val)*nSamples);
	    
	    // There's probably one sample left over by the rounding
	    if (leftSamples + rightSamples != nSamples) {
		float fval = val*nSamples - rightSamples;
		// if val is high we send it right, if val is low we send it left
		if (RAND_FLOAT < fval) {
		    leftSamples++;
		} else {
		    rightSamples++;
		}
	    }	    

	    int samplesFound = 0;
	    // Get the left samples
	    if (leftSamples > 0) {
		if (leftSamples > 1) {
		    samplesFound += left->gaussianLookup(value, gaussian, ids, weights, leftSamples, p*val);
		} else {
		    samplesFound += left->singleGaussianLookup(value, gaussian, ids, weights, p*val);
		}
	    }

	    // Get the right samples
	    if (rightSamples > 0) {
		if (rightSamples > 1) {
		    samplesFound += right->gaussianLookup(value, gaussian, ids, weights, rightSamples, p*(1-val));
		} else {
		    samplesFound += right->singleGaussianLookup(value, gaussian, ids, weights, p*(1-val));
		}
	    }	    	    

	    return samplesFound;
	}
	
	// a special case optimization of the above for when nSamples is 1
	int singleGaussianLookup(float *value, Gaussian *gaussian, int **ids, float **weights, float p) {
	    float val = pLeft(gaussian, value[cut_dim]);
	    if (RAND_FLOAT < val) {
		return left->singleGaussianLookup(value, gaussian, ids, weights, p*val);
	    } else {
		return right->singleGaussianLookup(value, gaussian, ids, weights, p*(1-val));
	    }
	}

	// which leaf does a given value lie within? Equivalent to a
	// gaussian lookup with std dev = 0
	int nearestLookup(float *value) {
	    if (value[cut_dim] < cut_val) return left->nearestLookup(value);
	    else return right->nearestLookup(value);
	}
	
	void blur(GKDTree *tree, float sigma, float *oldValues, float *newValues, int dataDimensions, int *blurids, float *blurweights, int *blurresults, int nSamples) {
	    left->blur(tree, sigma, oldValues, newValues, dataDimensions, blurids, blurweights, blurresults, nSamples);
	    right->blur(tree, sigma, oldValues, newValues, dataDimensions, blurids, blurweights, blurresults, nSamples);
	}

	void computeBounds(float *mins, float *maxs) {
	    min_val = mins[cut_dim];
	    max_val = maxs[cut_dim];

	    maxs[cut_dim] = cut_val;
	    left->computeBounds(mins, maxs);
	    maxs[cut_dim] = max_val;

	    mins[cut_dim] = cut_val;		
	    right->computeBounds(mins, maxs);
	    mins[cut_dim] = min_val;
	}

	Node *include(float *value, float sizeBound, int *leaves) {
	    /*
	      if (value[cut_dim] < min_val || value[cut_dim] > max_val) {
	      printf("A node for inclusion was sent down the wrong path!\n");
	      printf("%f %f %f\n", min_val, value[cut_dim], max_val);
	      }
	    */

	    if (value[cut_dim] < cut_val) left = left->include(value, sizeBound, leaves);
	    else right = right->include(value, sizeBound, leaves);
	    return this;
	}

	int cut_dim;
	float cut_val, min_val, max_val;
	Node *left, *right;
    };
    
    class Leaf : public Node {
    public:
	Leaf(int id_, float **data, int nData, int dimensions_) 
	    : id(id_), dimensions(dimensions_) {
	    position = new float[dimensions];
	    float invData = 1.0f/nData;
	    for (int i = 0; i < dimensions; i++) {
		position[i] = 0;
		for (int j = 0; j < nData; j++) {
		    position[i] += data[j][i];		    
		}
		position[i] *= invData;
	    }
	}
	    
	~Leaf() {
	    delete[] position;
	}
	
	int gaussianLookup(float *query, Gaussian *g, int **ids, float **weights, int nSamples, float p) {
	    // p is the probability with which one sample arrived here
	    // calculate the correct probability, q
	    
	    // TODO: sse?
	    float q = 0;
	    for (int i = 0; i < dimensions; i++) {
		float diff = query[i] - position[i];
		diff *= diff;
		q += diff;
	    }

	    q = g->sampleSquared(q);

	    *(*ids)++ = id;
	    *(*weights)++ = nSamples * q / p;
	    
	    return 1;
	}
	
	int singleGaussianLookup(float *query, Gaussian *g, int **ids, float **weights, float p) {
	    return gaussianLookup(query, g, ids, weights, 1, p);
	}
	
	int nearestLookup(float *) {
	    return id;
	}
	
	void blur(GKDTree *tree, float sigma, float *oldValues, float *newValues, int dataDimensions, int *blurids, float *blurweights, int *blurresults, int nSamples) {
	    // do a gaussian gather	    
	    //int results = tree->gaussianLookup(position, sigma, ids, weights, nSamples);

	    int& results = blurresults[id];
	    int* ids = blurids + id*nSamples;
	    float* weights = blurweights + id*nSamples;

	    results = tree->gaussianLookup(position, sigma, ids, weights, nSamples);
	    
	    int newIdx = id*dataDimensions;
	    for (int i = 0; i < results; i++) {
	    	int oldIdx = ids[i]*dataDimensions;
	    	// for (int d = 0; d < dataDimensions; d++) {
	    	//     newValues[newIdx+d] += weights[i] * oldValues[oldIdx+d];
	    	// }
		newValues[newIdx+dataDimensions-1] += weights[i] * oldValues[oldIdx+dataDimensions-1];
	    }
	    
	}

	Node *include(float *value, float sizeBound, int *leaves) {
	    float d = 0;
	    for (int i = 0; i < dimensions; i++) {
		float diff = value[i] - position[i];
		diff *= diff;
		d += diff;
	    }
	    if (d > sizeBound * sizeBound) {
		/*
		  printf("This value:\n");
		  for (int i = 0; i < dimensions; i++) {
		  printf("%f ", value[i]);
		  }
		  printf("\nIs too far from this node:\n");
		  for (int i = 0; i < dimensions; i++) {
		  printf("%f ", position[i]);
		  }		
		  printf("\n");
		*/
		// this value isn't well represented by this tree -
		// split up this node
		int longest = 0;
		d = (value[0] - position[0])*(value[0] - position[0]);
		for (int i = 0; i < dimensions; i++) {
		    float diff = value[i] - position[i];
		    diff *= diff;
		    if (diff > d) {
			longest = i;
			d = diff;
		    }
		}

		Split *s = new Split;
		s->cut_dim = longest;
		s->cut_val = (position[longest] + value[longest])/2;
		//printf("So I'll make a new split on dimension %i at position %f\n", s->cut_dim, s->cut_val);
		s->min_val = -INF;
		s->max_val = INF;
		Leaf *l = new Leaf((*leaves)++, &value, 1, dimensions);
		if (position[longest] < s->cut_val) {
		    //printf("I go on the left, the new guy goes on the right\n");
		    s->left = this;
		    s->right = l;
		} else {
		    //printf("I go on the right, the new guy goes on the left\n");
		    s->left = l;
		    s->right = this;
		}
		return s;
	    } else {
		// this new value is already well represented in the tree
		return this;
	    }
	}

	void computeBounds(float *mins, float *maxs) {
	    /*
	    bool ok = true;
	    for (int i = 0; i < dimensions && ok; i++) {
		ok = (position[i] >= mins[i]) && (position[i] <= maxs[i]);
	    }
	    if (!ok) {
		printf("Corrupt kdtree:\n");
		printf("mins: ");
		for (int i = 0; i < dimensions; i++)
		    printf("%3.3f ", mins[i]);
		printf("\n");

		printf("position: ");
		for (int i = 0; i < dimensions; i++)
		    printf("%3.3f ", position[i]);
		printf("\n");

		printf("maxs: ");
		for (int i = 0; i < dimensions; i++)
		    printf("%3.3f ", maxs[i]);
		printf("\n");
	    }
	    */
	}
	
      private:
	int id, dimensions;
	float *position;
    };
    
    Node *root;
    int dimensions;
    float sizeBound;
    int leaves;

    float *dataMins, *dataMaxs;
    Gaussian *gaussian;

    Node *build(float **data, int nData) {
	if (nData == 1) {
	    return new Leaf(leaves++, data, nData, dimensions);
	} else {
	    // calculate the data bounds in every dimension
	    for (int i = 0; i < dimensions; i++) {
		dataMins[i] = dataMaxs[i] = data[0][i];
	    }
	    for (int j = 1; j < nData; j++) {
		for (int i = 0; i < dimensions; i++) {
		    if (data[j][i] < dataMins[i]) dataMins[i] = data[j][i];
		    if (data[j][i] > dataMaxs[i]) dataMaxs[i] = data[j][i];
		}
	    }

	    // find the longest dimension
	    int longest = 0;
	    float diagonal = 0;
	    for (int i = 1; i < dimensions; i++) {
		float delta = dataMaxs[i] - dataMins[i];
		diagonal += delta*delta;
		if (delta > dataMaxs[longest] - dataMins[longest]) 
		    longest = i;
	    }

	    // if it's large enough, cut in that dimension
	    if (diagonal > 4*sizeBound*sizeBound) {
		Split *n = new Split;
		n->cut_dim = longest;
		n->cut_val = (dataMaxs[longest] + dataMins[longest])/2;

		// these get computed later		
		n->min_val = -INF; 
		n->max_val = INF; 
		
		// resort the input over the split
		int pivot = 0;
		for (int i = 0; i < nData; i++) {
		    // The next value is larger than the pivot
		    if (data[i][longest] >= n->cut_val) continue;
		    
		    // We haven't seen anything larger than the pivot yet
		    if (i == pivot) {
			pivot++;
			continue;
		    }
		
		    // The current value is smaller than the pivot
		    float *tmp = data[i];
		    data[i] = data[pivot];
		    data[pivot] = tmp;
		    pivot++;
		}

		// Build the two subtrees
		n->left = build(data, pivot);
		n->right = build(data+pivot, nData-pivot);

		return n;
	    } else { 
		return new Leaf(leaves++, data, nData, dimensions);
	    }
	}
    };


};


#endif
