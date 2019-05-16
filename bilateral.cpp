#include <stdio.h>
#include "macros.h"
#include "Image.h"
#include "gkdtree.h"
#include "jpg.h"

bool endswith(const char *str, const char *suffix) {
    return (strcmp(str + strlen(str) - strlen(suffix), suffix) == 0);
}

int main(int argc, char **argv) {
    if (argc < 5) {
	printf("Usage: ./bilateral input.png output.png <spatial standard deviation> <color standard deviation>\n");
	return 1;
    }

    Image input;
    if (endswith(argv[1], ".jpg")) {
	input = JPG::load(argv[1]);
    } else {
	printf("Input image is not a .jpg file\n");
    }

    Image out(1, input.width, input.height, 4);


    float invSpatialStdev = 1.0f/atof(argv[3]);
    float invColorStdev = 1.0f/atof(argv[4]);

    // make the reference image
    Image ref(1, input.width, input.height, 5);
    for (int y = 0; y < input.height; y++) {
	for (int x = 0; x < input.width; x++) {
	    ref(x, y)[0] = invSpatialStdev * x;
	    ref(x, y)[1] = invSpatialStdev * y;
	    ref(x, y)[2] = invColorStdev * input(x, y)[0];
	    ref(x, y)[3] = invColorStdev * input(x, y)[1];
	    ref(x, y)[4] = invColorStdev * input(x, y)[2];
	}
    }

    GKDTree::filter(input, out, ref, 0.5);
    
    if (endswith(argv[2], ".jpg")) {
	JPG::save(out, argv[2], 100);
    } else {
	printf("Output image is not a .jpg file\n");
    }    
    
    return 0;
}
