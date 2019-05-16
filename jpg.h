#ifndef JPG___H
#define JPG___H

// JPG loading and saving code

#include <string>
#include <jpeglib.h>

using namespace std;

namespace JPG {

    void save(Window im, string filename, int quality) {

        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;

        FILE *f = fopen(filename.c_str(), "wb");

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, f);

        cinfo.image_width = im.width;
        cinfo.image_height = im.height;
	cinfo.in_color_space = JCS_RGB;  
        cinfo.input_components = 3;

        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, 100, TRUE);
        jpeg_start_compress(&cinfo, TRUE);

        JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, im.width * 4, 1);

        while (cinfo.next_scanline < cinfo.image_height) {
            // convert the row
            JSAMPLE *dstPtr = buffer[0];
            for (int x = 0; x < im.width; x++) {
		float deg = 1.0f/im(x, cinfo.next_scanline)[3];
		for (int c = 0; c < 3; c++) {
                    *dstPtr++ = deg*im(x, cinfo.next_scanline)[c];
                }
            }
            jpeg_write_scanlines(&cinfo, buffer, 1);
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);

        fclose(f);
    }

    Image load(string filename) {

        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr jerr;

        FILE *f = fopen(filename.c_str(), "rb");

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, f);

        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);

	Image im(1, cinfo.output_width, cinfo.output_height, 4);
        JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, im.width * 4, 1);

        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            JSAMPLE *srcPtr = buffer[0];
            for (int x = 0; x < im.width; x++) {
		im(x, cinfo.output_scanline-1)[3] = 1.0f;
                for (int c = 0; c < 3; c++) {
                    im(x, cinfo.output_scanline-1)[c] = *srcPtr++;
                }
            }
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        fclose(f);

	return im;
    }
}

#endif
