!/bin/sh

ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 50 -pix_fmt yuv420p -i ducks_1280x720_50.yuv -vf scale=w=iw/2:h=ih/2:flags=spline -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -r 50 ducks_640x360_50.yuv
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 50 -pix_fmt yuv420p -i mobcal_1280x720_50.yuv -vf scale=w=iw/2:h=ih/2:flags=spline -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -r 50 mobcal_640x360_50.yuv
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 50 -pix_fmt yuv420p -i parkrun_1280x720_50.yuv -vf scale=w=iw/2:h=ih/2:flags=spline -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -r 50 parkrun_640x360_50.yuv
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 50 -pix_fmt yuv420p -i shields_1280x720_50.yuv -vf scale=w=iw/2:h=ih/2:flags=spline -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -r 50 shields_640x360_50.yuv
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 59.94 -pix_fmt yuv420p -i stockholm_1280x720_59.9401.yuv -vf scale=w=iw/2:h=ih/2:flags=spline -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -r 59.94 stockholm_640x360_59.9401.yuv


