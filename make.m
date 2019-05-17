if isunix || ismac
    mex -ljpeg jpeg_read.c
    
    mex CXXFLAGS='$CXXFLAGS -mavx' fct2.cpp
    mex CXXFLAGS='$CXXFLAGS -mavx' ifct2.cpp

    mex CXXFLAGS='$CXXFLAGS -mavx' fct8x8.cpp
    mex CXXFLAGS='$CXXFLAGS -mavx' ifct8x8.cpp
elseif ispc
    mex -L./winlibjpeg -I./winlibjpeg -ljpeg-static.lib jpeg_read.c
    
    mex fct2.cpp
    mex ifct2.cpp

    mex fct8x8.cpp
    mex ifct8x8.cpp
end

mex nlmeans_mex.cpp
mex nltemps_mex.cpp