mex -ljpeg jpeg_read.c

if isunix || ismac
    mex CXXFLAGS='$CXXFLAGS -mavx' fct2.cpp
    mex CXXFLAGS='$CXXFLAGS -mavx' ifct2.cpp

    mex CXXFLAGS='$CXXFLAGS -mavx' fct8x8.cpp
    mex CXXFLAGS='$CXXFLAGS -mavx' ifct8x8.cpp
elseif ispc
    mex fct2.cpp
    mex ifct2.cpp

    mex fct8x8.cpp
    mex ifct8x8.cpp
end

mex nlmeans_mex.cpp