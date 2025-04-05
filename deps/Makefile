all: opencv

WORKING_DIR ?= ${HOME}/jolopy-deps

clean: opencv-clean

opencv-clean:
	rm -rf ${WORKING_DIR}/opencv/source ${WORKING_DIR}/opencv/build ${WORKING_DIR}/opencv/install

opencv:
	mkdir -p ${WORKING_DIR}/opencv/source ${WORKING_DIR}/opencv/build ${WORKING_DIR}/opencv/install
	wget -O ${WORKING_DIR}/opencv/source.zip https://github.com/opencv/opencv/archive/refs/tags/4.11.0.zip
	unzip ${WORKING_DIR}/opencv/source.zip -d ${WORKING_DIR}/opencv/source
	mkdir -p ${WORKING_DIR}/opencv/build
	cd ${WORKING_DIR}/opencv/build && cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORKING_DIR}/opencv/install -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_SHARED_LIBS=OFF -DWITH_TBB=OFF -DWITH_IPP=OFF -DWITH_GSTREAMER=OFF -DWITH_FFMPEG=OFF -DWITH_VTK=OFF -DWITH_OPENCL=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_LIST=core,dnn -DWITH_1394=OFF -DBUILD_ZLIB=ON -DWITH_V4L=OFF -DWITH_CAROTENE=OFF -DWITH_ITT=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_JPEG=OFF -DWITH_WEBP=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_JPEG2000=OFF -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_PFM=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_PROTOBUF=ON -DWITH_FLATBUFFERS=OFF ../source/opencv*
	cd ${WORKING_DIR}/opencv/build && ninja
	cd ${WORKING_DIR}/opencv/build && ninja install

.PHONY: all clean opencv-clean opencv
