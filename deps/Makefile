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
	cd ${WORKING_DIR}/opencv/build && cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORKING_DIR}/opencv/install -DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DWITH_VTK=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DBUILD_ZLIB=ON -DWITH_JPEG=OFF -DWITH_WEBP=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_JPEG2000=OFF -DWITH_OPENEXR=OFF -DWITH_V4L=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_PFM=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DBUILD_LIST=core,dnn ../source/opencv*
	cd ${WORKING_DIR}/opencv/build && ninja
	cd ${WORKING_DIR}/opencv/build && ninja install

.PHONY: all clean opencv-clean opencv
