BUILDDIR ?= builddir

all:
	meson setup --buildtype release --strip --prefer-static ${BUILDDIR}
	meson compile -C ${BUILDDIR}

clean:
	rm -rf ${BUILDDIR}

.PHONY: all clean
