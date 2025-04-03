yolocpp_lib = static_library('yolocpp', 'yolocpp.cpp', include_directories: incdir, dependencies: opencv_dep)
yolocpp_dep = declare_dependency(link_with: yolocpp_lib)

yoloc_lib = static_library('yoloc', 'yoloc.cpp', include_directories: incdir, dependencies: yolocpp_dep)
yoloc_dep = declare_dependency(link_with: yoloc_lib)

py.extension_module('jolopy',
                   sources: 'jolopy_module.c',
                   include_directories: incdir,
                   dependencies: [py_dep, yoloc_dep])
