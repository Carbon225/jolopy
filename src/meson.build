yolocpp_lib = static_library('yolocpp', 'yolocpp.cpp', include_directories: incdir, dependencies: opencv_dep)
yolocpp_dep = declare_dependency(link_with: yolocpp_lib)

yoloc_lib = static_library('yoloc', 'yoloc.cpp', include_directories: incdir, dependencies: yolocpp_dep)
yoloc_dep = declare_dependency(link_with: yoloc_lib)

numpy_inc = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()
numpy_dep = declare_dependency(include_directories: numpy_inc)

py.extension_module('jolopy',
                   sources: 'jolopy_module.c',
                   include_directories: incdir,
                   dependencies: [py_dep, yoloc_dep, numpy_dep])
