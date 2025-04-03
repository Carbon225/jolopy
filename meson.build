project('jolopy', 'c', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=3', 'c_std=c11', 'cpp_std=c++14']
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

opencv_dep = dependency('opencv4')

incdir = include_directories('include')

subdir('src')
