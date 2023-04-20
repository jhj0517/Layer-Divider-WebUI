import os
from setuptools.command.build_ext import build_ext

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pytoshop_path = os.path.join(base_dir, 'venv', 'Lib', 'site-packages', 'pytoshop')


def install_packbits():
    is_extension_there = any(filename.endswith(('.so', '.pyd')) for filename in os.listdir(pytoshop_path))
    if not is_extension_there:
        build_packbits()


def build_packbits():
    from setuptools import setup
    from setuptools.extension import Extension

    try:
        from Cython.Build import cythonize
    except ImportError:
        extensions = []
    else:
        extensions = cythonize([
            Extension(
                f"pytoshop.packbits",
                [os.path.join(pytoshop_path, "packbits.pyx")],
            )
        ])

    setup(
        name="pytoshop-packbits",
        ext_modules=extensions,
        cmdclass={'build_ext': CustomBuildExtCommand},
        script_args=['build_ext'],
    )


class CustomBuildExtCommand(build_ext):
    def get_ext_fullpath(self, ext_name):
        ext_path = ext_name.split('.')
        ext_suffix = self.get_ext_filename(ext_path[-1])
        filepath = os.path.join(pytoshop_path, 'packbits_temp' + ext_suffix)
        return filepath

    def run(self):
        # I don't know why, but the extension file is not being generated with the normal file name, so this needs to be done
        build_ext.run(self)

        for filename in os.listdir(pytoshop_path):
            if filename.startswith('packbits_temp'):
                new_filename = filename.replace('packbits_temp', '', 1)
                src = os.path.join(pytoshop_path, filename)
                dst = os.path.join(pytoshop_path, new_filename)

                os.rename(src, dst)