# napari-sam2

> [!NOTE]
> This is a **work in progress** as part of a live project within the SEAI STP. If you come across this repo and are interested please contact [Cameron Shand](mailto:cameron.shand@crick.ac.uk).

A plugin for interactive use of SAM2 for segmentation and tracking

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

## Features/Why this SAM2 plugin?

- SAM2 improvements:
    - Low-memory mode allows easier use of SAM2 for longer videos/larger stacks by avoiding GPU memory accumulation [issue](https://github.com/facebookresearch/sam2/issues/264)
    - Separated inference states to allow for tracking/propagation of new objects over time [issue](https://github.com/facebookresearch/sam2/issues/185)
- No hard CUDA/GPU requirement
- Intuitive usage, utilizing keybinds to speed up annotation
    - Includes proper resetting of model state when adding/removing prompts for existing objects
- Prompt import/export for working across sessions

## Installation
While we include `napari` in the requirements, it is recommended to first install Napari in the best way for your system, for which the [official documentation has some pointers](https://napari.org/stable/tutorials/fundamentals/installation.html).

You can install `napari-sam2` from this repo via:

    pip install git+https://github.com/FrancisCrickInstitute/napari_sam2.git

> [!NOTE]  
> If you are at the Crick, and wish to run this on NEMO, you can create a conda environment using the provided YAML file `napari_sam_env.yaml`, or use the dedicated OnDemand app.

## Usage
For guidance on best usage and further detail on features such as the "low-memory mode", please see the [Usage Guide](USAGE.md).

## Existing To-Dos
- Add support for bounding box prompt input
- Add support for adding masks to guide prompts (low priority)
- Properly handle movement of existing point prompts

## Contributing
Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-sam2" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[file an issue]: https://github.com/FrancisCrickInstitute/napari_sam2/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
