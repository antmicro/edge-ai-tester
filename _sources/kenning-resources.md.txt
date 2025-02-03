# Kenning resources

For managing resources, such as models, compilers, datasets and more Kenning provides [](resourcemanager-api).
It handles:

* Downloading of remote files,
* Resolving custom URL schemes for cleaner and easy to follow file addresses,
* Managing directory with files, including quota checking, old files removal and full cleanup,
* Providing actual path to the resources in the filesystem.

To run below examples it is required to install Kenning as follows:
```bash
pip install "kenning @ git+https://github.com/antmicro/kenning.git"
```

## Accessing resources

To download a resource you can use [`ResourceManager.get_resource`](resourcemanager-api) method.
The parameters of this method are a URI of the resource and optional output path, where the file should be saved.
If the output path is not specified, the file will be saved in `$KENNING_CACHE_DIR/uri.path`.

```python
import pathlib
from kenning.utils.resource_manager import ResourceManager, ResourceURI

model_path = pathlib.Path('./model.h5')
ResourceManager().get_resource('kenning:///models/classification/magic_wand.h5', model_path)
```

Another way of accessing those resources is to use [](resourceuri-api) class.
You can simply create an instance of this class and use it similarly to [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path):

```python
model_path = ResourceURI('kenning:///models/classification/magic_wand.h5')
```

The provided path can be URL to remote resource, path to local file or URI which scheme is defined in `ResourceManager` conversions dictionary.

(kenning-cache-dir)=
## Caching

If the resource was not accessed before, the cached file was modified or the remote resource was updated, then it will be downloaded to local cache directory.
The location of cache directory can be set using the environment variable `KENNING_CACHE_DIR`.
If it is not set, then the default `~/.kenning` path will be used.
You can also customize it using [`ResourceManager.set_cache_dir`](resourcemanager-api) method.

By default, the maximum size of the cache is set to 50 GB, but it can be changed using the environment variable `KENNING_MAX_CACHE_SIZE` or in the runtime using [`ResourceManager.set_max_cache_size`](resourcemanager-api) method.
Before each download the file size is checked and if the available cache space is not large enough, then the oldest files are deleted until there is enough space.
If the file is bigger than maximum size of the cache, then the exception is raised.

## Custom URI schemes

As you can see above, the `ResourceManager` supports non-standard URI scheme `kenning://`.
In fact, it can support various schemes defined by the user.

The default available schemes are:

* `http://`, `https://`
* `file://` - provides the absolute path to the file in the filesystem
* `kenning://` - a scheme for downloading Kenning-specific resources from `https://dl.antmicro.com`.
  For example, `kenning:///models/classification/magic_wand.h5` resolves to `https://dl.antmicro.com/kenning/models/classification/magic_wand.h5`
* `gh://` - used for downloading files from Github repositories, e.g. `gh://antmicro:kenning-bare-metal-iree-runtime/sim/config/platforms/springbok.repl;branch=main` resolves to `https://raw.githubusercontent.com/antmicro/kenning-bare-metal-iree-runtime/main/sim/config/platforms/springbok.repl`
* `hf://` - used for downloading files from Hugging Face model hub, e.g. `hf://company/model-id` resolves to `https://huggingface.co/company/model-id`. Using this scheme may require logging in to `huggingface_hub` using `huggingface-cli`.


This allows using links such as:
- `https://dl.antmicro.com/kenning/models/classification/magic_wand.h5`
- `kenning:///models/classification/magic_wand.h5`
- `gh://antmicro:kenning-bare-metal-iree-runtime/sim/config/platforms/springbok.repl;branch=main`
- `file:///some/directory/magic_wand.h5`
- `hf://mistralai/Mistral-7B-Instruct-v0.1`


If the provided path does not have any scheme, the it will be interpreted as a path.
For example:

```python
ResourceURI('/some/path/magic_wand.h5')
```

Would behave the same as `Path('/some/path/magic_wand.h5')`.

Before the URI is passed to format string or callable converter it is parsed into parts: scheme, netloc, path, params, query and fragment.
Additionally the netloc and path are parsed into lists by splitting using either `'.'` or `'/'` separator and the params and query parts are parsed into dictionaries.

The conversion can be:

* `None` - no conversion is done.
* a format string - user can use URI attributes such as: scheme, netloc, path, params, query and fragment enclosed in curly braces.
  The netloc and path can be also used as lists (i.e. it is possible to use `{path[0]}` or `{netloc[1:]}` inside the format string) and the params and query can be used as dictionary (i.e. you can use `{params[branch]}` for URIs like `gh://antmicro:kenning-bare-metal-iree-runtime/sim/config/platforms/springbok.repl;branch=main`).
- a callable - the provided URI is parsed and passed into it.
  The callable can have parameters such as `scheme`, `netloc`, `path`, `params`, `query`, `netloc_list`, `path_list`, `params_dict`, `query_dict` and should return string.
  The parameters with `_list` or `_dicts` suffix are corresponding parts parsed into lists or dicts.
  Example implementation of such conversion can be seen above as `_gh_converter`.

The above default scheme conversions can be defined as:

```python test-skip
BASE_URL_SCHEMES = {
    'http': None,
    'https': None,
    'kenning': 'https://dl.antmicro.com/kenning/{path}',
    'gh': _gh_converter,
    'file': lambda path: Path(path).expanduser().resolve(),
}
```

The keys in this dictionary are scheme names, and values are conversion methods.

The `_gh_converter` is defined as:

```python
from typing import Dict

def _gh_converter(netloc: str, path: str, params_dict: Dict[str, str]) -> str:
    netloc = netloc.split(':')
    return (
        f'https://raw.githubusercontent.com/{netloc[0]}/{netloc[1]}/'
        f'{params_dict["branch"]}{path}'
    )
```

Those schemes can be customized by user using the [`ResourceManager.add_custom_url_schemes`](resourcemanager-api) method which takes a dictionary similar to the above as an input - the key is the name of scheme and the value is the converter (`None`, string or callable).
If any scheme already has defined conversion, then it is overwritten.

## CLI commands

You can manage cache using Kenning CLI commands.
The available commands are:

* `kenning cache list_files` - list cached files, their size and total cache size.
  To see full paths add `-v` argument.
* `kenning cache clear` - removes all cached files,
* `kenning cache settings` - prints default cache directory path and default max cache size.
