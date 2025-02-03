# Kenning environment variables

This section contains information about Kenning environment variables and how they influence the behavior of the program.

## `KENNING_CACHE_DIR`

`KENNING_CACHE_DIR` defines directory for resources used by Kenning, like downloaded datasets.
If it is not defined, `$HOME/.kenning` will be used.

## `KENNING_MAX_CACHE_SIZE`

`KENNING_MAX_CACHE_SIZE` specifies the maximum amount of space used by [cache](kenning-cache-dir).
The default value is 50GB.

## `KENNING_USE_DEFAULT_EXCEPTHOOK`

If `KENNING_USE_DEFAULT_EXCEPTHOOK` environmental variable is defined, `sys.excepthook` will not be overridden.

This function is overridden in order to enable mechanism for deducing which optional dependencies are missing.
For instance, if `gluoncv` is not available, Kenning will suggest installing `kenning[mxnet]` which contains missing package with defined version restriction and other requirements.

Kenning CLI does not require custom `sys.excepthook`, so it will not be affected by `USE_DEFAULT_EXCEPTHOOK` variable.

## `KENNING_DOCS_VENV`

`KENNING_DOCS_VENV` defines path to the virtual environment used for tests marked as `docs`.
If is not defined, system's default environment will be used.

## `KENNING_ENABLE_ALL_LOGS`

If `KENNING_ENABLE_ALL_LOGS` environmental variable is defined, logs from other libraries will be enabled.
Passes verbosity of Kenning logs to other libraries.

This is useful for debugging purposes, but it may cause a lot of noise in the logs.

## `KENNING_DISABLE_IO_VALIDATION`

By default Kenning, before and after each inference, validates received data with defined specification.
Similar check is done before evaluation.

These validations can be disabled defining `KENNING_DISABLE_IO_VALIDATION` variable.

