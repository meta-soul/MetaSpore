# metasporecli: Command-line Interface for MetaSpore Flow Management

metasporecli implements the ``metaspore`` command, which can be used to manage MetaSpore flows.

With metasporecli installed and the configuration file ``metaspore-flow.yml`` in the current
directory, we can use the follwing command to start a MetaSpore flow:

```bash
metaspore flow up
```

and the following command to stop a MetaSpore flow:

```bash
metaspore flow down
```

Use the following commands to see help of the ``metaspore`` command:

```bash
metaspore -h
metaspore flow -h
```

## Install

metasporecli can be installed using pip. We recommend the user to create a virtualenv to install metasporecli.

```bash
python3 -m pip install metasporecli
```

The latest release of metasporecli can be found on PyPI [metasporecli: Command-line Interface for MetaSpore Flow Management](https://pypi.org/project/metasporecli/).

## Build

```bash
python3 -m pip wheel .
```

metasporecli depends on [metasporeflow](https://github.com/meta-soul/MetaSpore/tree/main/python/metasporeflow), to build metasporecli with locally built metasporeflow, use the following command:

```bash
python3 -m pip wheel . ../metasporeflow/metasporeflow-*-py3-none-any.whl
```
