## Wheels? What's that?
I'm using install from local wheels if avaiable. This will speed up build and tests, avoding to transfer several times data over the internet:

```bash
Collecting torch==1.5.0
  Downloading https://files.pythonhosted.org/packages/76/58/668ffb25215b3f8231a550a227be7f905f514859c70a65ca59d28f9b7f60/torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl (752.0MB)
```
  
I download once the big wheels for `pytorch` (752 MB) and `tensorflow` ((516.2 MB) in the `wheels` folder and check for them before building:

```bash
└── wheels
    ├── tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl
    └── torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl
```

Check the downloadable wheels from pypi here:

- tensorflow, https://pypi.org/project/tensorflow/#files
- pytorch, https://pypi.org/project/torch/#files 