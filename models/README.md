# Models files
Where are models files saved? Models files are typically big. It's preferable to save them to a custom folder like an external HDD of a shared disk. For this reason a docker environment variable `cache_dir` can specified at run:

```bash
./run.sh emotions models/
```

the `models` folder will be assigned to the `cache_dir` variable to be used as default alternative location to download pretrained models. A `os.getenv("cache_dir")` will be used to retrieve the environemnt variable in the code.