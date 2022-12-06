MT ablation-related projects

To run demo pipeline:

1.  Install anaconda

```bash
make anaconda.sh
```

2. Create sockeye environment.
```bash
make conda_environment
conda activate sockeye
make sockeye
make sockeye_optional
```

3. Install apex

```bash
make apex_install
```

Now, you can run the MT demo pipelines from Sockeye web site.

Sequence copy demo:
```bash
make genseqcopy.py
```

(Longer) WMT2014  Demo:
```bash
make wmt_demo
```



