# LPDUtils
LP Deeplearning Utilities

## How to install
```
pipinstall -r src/lpdutils/requirements.txt
```

## How to use

```python
from lpdutils.lpimagedataset import LPImageDataSet
```

or if you are located at the same folder level

```python
import os, sys
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, '..'))
from lpdutils.lpimagedataset import LPImageDataSet
```
