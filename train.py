#!/usr/bin/env python3
"""Train a jimm model. Thin timm-style wrapper around ``jimm.train``.

Example:
    python train.py --model resnet50 --data-dir /path/to/dataset -b 128 --epochs 90
"""

from jimm.train import main

if __name__ == "__main__":
    main()
