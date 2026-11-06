# Datasets

FGDN is trained and evaluated on five deraining benchmarks. Place each dataset
under this folder using the structure below (matching the paths in
`options/*.yml`):

```
datasets/
├── Rain200H/          # train: 1,800 pairs / test: 200 pairs
│   ├── train/{input, target}
│   └── test/{input, target}
├── Rain200L/          # train: 1,800 pairs / test: 200 pairs
├── DID-Data/          # train: 12,000 pairs / test: 1,200 pairs
├── DDN-Data/          # train: 12,600 pairs / test: 1,400 pairs
└── SPA-Data/          # train: 638,492 pairs / test: 1,000 pairs (real rain)
```

`input` = rainy image, `target` = clean ground truth.

## Sources

| Dataset | Introduced by | Where to get it |
|---|---|---|
| Rain200L / Rain200H | Yang et al., *Deep Joint Rain Detection and Removal from a Single Image*, CVPR 2017 | [Project page](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) (RainTrainL / RainTrainH) |
| DID-Data | Zhang et al., *Density-aware Single Image De-raining using a Multi-stream Dense Network*, CVPRW 2018 | [DID-MDN repository](https://github.com/hezhangsprinter/DID-MDN) |
| DDN-Data | Fu et al., *Removing Rain from Single Images via a Deep Detail Network*, CVPR 2017 | [Project page](https://xueyangfu.github.io/project/rain/rain_removal.html) |
| SPA-Data | Wang et al., *A Model-Driven Deep Neural Network for Single Image Rain Removal*, CVPR 2019 | Follow the data instructions in the cited paper |

Note: if a project page moves, follow the dataset links referenced by recent
deraining papers (Restormer, DRSformer) which all use the same five benchmarks.
