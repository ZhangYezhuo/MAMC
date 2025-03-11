# MAMC
A method for Automatic Modulation Classification using [Mamba](https://github.com/state-spaces/mamba) structure

A project employing the **Selective State Space Model (Mamba)** method for **Automatic Modulation Classification (AMC)** in a scenario of **extended signal length**.

The increased sequence length complicates the learning process and diminishes accuracy, while simultaneously escalating memories and reduces timeliness. This issue brings the following impacts:
![Length effects](lengths_effects.jpg)


if our codes helped your reasearch, please consider citing the corresponding submission

## arXiv
> @article{zhang2024MAMC,<br>
> &emsp;&emsp;title={MAMCA -- Optimal on Accuracy and Efficiency for Automatic Modulation Classification with Extended Signal Length},<br>
> &emsp;&emsp;author={Yezhuo Zhang and Zinan Zhou and Yichao Cao and Guangyu Li and Xuanpeng Li},<br>
> &emsp;&emsp;year={2024},<br>
> &emsp;&emsp;journal={arXiv preprint arXiv:2405.11263},<br>
> }
## IEEE Communications Letters (Early Access)
[MAMC](https://ieeexplore.ieee.org/document/10705364)
> @article{10705364,<br>
> &emsp;&emsp;author={Zhang, Yezhuo and Zhou, Zinan and Cao, Yichao and Li, Guangyu and Li, Xuanpeng},<br>
> &emsp;&emsp;journal={IEEE Communications Letters}, <br>
> &emsp;&emsp;title={MAMC - Optimal on Accuracy and Efficiency for Automatic Modulation Classification with Extended Signal Length}, <br>
> &emsp;&emsp;year={2024},<br>
> &emsp;&emsp;volume={},<br>
> &emsp;&emsp;number={},<br>
> &emsp;&emsp;pages={1-1},<br>
> &emsp;&emsp;doi={10.1109/LCOMM.2024.3474519}<br>
> }

We utilize a denosing unit for better accuracy performance under **noise interference**, while using Mamba as the backbone for **low GPU occupancy** and **training/inference time**.
- MAMC structure
![MAMC_structure](framework.jpg)

To related AMC works, as well as sorce code: 
- Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges

    [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark?tab=readme-ov-file)

To the denosing method employed in our work, as well as source code: 
- IEEE Transactions on Industrial Informatics 2020

    [Deep Residual Shrinkage Networks for Fault Diagnosis](https://ieeexplore.ieee.org/abstract/document/8850096)

- Deep-Residual-Shrinkage-Networks-for-intelligent-fault-diagnosis-DRSN

    [pytorch-Deep-Residual-Shrinkage-Networks (DRSN)](https://github.com/liguge/Deep-Residual-Shrinkage-Networks-for-intelligent-fault-diagnosis-DRSN-)


To the Mamba method employed in our work, as well as source code: 
- arXiv preprint arXiv:2312.00752

    [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

- mamba

    [mamba](https://github.com/state-spaces/mamba)

## Requirements

```
pip install -r requirements.txt
```
please refer tllib==0.4.0 to [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)

## Training
cd into ```code/script``` and do
    
```python
bash RML2016.10a.sh
```

## Contact

If you have any problem with our code or any suggestions, including discussion on SEI, please feel free to contact

- Yezhuo Zhang (zhang_yezhuo@seu.edu.cn | zhang_yezhuo@outlook.com)
- Zinan Zhou (zhouzinan919@seu.edu.cn)
- Xuanpeng Li (li_xuanpeng@seu.edu.cn)

