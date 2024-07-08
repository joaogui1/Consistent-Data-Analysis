# Data Analysis

Code to generate the images for the paper [On the consistency of hyper-parameter selection in value-based deep reinforcement learning](https://arxiv.org/abs/2406.17523) by Johan Obando-Ceron, João G.M. Araújo, Aaron Courville, Pablo Samuel Castro.

Check out the accompanying website in [https://consistent-hparams.streamlit.app](https://consistent-hparams.streamlit.app/)!

To run the website locally use [this repo](https://github.com/joaogui1/Consistent-Website/tree/main)

+ To generate the plots comparing data regimes, run `split_graph.py`
+ To compute the Game Comparisons, Hyperparameter Comparisons, and THC metrics, run `precompute.py`

Please cite our work if you find it useful in your research:
```latex
@inproceedings{
      ceron2024consistency,
      title={On the consistency of hyper-parameter selection in value-based deep reinforcement learning},
      author={Johan Obando-Ceron and João G. M. Araújo and Aaron Courville and Pablo Samuel Castro},
      booktitle={Reinforcement Learning Conference},
      year={2024},
      url={https://openreview.net/forum?id=szUyvvwoZB}
}
```
