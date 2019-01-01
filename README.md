# Acapella Tuning

"Smart Acapella Autotuning without Specifying Pitch Correction"




**seqGAN:** A PyTorch implementation of "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient." (Yu, Lantao, et al.). The code is highly simplified, commented and (hopefully) straightforward to understand. The policy gradients implemented are also much simpler than in the original work (https://github.com/LantaoYu/SeqGAN/) and do not involve rollouts- a single reward is used for the entire sentence (inspired by the examples in http://karpathy.github.io/2016/05/31/rl/).

The architectures used are different than those in the orignal work(https://github.com/suragnair/seqGAN). Specifically, a recurrent bidirectional GRU network is used as the discriminator.

To run the code:
```bash 
python main.py
```