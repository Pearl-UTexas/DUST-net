---
layout: default
title: "Distributional Depth-Based Estimation of Object Articulation Models"
---

### Abstract
We propose a method that efficiently learns distributions over articulation model parameters directly from depth images without the need to know articulation model categories a priori. By contrast, existing methods that learn articulation models from raw observations typically only predict point estimates of the model parameters, which are insufficient to guarantee the safe manipulation of articulated objects. Our core contributions include a novel representation for distributions over rigid body transformations and articulation model parameters based on screw theory, von Mises-Fisher distributions, and Stiefel manifolds. Combining these concepts allows for an efficient, mathematically sound representation that implicitly satisfies the constraints that rigid body transformations and articulations must adhere to. Leveraging this representation, we introduce a novel deep learning based approach, DUST-net, that performs category-independent articulation model estimation while also providing model uncertainties. We evaluate our approach on several benchmarking datasets and real-world objects and compare its performance with two current state-of-the-art methods. Our results demonstrate that DUST-net can successfully learn distributions over articulation models for novel objects across articulation model categories, which generate point estimates with better accuracy than state-of-the-art methods and effectively capture the uncertainty over predicted model parameters due to noisy inputs.

### Cite
If you find this work useful in your own research, please consider citing:
```
@article{jain2021distributional,
  title={Distributional Depth-Based Estimation of Object Articulation Models},
  author={Jain, Ajinkya and Giguere, Stephen and Lioutikov, Rudolf and Niekum, Scott},
  journal={arXiv preprint arXiv:2108.05875},
  year={2021}
}
```
