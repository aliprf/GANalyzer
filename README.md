# GANalyzer

	
```diff
! plaese STAR the repo if you like it.
```


## GANalyzer: Analysis and Manipulation of GANs Latent Space for Controllable Face Synthesis


#### Link to the pre-print:
https://arxiv.org/abs/2302.00908

#### Link to the paper:

#### Link to the paper-with-codes:
https://paperswithcode.com/paper/ganalyzer-analysis-and-manipulation-of-gans

```
Please cite this work as:
@misc{https://doi.org/10.48550/arxiv.2302.00908,
  doi = {10.48550/ARXIV.2302.00908},
  url = {https://arxiv.org/abs/2302.00908},
  author = {Fard, Ali Pourramezan and Mahoor, Mohammad H. and Lamer, Sarah Ariel and Sweeny, Timothy},
  title = {GANalyzer: Analysis and Manipulation of GANs Latent Space for Controllable Face Synthesis},
  publisher = {arXiv},
  year = {2023},
}


```

##  Qualitative Results Evaluation:
The following are some examples from the paper, showing the performance of GANalyzer for both facial attribute editing, and intensity-based modification.

#### Facial Attribute Editing:

![Sample All](https://github.com/aliprf/GANalyzer/blob/master/img/all.png?raw=true)

#### Intensity-Based Attribute Editing:
![Intensity-based](https://github.com/aliprf/GANalyzer/blob/master/img/intensirt.png?raw=true)


## Creating Environment
```
$ conda env create -f environment.yml
$ conda activate ganalyzer
$ pip install -r requirements.txt

```


## Facial Attribute Editing
In the first step, we need to create the modification task using test.py->create_single_FAE_task() function. The following tasks are generated as default. You can modify each task according to explaination in the paper.
```
  fae_tasks = [{'tn': 'ANGRY', 'alpha': 2.5, 'beta': 1.0},
                 {'tn': 'BLACK', 'alpha': 2.0, 'beta': 1.0},
                 {'tn': 'FEMALE', 'alpha': 3.5, 'beta': 1.0},
                 {'tn': 'MALE', 'alpha': 3.0, 'beta': 1.0},
                 {'tn': 'OLD', 'alpha': 4.0, 'beta': 1.0},
                 {'tn': 'YOUNG', 'alpha': 4.0, 'beta': 1.0},
                 ]
```
After creating the tasks, you can pass each task to test.py->modify_noise() function and modify the arbitrary input latent vectors.

## Feature-based Synthesis
In the first step, we need to create the modification task using test.py->create_single_FB_task() function. The following tasks are generated as default. You can modify each task according to explaination in the paper.
```
   fb_tasks = [{'tn': 'ANGRY', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'BLACK', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'FEMALE', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'MALE', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'OLD', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'YOUNG', 'alpha': 1.0, 'beta': 0.25},
                ]
```
After creating the tasks, you can pass each task to test.py->modify_noise() function and modify the arbitrary input latent vectors.


## Image Generation
GANalyzer is a framework designed for latent space modification, and no face synthesis is part of it. We use StylGAN3 for image synthesis. Hence, the image generation function which is located in 'test.py->generate_images()' is provided by stylegan3 repo in the following address:
```
https://github.com/NVlabs/stylegan3
```

```diff
! IF YOU USE THIS FUNCTION, YOU NEED TO FOLLOW StyleGAN3 Licence:
https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt
```

### Reference:
```
Please cite this work as:
@misc{https://doi.org/10.48550/arxiv.2302.00908,
  doi = {10.48550/ARXIV.2302.00908},
  url = {https://arxiv.org/abs/2302.00908},
  author = {Fard, Ali Pourramezan and Mahoor, Mohammad H. and Lamer, Sarah Ariel and Sweeny, Timothy},
  title = {GANalyzer: Analysis and Manipulation of GANs Latent Space for Controllable Face Synthesis},
  publisher = {arXiv},
  year = {2023},
}


```


```diff
@@plaese STAR the repo if you like it.@@
```
