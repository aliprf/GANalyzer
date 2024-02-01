import numpy as np
from numpy import load
import os
from tqdm import tqdm
import math
import torch
import pickle
from skimage.io import imread, imsave


def create_single_FB_task():  # Feature-based modification
    fb_tasks = [{'tn': 'ANGRY', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'BLACK', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'FEMALE', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'MALE', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'OLD', 'alpha': 1.0, 'beta': 0.25},
                {'tn': 'YOUNG', 'alpha': 1.0, 'beta': 0.25},
                ]
    return fb_tasks


def create_single_FAE_task():  # Facial attribute editing
    fae_tasks = [{'tn': 'ANGRY', 'alpha': 2.5, 'beta': 1.0},
                 {'tn': 'BLACK', 'alpha': 2.0, 'beta': 1.0},
                 {'tn': 'FEMALE', 'alpha': 3.5, 'beta': 1.0},
                 {'tn': 'MALE', 'alpha': 3.0, 'beta': 1.0},
                 {'tn': 'OLD', 'alpha': 4.0, 'beta': 1.0},
                 {'tn': 'YOUNG', 'alpha': 4.0, 'beta': 1.0},
                 ]
    return fae_tasks


def calculate_b_vector(sample, eigenvalues, eigenvectors, meanvector):
    tmp1 = sample - meanvector
    b_vector = np.dot(eigenvectors.T, tmp1)

    i = 0
    for b_item in b_vector:
        lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

        if b_item > 0:
            b_item = min(b_item, lambda_i_sqr)
        else:
            b_item = max(b_item, -1 * lambda_i_sqr)
        b_vector[i] = b_item
        i += 1

    return b_vector


def modify_noise(task, num=10, latent_vectors=None):
    id_vectors = []
    fb_vectors = []
    orig_vectors = []

    task_name = task['tn']
    alpha = task['alpha']
    beta = task['beta']

    eigenvalues = load('PCA_DATA/_' + task_name + "_eigenvalues.npy")
    eigenvectors = load('PCA_DATA/_' + task_name + "_eigenvectors.npy")
    meanvector = load('PCA_DATA/_' + task_name + "_meanvector.npy")

    if latent_vectors is None:
        latent_vectors = []
        for i in range(num):
            latent_vectors.append(np.round(np.random.RandomState(i).randn(512), decimals=3))

    for latent_vector in tqdm(latent_vectors):
        k_seg = int(beta * len(eigenvalues))

        eigenvectors_sem = eigenvectors[:, :k_seg]
        eigenvalues_sem = eigenvalues[:k_seg]

        b_vector_p_sem = calculate_b_vector(latent_vector, eigenvalues_sem, eigenvectors_sem, meanvector)
        b_vector_p_id = calculate_b_vector(latent_vector, eigenvalues, eigenvectors, meanvector)

        vec_sem = np.expand_dims((1.0 * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem)), 0)
        vec_id = np.expand_dims((alpha * meanvector + np.dot(eigenvectors, b_vector_p_id)), 0)
        id_vectors.append(vec_id)
        fb_vectors.append(vec_sem)
        orig_vectors.append(np.expand_dims(latent_vector,0))
    return orig_vectors, id_vectors, fb_vectors


def generate_images(vectors, task, save_path):
    """This code is provided by   https://github.com/NVlabs/stylegan3
        IF YOU ARE USING THIS FUNCTION, YOU NEED TO FOLLOW StyleGAN3 Licence:
        https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt
    """
    with open('./ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    for i, vec in enumerate(vectors):
        z = torch.from_numpy(vec).cuda()  # latent codes
        img = G(z, None, truncation_psi=0.55, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        npy_img = img[0].cpu().numpy()
        imsave(fname=f'result_img/{save_path}/{task}_{str(i)}.jpg', arr=npy_img)
    pass


if __name__ == '__main__':
    """ This is a sample for testing the performance of the GANalyzer."""
    """ The result of all the methods are noise vectors, and you need to use any of the StyleGAN Families to 
        generate the corresponding images:
        https://github.com/NVlabs/stylegan
        https://github.com/NVlabs/stylegan2
        https://github.com/NVlabs/stylegan3
        """

    '''feature-based editing'''
    # create setting for feature-based synthesis
    fb_tasks = create_single_FB_task()
    for task in fb_tasks:
        print('Feature-Based Synthesis For => ' + task['tn'])
        latent_vectors, _, fb_vectors = modify_noise(task, num=10)
        # use StyleGAN Family and synthesize images using fb_vectors
        display(latent_vectors,  'fb_'+task['tn']+'_orig_', 'fb')
        display(fb_vectors,  'fb_'+task['tn']+'_new_', 'fb')


    '''facial attribute editing'''
    # create setting for facial attribute editing.
    fae_tasks = create_single_FAE_task()
    for task in fae_tasks:
        print('Facial Attribute Editing For => ' + task['tn'])
        latent_vectors, id_vectors, _ = modify_noise(task, num=10)
        # use StyleGAN Family and synthesize images using  latent_vectors, id_vectors
        display(latent_vectors, 'fae_' + task['tn'] + '_orig_', 'fae')
        display(id_vectors, 'fae_' + task['tn'] + '_new_', 'fae')
