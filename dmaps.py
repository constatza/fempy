# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""


import numpy as np

from scipy import spatial
import manilearn as ml
import maths



def nearest_neighbour_mapping(vectors, natural_coordinates, transformed_coordinates, k=3):
     coordinates_tree = spatial.cKDTree(natural_coordinates)
     
     distances, neighbours_id  = coordinates_tree.query(vectors, k=k)
     
     d_inv = 1/distances
     weights = d_inv/np.sum(d_inv, axis=1, keepdims=True)
     weights[np.isnan(weights)] = 1
     transformed_vectors = weights[:, :, None] * transformed_coordinates[neighbours_id, :]
     transformed_vectors = np.sum(transformed_vectors, axis=1)
     return transformed_vectors
     
     
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import smartplot
    from sklearn import manifold, datasets
    
    plt.close('all')
    
    epsilon = .7
    alpha = 1
    timesteps = 100
    numeigs = 2
    
    # set parameters
    length_phi = 10 #length of swiss roll in angular direction
    length_Z = 15 #length of swiss roll in z direction
    sigma = 0.1 #noise strength
    m = 1000 #number of samples
    # create dataset
    phi = length_phi*np.random.rand(m)
    
    X = length_Z*np.random.rand(m)
    Y = (phi )*np.sin(phi)
    Z = (phi )*np.cos(phi)
    swiss_roll = np.array([X, Y, Z])
    U = np.concatenate([[X],[Y],[Z]])
    U, Umean, Ustd = maths.znormalized(U, no_return=False)
    d = U.shape[0]
    N = U.shape[1]
    Phi = np.tile(phi, (numeigs, 1))
    color = phi#np.arange(swiss_roll.shape[1])
    X_r, err = manifold.locally_linear_embedding(swiss_roll.T, n_neighbors=20,
                                                 n_components=numeigs)
    Xr = X_r.T
#----------------------------------------------------------------------
# Plot result

    
    
    """Diffusion Maps"""   
    
    dmaps = ml.DiffusionMap(U, epsilon=epsilon, alpha=alpha)
    dmaps.fit(numeigs=numeigs, t=1) 
    
    linear_dmaps = maths.LinearMap(domain_data=dmaps.reduced_coordinates, 
                                    codomain_data=U)

    res_dm = linear_dmaps.residuals
   
    U_dm = linear_dmaps.direct_transform(dmaps.reduced_coordinates) 
    x_dm = U_dm[0,:]
    y_dm = U_dm[1,:]
    z_dm = U_dm[2,:]
   
    """PCA"""
    pca = ml.PCA(U)
    pca.fit(numeigs=numeigs)
    
    pca_map = maths.LinearMap(domain_data= pca.reduced_coordinates,
                              codomain_data=U)
    
    U_pca = pca_map.direct_transform(pca.reduced_coordinates)
    x_pca = U_pca[0, :]
    y_pca = U_pca[1, :]
    z_pca = U_pca[2, :]
    
    """Plots"""
    x = maths.znormalized(X)
    y = maths.znormalized(Y)
    z = maths.znormalized(Z)
    
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111, projection='3d')
    ax0.scatter(x, y, z, alpha=0.5, label='Original Data', c=phi, marker='.')
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(x, y, z, alpha=0.5, label='Original Data', color='b', marker='.')
    ax1.scatter(x_dm,y_dm,z_dm, color='r', label='Diffusion Maps', marker='.')  
    ax1.scatter(x_pca,y_pca,z_pca, color='g', label='PCA',  marker='.')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    e = np.logspace(-3, 3, num=20)
    plt.loglog(e, dmaps.kernel_sums_per_epsilon(U, e))
    
    ax3 = smartplot.plot_eigenvalues(dmaps.eigenvalues, marker='+', label='DMAPS')
    smartplot.plot_eigenvalues(pca.eigenvalues/np.max(pca.eigenvalues),ax=ax3, marker='o', label='PCA')
    ax3.legend()
    plt.grid()
    
    
     
    fig4, axes4 = plt.subplots(1, 3)
    fig4.suptitle('Normalized Eigenvectors')
    smartplot.plot_eigenvectors(dmaps.reduced_coordinates.T, ax=axes4[0], title='DMAPS', c=color, marker='.')
    smartplot.plot_eigenvectors(Xr.T, ax=axes4[1], title='LLE', c=color, marker='.')
    smartplot.plot_eigenvectors(pca.reduced_coordinates.T, ax=axes4[2], title='PCA', c=color, marker='.')
    
     
    fig5, axes5 = plt.subplots(1, 3)
    fig5.suptitle('Correlation')
    
    axes5[0].scatter(Phi, dmaps.reduced_coordinates, marker='.')
    axes5[1].scatter(Phi, Xr, marker='.')
    axes5[2].scatter(Phi, pca.reduced_coordinates, marker='.')
    
    correl_dfm = np.corrcoef(x=phi, y=dmaps.reduced_coordinates)
    correl_lle = np.corrcoef(x=phi, y=Xr)
    correl_pca = np.corrcoef(x=phi, y=pca.reduced_coordinates)
    
    fig6, axes6 = plt.subplots(1,1)
    axes6.spy(Phi.T @ dmaps.reduced_coordinates)
    plt.show()