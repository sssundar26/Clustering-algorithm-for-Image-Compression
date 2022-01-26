#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os


def mykmeans(pixels, K):
    """
    Your goal of this assignment is implementing your own K-means.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    #random initiliasation of K centers
    pixels = pixels.reshape((-1,3))
    maxiter=20
    iteration=1
    classassign=[]
    newcenter=pixels[np.random.choice(pixels.shape[0],K)]
    oldcenter=np.zeros((K,pixels.shape[1]))
    distancemetric=np.zeros((len(pixels),K))
 
    
    while((iteration<=maxiter) & (np.linalg.norm(newcenter-oldcenter)!=0)):
        iteration=iteration+1
        oldcenter=np.array(newcenter)
        newcenter=[]
        distancefromcenter=np.zeros((len(pixels),K))
        clusterassign=[]
        assigndata=[]
    
        # assigning points to the centers
        for i in range(len(pixels)):
            for j in range(K):
                distancefromcenter[i,j]=np.linalg.norm(pixels[i,:]-oldcenter[j,:])
            clusterassign.append(np.argmin(distancefromcenter[i]))
    
       
        # computing new centers
        for i in range(K):
            assigndata=[]
            for j in range(len(pixels)):
                if(clusterassign[j]==i):
                    assigndata.append(pixels[j])
            newcenter.append(np.mean(assigndata,axis=0))
    
    centroid=np.array(newcenter)
    for i in range(len(pixels)):
            for j in range(len(centroid)):
                distancemetric[i,j]=np.linalg.norm(pixels[i,:]-centroid[j,:])
            classassign.append(np.argmin(distancemetric[i]))
    
    classassign=np.array(classassign).reshape(len(pixels),1)
    
    return classassign,centroid   
    
    raise NotImplementedError

def mykmedoids(pixels, K):
    """
    Your goal of this assignment is implementing your own K-medoids.
    Please refer to the instructions carefully, and we encourage you to
    consult with other resources about this algorithm on the web.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.
    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    maxIter=30  #maximum no of iterations
    iteration=1
    #randomising initialisation of cluster representatives
    pixels = pixels.reshape((-1,3))
    pixels = np.float32(pixels)
    choice = np.random.permutation(pixels.shape[0])[:K]
    old_medoid = np.copy(pixels[choice])
    #initialization of loss functions
    new_loss = [1]*K
    old_loss = [0]*K
    loss_function = []
    # function to calculate distance measure
    def calc_dist_medoid(pixels,medoid,l=2):
        rep_medoid = np.repeat(medoid,repeats=len(pixels),axis=0)
        data_matrix_dup = np.concatenate([pixels]*len(medoid))
        distance_from_mediod = np.linalg.norm(data_matrix_dup-rep_medoid,ord=l,axis=1).reshape(len(medoid),len(pixels))
        return distance_from_mediod.T
    #function to assign points to the nearest cluster centes
    def classassign(distance):
        return np.amin(distance,axis=1),np.argmin(distance,axis=1)
    #finding the J unique nearest points to the cluster centroid
    def SelectTopCandidates(cluster,j=30):
        centroid = np.mean(cluster,axis=0)
        distance_from_centroid = calc_dist_medoid(cluster,centroid[np.newaxis,:]).reshape((len(cluster),))
        unq_arr, indicesList = np.unique(distance_from_centroid,return_index=True)
        return_shape = min(len(indicesList),j)
        return cluster[indicesList[:return_shape]]
    #function for calculating new cluster representatives from the J candidatess
    def costcalculation(pixels,medoid):
        assign_out = classassign(calc_dist_medoid(pixels,medoid))
        class_assign = assign_out[1]
        distance_assign = assign_out[0]
        cluster_distance_old = []
        cluster_distance_new = []
        new_medoid = []
        for k in range(len(medoid)):
            cluster_k = class_assign[class_assign==k]
            if len(cluster_k)==1:   ###Edge Case Check
                cluster_distance_new.append(0)
                cluster_distance_old.append(0)
                new_medoid.append(np.array([medoid[k]]))
                continue
            old_distance_k = np.sum(distance_assign[class_assign==k])
            cluster_distance_old.append(old_distance_k)
            sub_cluster_k = pixels[class_assign==k]
            top_points_k = SelectTopCandidates(sub_cluster_k,1000)
            if len(top_points_k)>=1:
                distance =  calc_dist_medoid(sub_cluster_k,top_points_k)
            else:
                cluster_distance_new.append(0)
                cluster_distance_old.append(0)
                new_medoid.append(np.array([medoid[k]]))
                continue
                #print('Error in Cost Calculation Function')
            
            new_distance_k = np.sum(distance,axis=0)
            new_loss_k = np.amin(new_distance_k)
            if new_loss_k<old_distance_k:
                cluster_distance_new.append(new_loss_k)
                new_medoid_k = top_points_k[np.argmin(new_distance_k)]
            else:
                cluster_distance_new.append(old_distance_k)
                new_medoid_k = medoid[k]
            
            new_medoid.append(np.array([new_medoid_k]))

        return cluster_distance_old,cluster_distance_new,np.concatenate(new_medoid),class_assign
    # run until convergence while (convergence criteria)
    while iteration<=maxIter and np.linalg.norm(np.array(new_loss)-np.array(old_loss))>=(10**(-6)):
        old_loss,new_loss,new_medoid,class_assign = costcalculation(pixels,old_medoid)
        loss_function.append(new_loss)
        print('--Iteration--')
        print(iteration)
        old_medoid = np.copy(new_medoid)
        iteration+=1
        final_loss = sum(new_loss)

    # return the classassignments and centers
    return class_assign,old_medoid
    raise NotImplementedError

def main():
	if(len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]
	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	print(image_file_name, K)
	im = np.asarray(imageio.imread(image_file_name))

	fig, axs = plt.subplots(1, 2)

	classes, centers = mykmedoids(im, K)
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')

	classes, centers = mykmeans(im, K)
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')

	plt.show()

if __name__ == '__main__':
	main()

