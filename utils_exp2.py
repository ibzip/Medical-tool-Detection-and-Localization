##############################################################################
# Contains methods to generate gaussian and uniform distribution image maps.##
# These maps are then used as labels in cross_entropy for the localization  ##
# task. ######################################################################
import numpy as np

def get_gauss(p_maps, image_size, pmaps_offset, label_offset, label):
    sigma2=10
    rescale_factor_1 = float(image_size[1])/720.0
    rescale_factor_2 = float(image_size[0])/576.0
    sigma2_rescaled = sigma2*rescale_factor_1*rescale_factor_2
    i = 0
    for joint in range(label_offset,label_offset + 9,2):
        location = [label[joint]*rescale_factor_1,label[joint+1]*rescale_factor_2]

        if location[0] < 0 and location[1] < 0: # If this joint is not visible in the image
            print("One tool with a None joint")
            p_maps[pmaps_offset + i,:,:]  = 1.0/(image_size[0]*image_size[1]-0)*np.ones((image_size[0],image_size[1]))
            i = i + 1
            continue
        #print("location:",location)
        p_maps[pmaps_offset + i,:,:] = gaussian2d_map(image_size, location, sigma2_rescaled)
        i = i + 1

    return p_maps

def p_map_generator(label,image_size = (576,720),sigma2=10):
    """
    Input : 
    label : FUll labels array containing labels of all the samples
    sigma2 : variance of gaussian

    Output: 
    p_maps : if no tool is present then from unifrom distribution otherwise from gaussian distribution
    of size (576, 720, no_joints, no_tools)
    """

    p_maps = np.zeros((10,image_size[0],image_size[1]))
    
    rescale_factor_1 = float(image_size[1])/720.0
    rescale_factor_2 = float(image_size[0])/576.0

    sigma2_rescaled = sigma2*rescale_factor_1*rescale_factor_2

    if len(label) == 0: # This image sample containg no labels.
        #p_maps = 1.0/(576*720-0)*np.ones(image_size)
        print("do something later on")
        
    else:
        tool1 = False
        tool2 = False
        tool3 = False
        tool4 = False
        for l in label:
            if l == 1111:
                tool1 = True
            elif l == 1110:
                tool2 = True
            elif l == 1100:
                tool3 = True
            elif l == 1000:
                tool4 = True

        # Generate maps for tool1
        pmaps_offset = 0
        label_offset = 1
        if tool1:
            # Generate gaussian maps for the joints of present tool
            p_maps = get_gauss(p_maps, image_size, pmaps_offset, label_offset, label)
            label_offset += 11
        else:
            # Generate uniform distribution maps for the joints of absent tool
            p_maps[pmaps_offset:pmaps_offset + 5,:,:] = 1.0/(image_size[0]*image_size[1]-0)*np.ones((5,image_size[0],image_size[1]))

        #Generate maps for tool2
        pmaps_offset = 5
        if tool2:
            # Generate gaussian maps for the joints of present tool
            p_maps = get_gauss(p_maps, image_size, pmaps_offset, label_offset, label)
            label_offset += 11
        else:
            # Generate uniform distribution maps for the joints of absent tool
            p_maps[pmaps_offset:pmaps_offset + 5,:,:] = 1.0/(image_size[0]*image_size[1]-0)*np.ones((5,image_size[0],image_size[1]))
        """ 
        #Generate maps for tool3
        pmaps_offset = 10
        if tool3:
            # Generate gaussian maps for the joints of present tool
            p_maps = get_gauss(p_maps, image_size, pmaps_offset, label_offset, label)
            label_offset += 11
        else:
            # Generate uniform distribution maps for the joints of absent tool
            p_maps[pmaps_offset:pmaps_offset + 5,:,:] = 1.0/(image_size[0]*image_size[1]-0)*np.ones((5,image_size[0],image_size[1]))

        #Generate maps for tool4
        pmaps_offset = 15
        if tool4: 
            # Generate gaussian maps for the joints of present tool
            p_maps = get_gauss(p_maps, image_size, pmaps_offset, label_offset, label)
            label_offset += 11
        else:
            # Generate uniform distribution maps for the joints of absent tool
            p_maps[pmaps_offset:pmaps_offset + 5,:,:] = 1.0/(image_size[0]*image_size[1]-0)*np.ones((5,image_size[0],image_size[1]))
        """

    return p_maps

def gaussian2d_map(size, location,sigma2):
    x = np.array(range(size[1])) # Real world X coordinates are actually 720
    y = np.array(range(size[0])) # Y are 576
    X, Y = np.meshgrid(x, y)
    # Mean vector and covariance matrix
    mu = np.array(location)
    Sigma = sigma2*np.eye(2)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    p_map = multivariate_gaussian(pos, mu, Sigma)

    return p_map 

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / float(N)

