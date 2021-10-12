"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    Ix = cv2.Sobel(image, dx = 1, dy = 0, scale = 1/8, ksize = 3, ddepth = -1)
    return np.array(Ix)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    Iy = cv2.Sobel(image, dx = 0, dy = 1, scale = 1/8, ksize = 3, ddepth = -1)
    return np.array(Iy)

def get_S_matrix(imagea, imageb, x, y):
    S = np.array([[imagea[x-1, y-1], imageb[x-1, y-1]],
                   [imagea[x-1, y], imageb[x-1, y]],
                   [imagea[x-1, y+1], imageb[x-1, y+1]],
                   [imagea[x, y-1], imageb[x, y-1]],
                   [imagea[x, y], imageb[x, y]],
                  [imagea[x, y+1], imageb[x, y+1]],
                  [imagea[x+1, y-1], imageb[x+1, y-1]],
                  [imagea[x+1, y], imageb[x+1, y]],
                 [imagea[x+1, y+1], imageb[x+1, y+1]]])
    return S

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """''
    # no need to normalize to uint8 here
    #img_a = cv2.normalize(img_a, 0, 255, cv2.NORM_MINMAX)
    #img_b = cv2.normalize(img_b, 0, 255, cv2.NORM_MINMAX)
    # initialize return displacement
    U = np.zeros((img_a.shape[0], img_a.shape[1]))
    V = np.zeros((img_b.shape[0], img_b.shape[1]))
    # obtain Ix and Iy
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    if k_type == 'uniform':
        kernel = np.ones((k_size, k_size), np.float32) / k_size**2
    else:
        kernel = cv2.getGaussianKernel(k_size, sigma)
    Ix_weighted = cv2.filter2D(src = Ix, kernel = kernel, ddepth = -1)
    Iy_weighted = cv2.filter2D(src = Iy, kernel = kernel, ddepth = -1)
    Ix_padded = np.pad(Ix_weighted, ((1,1), (1,1)), 'mean')
    Iy_padded = np.pad(Iy_weighted, ((1,1), (1,1)), 'mean')
    # obtain t
    t = np.pad(img_b, ((1,1),(1,1)), 'mean') - np.pad(img_a, ((1,1),(1,1)), 'mean')
    t = cv2.filter2D(src = t, kernel = kernel, ddepth = -1)
    # obtain 9x2 matrix S
    for u in range(img_a.shape[0]):
        for v in range(img_a.shape[1]):
            S = get_S_matrix(Ix_padded, Iy_padded, u, v)
            t_vec = -1 * np.array([t[u-1, v-1], t[u-1, v], t[u-1, v+1],
                              t[u, v-1], t[u, v], t[u, v+1],
                              t[u+1, v-1], t[u+1, v], t[u+1, v+1]])
            STS_inv = np.linalg.pinv(np.dot(np.transpose(S), S))
            #2x9 * 9x2 * 2x9 * 9x1=2x1
            displacement = np.dot(np.dot(STS_inv, np.transpose(S)), np.transpose(t_vec))
            U[u, v] = displacement[0]
            V[u, v] = displacement[1]
    return (U, V)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    # gaussian blur image first to avoid aliasing
    img_blurred = cv2.GaussianBlur(image, ksize = (5,5), sigmaX = 1, sigmaY=1)
    # subsample
    return np.array(img_blurred)[::2, ::2]


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    gaussian_pyramid_list = []
    gaussian_pyramid_list.append(image)
    for i in range(levels - 1):
        downsampled = reduce_image(image)
        gaussian_pyramid_list.append(downsampled)
        image = downsampled
    return gaussian_pyramid_list


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    return cv2.pyrUp(image, dstsize= (2*image.shape[1], 2*image.shape[0]))


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []
    for i in range(len(g_pyr)-1):
        previous_gaussian = g_pyr[i]
        next_gaussian = g_pyr[i + 1]
        upsampled_next_gaussian = expand_image(next_gaussian)
        laplacian_temp = previous_gaussian - upsampled_next_gaussian
        l_pyr.append(laplacian_temp)
    l_pyr.append(g_pyr[-1])
    return l_pyr

def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    # np.meshgrid: generate a 2D array of indices
    M, N = image.shape
    X, Y = np.meshgrid(range(N), range(M))
    map_x = X + U
    map_y = Y + V
    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), borderMode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_CUBIC)

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError
