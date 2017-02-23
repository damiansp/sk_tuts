import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse
from sklearn.linear_model import Lasso, Ridge

def _weights(x, dx = 1, orig = 0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y

def build_projection_operator(l_x, n_dir):
    '''
    Compute the tomography design matrix.

    @params
    -------
    l_x: int
        linear size of image array
    n_dir: int
        number of angles at which projections are acquired

    @returns
    --------
    p: sparse matrix of shape (n_dir l_x, l_x ** 2)
    '''
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint = False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices,
                                      data_unravel_indices))

    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx = 1, orig = X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])

    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator

def generate_synthetic_data():
    '''Synthetic binary data'''
    rs = np.random.RandomState(0)
    n_pts = 36.
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l / 2) ** 2 + (y - l / 2) ** 2 < (l / 2) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)

    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1

    mask = ndimage.gaussian_filter(mask, sigma = l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return res - ndimage.binary_erosion(res)

# Generate synthetic images and projections
l = 128
proj_operator = build_projection_operator(l, l / 7.)
data = generate_synthetic_data()
proj = proj_operator * data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# Reconstruction with L2 penalization (Ridge)
ridge = Ridge(alpha = 0.2)

ridge.fit(proj_operator, proj.ravel())

rec_l2 = ridge.coef_.reshape(l, l)

# Reconstruction with L1 penalization (Lasso)
lasso = Lasso(alpha = 0.001)

lasso.fit(proj_operator, proj.ravel())

rec_l1 = lasso.coef_.reshape(l, l)

plt.figure(figsize = (8, 3.3))
plt.subplot(131)
plt.imshow(data, interpolation = 'nearest', cmap = plt.cm.gray)
plt.axis('off')
plt.title('Original image')

plt.subplot(132)
plt.imshow(rec_l2, interpolation = 'nearest', cmap = plt.cm.gray)
plt.axis('off')
plt.title('Ridge')

plt.subplot(133)
plt.imshow(rec_l1, interpolation = 'nearest', cmap = plt.cm.gray)
plt.axis('off')
plt.title('Lasso')

plt.subplots_adjust(
    hspace = 0.01, wspace = 0.01, bottom = 0, left = 0, right = 1)
plt.show()


    
