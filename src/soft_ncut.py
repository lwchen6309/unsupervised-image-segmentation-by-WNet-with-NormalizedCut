import numpy as np
import tensorflow as tf
import TensorflowUtils as utils

def sparse_tensor_dense_tensordot(sp_a, b, axes, name=None):
    r"""Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
        a: `SparseTensor` of type `float32` or `float64`.
        b: `Tensor` with the same type as `a`.
        axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
         If axes is a scalar, sum over the last N axes of a and the first N axes
         of b in order.
         If axes is a list or `Tensor` the first and second row contain the set of
         unique integers specifying axes along which the contraction is computed,
         for `a` and `b`, respectively. The number of axes for `a` and `b` must
         be equal.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` with the same type as `a`.
    Raises:
        ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
        IndexError: If the values in axes exceed the rank of the corresponding
            tensor.
            
    authors: kojino
    source: https://github.com/tensorflow/tensorflow/issues/9210
    """

    def _tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            # print(sess.run(rank_a), sess.run(axes))
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.name_scope(name, "SparseTensorDenseTensordot", [sp_a, b, axes]) as name:
        # a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
        sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(sp_a, sp_a_axes)
        b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
                b, b_axes, True)
        ab_matmul = tf.sparse_tensor_dense_matmul(sp_a_reshape, b_reshape)
        if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(ab_matmul, sp_a_free_dims + b_free_dims, name=name)
        else:
            sp_a_free_dims = tf.convert_to_tensor(sp_a_free_dims, dtype=tf.int32)
            b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.int32)
            product = tf.reshape(
                    ab_matmul, tf.concat([sp_a_free_dims, b_free_dims], 0), name=name)
            if sp_a_free_dims_static is not None and b_free_dims_static is not None:
                product.set_shape(sp_a_free_dims_static + b_free_dims_static)
            return product    

def circular_neighbor(index_centor, r, image_shape):
    """
    Given center index of circle, 
    calculate the indeces of neighbor in the circle.
    
    In [1]: image_shape = (10,10)
    In [2]: x = np.zeros(image_shape)
    In [3]: i,j= circular_neighbor((3,3),5,image_shape)
    In [4]: x[i,j] = 1
    In [5]: x
    Out[5]:
    array([[1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    
    xc, yc = index_centor
    x = np.arange(0, 2*r+1)
    y = np.arange(0, 2*r+1)
    in_circle = ((x[np.newaxis,:]-r)**2 + (y[:,np.newaxis]-r)**2) < r**2
    in_cir_x, in_cir_y = np.nonzero(in_circle)
    in_cir_x += (xc-r)
    in_cir_y += (yc-r)
    x_in_array = (0 <= in_cir_x) * (in_cir_x < image_shape[0])
    y_in_array = (0 <= in_cir_y) * (in_cir_y < image_shape[1])
    in_array = x_in_array * y_in_array
    return in_cir_x[in_array], in_cir_y[in_array]
    
def gaussian_neighbor(image_shape, sigma_X = 4, r = 5):
    """
    Given shape of image, calculate neighbor of ravel index i 
    where L2(neighbor j, i) < r    
    and their gaussian likelihood: exp(-norm((Xi,Yi)-(Xj,Yj))**2/sigma_X**2)
    
    Args:
        sigma_X: sigma for metric of distance.
        r: radius of circle that only the neighbor in circle is considered.
    Returns:
        indeces: (rows, cols) [#nonzero, 2]
        vals: tensor [#nonzero]
        
        where rows, and cols are pixel in image,
        val is their likelihood in distance.
    
    """
    
    row_lst, col_lst, val_lst = [],[],[]
    for i, (a,b) in enumerate(np.ndindex(*image_shape)):
        neighbor_x, neighbor_y  = circular_neighbor((a,b), r, image_shape)
        neighbor_value = np.exp(-((neighbor_x-a)**2 + (neighbor_y-b)**2) / sigma_X**2)
        ravel_index = np.ravel_multi_index([neighbor_x, neighbor_y], image_shape)
        row_lst.append(np.array([i]*len(neighbor_x)))
        col_lst.append(ravel_index)
        val_lst.append(neighbor_value)
    rows = np.hstack(row_lst)
    cols = np.hstack(col_lst)
    indeces = np.vstack([rows, cols]).T.astype(np.int64)
    vals = np.hstack(val_lst).astype(np.float)
    return indeces, vals
    
    
def brightness_weight(image, neighbor_filter, sigma_I = 0.05):
    """
    Calculate likelihood of pixels in image by their metric in brightness.
    
    Args:
        image: tensor [B, H, W, C]
        neighbor_filter: is tensor list: [rows, cols, vals].
                        where rows, and cols are pixel in image,
                        val is their likelihood in distance.
        sigma_I: sigma for metric of intensity.
    returns:
        SparseTensor properties:\
            indeces: [N, ndims]
            bright_weight: [N, batch_size]
            dense_shape
    """
    
    indeces, vals, dense_shape = neighbor_filter
    rows = indeces[:,0]
    cols = indeces[:,1]
    image_shape = image.get_shape()
    weight_size = image_shape[1].value * image_shape[2].value
    
    hsv_image = tf.image.rgb_to_hsv(image / 255)
    bright_image = hsv_image[:,:,:,2]
    bright_image = tf.reshape(bright_image, shape=(-1, weight_size)) # [B, W*H]
    bright_image = tf.transpose(bright_image, [1,0]) # [W*H,B]
    
    Fi = tf.transpose(tf.nn.embedding_lookup(bright_image, rows),[1,0]) # [B, #elements]
    Fj = tf.transpose(tf.nn.embedding_lookup(bright_image, cols),[1,0]) # [B, #elements]
    bright_weight = tf.exp(-(Fi - Fj)**2 / sigma_I**2) * vals
    bright_weight = tf.transpose(bright_weight,[1,0]) # [#elements, B]
    
    return indeces, bright_weight, dense_shape

def hue_weight(image, neighbor_filter, sigma_I = 0.05):
    """
    Calculate likelihood of pixels in image by their metric in hue.
    
    Args:
        image: tensor [B, H, W, C]
        neighbor_filter: is tensor list: [rows, cols, vals].
                        where rows, and cols are pixel in image,
                        val is their likelihood in distance.
        sigma_I: sigma for metric of intensity.
    returns:
        SparseTensor properties:\
            indeces: [N, ndims]
            bright_weight: [N, batch_size]
            dense_shape
    """
    
    indeces, vals, dense_shape = neighbor_filter
    rows = indeces[:,0]
    cols = indeces[:,1]
    image_shape = image.get_shape()
    weight_size = image_shape[1].value * image_shape[2].value
    
    hsv_image = tf.image.rgb_to_hsv(image / 255)
    hue_image = hsv_image[:,:,:,0] # [B, H, W]
    hue_image = tf.reshape(hue_image, shape=(-1, weight_size)) # [B, W*H]
    hue_image = tf.transpose(hue_image, [1,0]) # [W*H,B]
    
    Fi = tf.transpose(tf.nn.embedding_lookup(hue_image, rows),[1,0]) # [B, #elements]
    Fj = tf.transpose(tf.nn.embedding_lookup(hue_image, cols),[1,0]) # [B, #elements]
    bright_weight = tf.exp(-(Fi - Fj)**2 / sigma_I**2) * vals
    bright_weight = tf.transpose(bright_weight,[1,0]) # [#elements, B]
    
    return indeces, bright_weight, dense_shape

def sat_weight(image, neighbor_filter, sigma_I = 0.05):
    """
    Calculate likelihood of pixels in image by their metric in saturation.
    
    Args:
        image: tensor [B, H, W, C]
        neighbor_filter: is tensor list: [rows, cols, vals].
                        where rows, and cols are pixel in image,
                        val is their likelihood in distance.
        sigma_I: sigma for metric of intensity.
    returns:
        SparseTensor properties:\
            indeces: [N, ndims]
            bright_weight: [N, batch_size]
            dense_shape
    """
    
    
    indeces, vals, dense_shape = neighbor_filter
    rows = indeces[:,0]
    cols = indeces[:,1]
    image_shape = image.get_shape()
    weight_size = image_shape[1].value * image_shape[2].value
    
    hsv_image = tf.image.rgb_to_hsv(image / 255)
    sat_image = hsv_image[:,:,:,1] # [B, H, W]
    sat_image = tf.reshape(sat_image, shape=(-1, weight_size)) # [B, W*H]
    sat_image = tf.transpose(sat_image, [1,0]) # [W*H,B]
    
    Fi = tf.transpose(tf.nn.embedding_lookup(sat_image, rows),[1,0]) # [B, #elements]
    Fj = tf.transpose(tf.nn.embedding_lookup(sat_image, cols),[1,0]) # [B, #elements]
    bright_weight = tf.exp(-(Fi - Fj)**2 / sigma_I**2) * vals
    bright_weight = tf.transpose(bright_weight,[1,0]) # [#elements, B]
    
    return indeces, bright_weight, dense_shape
    
    
def convert_to_batchTensor(indeces, batch_values, dense_shape):
    """
    Create a sparse tensor by:
    for vals in each batch:
        feed [row, cols, vals] into sparse tensor
    
    Args:
        indeces: indeces of value. (tensor [row, cols])
        batch_values: batches valus. (tensor [B, vals])
        dense_shape: dense shape for sparse tensor [H, W].
    Returns:
        batchTensor: sparse tensor [B, H, W]
    """
    
    batch_size = tf.cast(tf.shape(batch_values)[1],tf.int64)
    num_element = tf.cast(tf.shape(indeces)[0],tf.int64)
    
    # Expand indeces, values
    tile_indeces = tf.tile(indeces, tf.stack([batch_size,1]))
    tile_batch = tf.range(batch_size,dtype=tf.int64)
    tile_batch = tf.tile(tf.expand_dims(tile_batch,axis=1), tf.stack([1,num_element]))
    tile_batch = tf.reshape(tile_batch,[-1,1])
    
    # Expand dense_shape
    new_indeces = tf.concat([tile_batch,tile_indeces],axis=1)
    new_batch_values = tf.reshape(batch_values,[-1])
    new_dense_shape = tf.concat(
                        [tf.cast(tf.reshape(batch_size,[-1]),tf.int32),
                        tf.cast(dense_shape, tf.int32)], axis=0)
    new_dense_shape = tf.cast(new_dense_shape, tf.int64)
    # Construct 3D tensor [B, W*H, W*H]
    batchTensor = tf.SparseTensor(new_indeces,new_batch_values,new_dense_shape)
    return batchTensor
    
def sycronize_axes(tensor, axes, tensor_dims=None):
    """
    Synchronize first n dims of tensor
    
    Args:
        axes: a list of axes of tensor to be sycronized
        tensor_dims: dimension of tensor,
                    specified this if tensor has None shape if any.
    Returns:
        syn_tensor: sycronized tensor where axes is reduced.
    """
    
    # Swap axes to head dims
    if tensor_dims is None:
        tensor_dims = len(tensor.get_shape().as_list())
    perm_axes = list(axes)
    perm_axes.extend([i for i in range(tensor_dims) if i not in axes])
    perm_tensor = tf.transpose(tensor, perm_axes)
    
    # Expand 
    contract_axis_0_len = tf.shape(perm_tensor)[0]
    contract_axis_len = len(axes)
    diag_slice = tf.range(contract_axis_0_len)
    diag_slice = tf.expand_dims(diag_slice,axis=1)
    diag_slice = tf.tile(diag_slice, tf.stack([1,contract_axis_len]))
    
    # Slice diagonal elements
    syn_tensor = tf.gather_nd(perm_tensor, diag_slice)
    return syn_tensor
    
def soft_ncut(image, image_segment, image_weights):
    """
    Args:
        image: [B, H, W, C]
        image_segment: [B, H, W, K]
        image_weights: [B, H*W, H*W]
    Returns:
        Soft_Ncut: scalar
    """
    
    batch_size = tf.shape(image)[0]
    num_class = tf.shape(image_segment)[-1]
    image_shape = image.get_shape()
    weight_size = image_shape[1].value * image_shape[2].value
    image_segment = tf.transpose(image_segment, [0, 3, 1, 2]) # [B, K, H, W]
    image_segment = tf.reshape(image_segment, tf.stack([batch_size, num_class, weight_size])) # [B, K, H*W]
    
    # Dis-association
    # [B0, H*W, H*W] @ [B1, K1, H*W] contract on [[2],[2]] = [B0, H*W, B1, K1]
    W_Ak = sparse_tensor_dense_tensordot(image_weights, image_segment, axes=[[2],[2]])
    W_Ak = tf.transpose(W_Ak, [0,2,3,1]) # [B0, B1, K1, H*W]
    W_Ak = sycronize_axes(W_Ak, [0,1], tensor_dims=4) # [B0=B1, K1, H*W]
    # [B1, K1, H*W] @ [B2, K2, H*W] contract on [[2],[2]] = [B1, K1, B2, K2]
    dis_assoc = tf.tensordot(W_Ak, image_segment, axes=[[2],[2]])
    dis_assoc = sycronize_axes(dis_assoc, [0,2], tensor_dims=4) # [B1=B2, K1, K2]
    dis_assoc = sycronize_axes(dis_assoc, [1,2], tensor_dims=3) # [K1=K2, B1=B2]
    dis_assoc = tf.transpose(dis_assoc, [1,0]) # [B1=B2, K1=K2]
    dis_assoc = tf.identity(dis_assoc, name="dis_assoc")
    
    # Association
    # image_segment: [B0, K0, H*W]
    sum_W = tf.sparse_reduce_sum(image_weights,axis=2) # [B1, W*H]
    assoc = tf.tensordot(image_segment, sum_W, axes=[2,1]) # [B0, K0, B1]
    assoc = sycronize_axes(assoc, [0,2], tensor_dims=3) # [B0=B1, K0]
    assoc = tf.identity(assoc, name="assoc")
    
    utils.add_activation_summary(dis_assoc)
    utils.add_activation_summary(assoc)
    
    # Soft NCut
    eps = 1e-6
    soft_ncut = tf.cast(num_class, tf.float32) - \
                tf.reduce_sum((dis_assoc + eps) / (assoc + eps), axis=1)
    
    return soft_ncut
    
def dense_brightness_weight(image, sigma_X = 4, sigma_I = 10, r = 5):
    """
    Calculate bright_weight(connection) for image
    
    Args:
        image: ndarray [B,H,W,C].
        sigma_X: sigma for metric of distance.
        sigma_I: sigma for metric of intensity.
        r: radius of circle that only the neighbor in circle is considered.
    Returns:
        brightness_weight: ndarray [B, W*H, W*H]
    """
    
    weight_size = np.prod(image.shape[1:3])
    batch_size = image.shape[0]
    bright_weights = np.zeros((batch_size,weight_size,weight_size))
    reduce_image = np.mean(image,axis=3)
    
    for batch in range(batch_size):
        # Reduce channel
        flat_image = np.ravel(reduce_image[batch])
        
        # Gaussian neighbor
        Fj, Fi = np.meshgrid(flat_image, flat_image)
        X, Y = list(zip(*np.ndindex(image.shape[1:3])))
        Xj, Xi = np.meshgrid(X,X)
        Yj, Yi = np.meshgrid(Y,Y)
        X_metric = np.sqrt((Xi - Xj)**2 + (Yi - Yj)**2)
        F_metric = np.abs(Fi - Fj)
        
        # Brightness weight
        bright_weight = np.exp(-(X_metric**2 / sigma_X**2) -(F_metric**2 / sigma_I**2))
        bright_weight[X_metric >= r] = 0
        bright_weights[batch] = bright_weight
        
    return bright_weights
    
def dense_soft_ncut(image, image_segment):
    """
    Soft normalized cut for dense image and image_segment.
    
    Args:
        image: ndarray [B, H,W, C]
        image_segment: ndarray [B, H, W, K]
    Returns:
        soft_ncut: scalar
    """
    
    batch_num = image.shape[0]
    num_class = image_segment.shape[-1]
    weight_size = image.shape[1] * image.shape[2]
    image_segment = np.transpose(image_segment, [0, 3, 1, 2]) # [B, K, H, W]
    image_segment = np.reshape(image_segment, [batch_num, num_class, weight_size]) # [B, K, H*W]

    image_weights = dense_brightness_weight(image)
    sum_image_weights = np.sum(image_weights,axis=-1)
    
    dis_assoc = np.zeros((batch_num,num_class))
    assoc = np.zeros((batch_num,num_class))
    for batch in range(batch_num):
        # [K, H*W] @ [H*W, H*W] = [K, H*W]
        W_Ak = np.matmul(image_segment[batch], image_weights[batch]) # [K, H*W]
        dissoc = np.matmul(W_Ak, image_segment[batch].T) # [K, K]
        dis_assoc[batch] = np.diag(dissoc) # [K]
        assoc[batch] = np.matmul(image_segment[batch], sum_image_weights[batch]) # [K]
    eps = 0
    soft_ncut = num_class - np.sum((dis_assoc + eps) / (assoc + eps), axis=1)
    return soft_ncut
    
    
import unittest
if __name__ == '__main__': 
    class TestBrightWeight(unittest.TestCase):
        
        # Global setting
        NUM_OF_CLASSESS = 4
        IMAGE_SIZE = 10

        # Tf placeholder
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
        kernels = tf.cast(utils.weight_variable([3,3,3,NUM_OF_CLASSESS], name="weight"),tf.float32)
        bias = tf.cast(utils.bias_variable([NUM_OF_CLASSESS], name="bias"),tf.float32)
        image_segment = utils.conv2d_basic(image, kernels, bias)
        
        neighbor_indeces = tf.placeholder(tf.int64, name="neighbor_indeces")
        neighbor_vals = tf.placeholder(tf.float32, name="neighbor_vals")
        neighbor_shape = tf.placeholder(tf.int64, name="neighbor_shape")
        neighbor_filter = (neighbor_indeces, neighbor_vals, neighbor_shape)
        _image_weights = brightness_weight(image, neighbor_filter, sigma_I = 10)
        image_weights = convert_to_batchTensor(*_image_weights)
        dense_image_weights = tf.sparse_to_dense(
            image_weights.indices, 
            image_weights.dense_shape,
            image_weights.values)    

        soft_ncuts = soft_ncut(image, image_segment, image_weights)
        loss = tf.reduce_sum(soft_ncuts)
        
        # Optimizer
        trainable_var = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(loss, var_list=trainable_var)
        trainer = optimizer.apply_gradients(grads)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # Data
        x = np.arange(IMAGE_SIZE*IMAGE_SIZE).reshape(IMAGE_SIZE,IMAGE_SIZE)
        x = np.moveaxis(np.tile(x, [3, 1, 1]), [0, 1, 2], [2, 0, 1])
        x = x[np.newaxis,:,:,:]
        
        def test_image_weight(self):
            """
            """
            image_shape = self.image.get_shape().as_list()[1:3]
            gauss_indeces, gauss_vals = gaussian_neighbor(image_shape, sigma_X = 4, r = 5)
            weight_shapes = np.prod(image_shape)
            sparse_bright_weight = self.sess.run(self.dense_image_weights, 
                    feed_dict={
                        self.image: self.x,
                        self.neighbor_indeces: gauss_indeces,
                        self.neighbor_vals: gauss_vals,
                        self.neighbor_shape: [weight_shapes, weight_shapes]
                    })
            
            # Compare with dense version
            dense_bright_weight = dense_brightness_weight(self.x)
            max_err = np.max(np.abs(sparse_bright_weight - dense_bright_weight))
            print('max error of image_weights = %.4e'%max_err)
            np.testing.assert_allclose(sparse_bright_weight, dense_bright_weight, rtol=1e-6, atol=1e-6)

        def test_soft_ncut(self):
            """
            """
            image_shape = self.image.get_shape().as_list()[1:3]
            gauss_indeces, gauss_vals = gaussian_neighbor(image_shape, sigma_X = 4, r = 5)
            weight_shapes = np.prod(image_shape)
            sp_soft_ncut, image_segment = self.sess.run(
                    [self.soft_ncuts, self.image_segment], 
                    feed_dict={
                        self.image: self.x,
                        self.neighbor_indeces: gauss_indeces,
                        self.neighbor_vals: gauss_vals,
                        self.neighbor_shape: [weight_shapes, weight_shapes]
                    })
            
            # Compare with dense version
            dn_soft_ncut = dense_soft_ncut(self.x, image_segment)
            max_err = np.max(np.abs(sp_soft_ncut - dn_soft_ncut))
            print('max error of %s = %.4e'%('soft_ncut', max_err))
            np.testing.assert_allclose(sp_soft_ncut, dn_soft_ncut, rtol=1e-4, atol=1e-6)
            
        def test_train(self):
            image_shape = self.image.get_shape().as_list()[1:3]
            gauss_indeces, gauss_vals = gaussian_neighbor(image_shape, sigma_X = 4, r = 5)
            weight_shapes = np.prod(image_shape)
            result, _ = self.sess.run([self.soft_ncuts, self.trainer], 
                    feed_dict={
                        self.image: self.x,
                        self.neighbor_indeces: gauss_indeces,
                        self.neighbor_vals: gauss_vals,
                        self.neighbor_shape: [weight_shapes, weight_shapes]
                    })
    
    unittest.main()