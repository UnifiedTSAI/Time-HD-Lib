import numpy as np

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    x = np.array(x)
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def run_augmentation_single(x, y, args):
    """
    Simplified augmentation function that applies basic transformations without DTW dependencies
    """
    np.random.seed(getattr(args, 'seed', 42))

    x_aug = x
    y_aug = y

    if len(x.shape)<3:
        # Augmenting on the entire series: using the input data as "One Big Batch"
        #   Before  -   (sequence_length, num_channels)
        #   After   -   (1, sequence_length, num_channels)
        # Note: the 'sequence_length' here is actually the length of the entire series
        x_input = x[np.newaxis,:]
    elif len(x.shape)==3:
        # Augmenting on the batch series: keep current dimension (batch_size, sequence_length, num_channels)
        x_input = x
    else:
        raise ValueError("Input must be (batch_size, sequence_length, num_channels) dimensional")

    augmentation_ratio = getattr(args, 'augmentation_ratio', 0)
    if augmentation_ratio > 0:
        augmentation_tags = "%d"%augmentation_ratio
        for n in range(augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
        if getattr(args, 'extra_tag', None):
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = getattr(args, 'extra_tag', "")

    if(len(x.shape)<3):
        # Reverse to two-dimensional in whole series augmentation scenario
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags


def augment(x, y, args):
    """
    Apply basic augmentation techniques based on args configuration
    """
    augmentation_tags = ""
    
    if getattr(args, 'jitter', False):
        x = jitter(x)
        augmentation_tags += "_jitter"
    if getattr(args, 'scaling', False):
        x = scaling(x)
        augmentation_tags += "_scaling"
    if getattr(args, 'rotation', False):
        x = rotation(x)
        augmentation_tags += "_rotation"
    if getattr(args, 'permutation', False):
        x = permutation(x)
        augmentation_tags += "_permutation"
    if getattr(args, 'randompermutation', False):
        x = permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if getattr(args, 'magwarp', False):
        x = magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if getattr(args, 'timewarp', False):
        x = time_warp(x)
        augmentation_tags += "_timewarp"
    if getattr(args, 'windowslice', False):
        x = window_slice(x)
        augmentation_tags += "_windowslice"
    if getattr(args, 'windowwarp', False):
        x = window_warp(x)
        augmentation_tags += "_windowwarp"
    
    # Note: More advanced augmentation methods (spawner, wdba, dtw-based methods) 
    # require DTW implementation and are not included in this simplified version
    
    return x, augmentation_tags 