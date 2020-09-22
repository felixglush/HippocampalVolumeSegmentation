"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    intersection = np.sum((a>0)*(b>0))
    volumes = np.sum(a>0) + np.sum(b>0)
    
    if volumes == 0:
        return -1

    return 2. * float(intersection) / float(volumes)
    

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    intersection = np.sum((a>0)*(b>0))
    union = np.sum(a>0) + np.sum(b>0) - intersection

    if union == 0:
        return -1

    return intersection / union

def perf_metrics(pred, gt):   
    negative = gt == 0
    positive = gt > 0
    
    tp = np.sum(positive[pred==gt]) # gt > 0 and pred == gt
    tn = np.sum(negative[pred==gt]) # gt == 0 and pred == gt
    fp = np.sum(negative[pred!=gt]) # gt == 0 but pred != gt
    fn = np.sum(positive[pred!=gt]) # gt > 0 but pred != gt
    
    sens = Sensitivity(tp, fn)
    spec = Specificity(tn, fp)
    dc = Dice3d(pred, gt)
    jc = Jaccard3d(pred, gt) 
    
    results = {'tp': tp, 'tn': tn, 
               'fp': fp, 'fn': fn,
               'sens': sens, 'spec': spec,
               'dice': dc, 'jaccard': jc}
    
    return results

def Sensitivity(tp, fn):
    # Sensitivity = TP/(TP+FN)
    
    if (tp + fn) == 0:
        return -1
 
    return tp / (tp + fn)

def Specificity(tn, fp):
    # Specificity = TN/(TN+FP)
    
    if (tn + fp) == 0:
        return -1
    
    return tn / (tn + fp)
