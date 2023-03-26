import numpy as np
import numpy.linalg
import random
import torch

# Relevant links:
#   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
#   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)


# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.
def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n



    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

def umeyama_pytorch(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - torch.mean(P, dim=0)
    centeredQ = Q - torch.mean(Q, dim=0)

    C = centeredP.T @ centeredQ / n

    V, S, W = torch.svd(C)
    W = W.T
    d = (torch.det(V) * torch.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = V @ W


    varP = torch.sum(torch.var(P, dim=0, unbiased=False))
    c = 1 / varP * torch.sum(S) # scale factor

    t = torch.mean(Q, dim=0) - torch.mean(P, dim=0).matmul(c * R)

    return c, R, t

def umeyama_ransac(P, Q, epsilon=0.2, n_iters=80):
    assert P.shape == Q.shape
    inliner_set = []
    point_length = P.shape[0]
    for i in range(n_iters):
        sampled_points = random.sample(list(range(point_length)), 4)
        sampled_P = P[sampled_points]
        sampled_Q = Q[sampled_points]
        c, R, t = umeyama(sampled_P, sampled_Q)

        projected_P = P @ R * c + t
        new_inliner_set = []
        for j in range(point_length):
            if np.linalg.norm(projected_P[j] - Q[j], ord=2) < epsilon:
                new_inliner_set.append(j)
        if len(new_inliner_set) > len(inliner_set):
            inliner_set = new_inliner_set

    sampled_P = P[inliner_set]
    sampled_Q = Q[inliner_set]
    c, R, t = umeyama(sampled_P, sampled_Q)
    return c, R, t

def umeyama_dim_2(P, Q):
    assert P.shape == Q.shape
    n, dim1 = P.shape

    centeredP = P
    centeredQ = Q

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n



    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

if __name__ == '__main__':
    a = np.random.normal(size=(15, 3))
    b = np.random.normal(size=(15, 3))

    result1 = umeyama(a.copy(), b.copy())
    print(result1)
    result2 = umeyama_pytorch(torch.from_numpy(a), torch.from_numpy(b))
    print(result2)