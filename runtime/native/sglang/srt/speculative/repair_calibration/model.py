"""A small, explicit four-event classifier with structural missing-X masking.

This is a CPU reference and offline trainer, not a new CUDA inference backend.
"""
import numpy as np
from .contracts import FEATURE_NAMES


def features(p1, p2, native2_logit, best_score, depth, has_x):
    p1, p2, native2_logit, best_score, depth = np.broadcast_arrays(
        np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64),
        np.asarray(native2_logit, dtype=np.float64), np.asarray(best_score, dtype=np.float64),
        np.asarray(depth, dtype=np.float64))
    has_x = np.broadcast_to(np.asarray(has_x, dtype=bool), p1.shape)
    if (not np.isfinite(p1).all() or not np.isfinite(p2).all() or
        not np.isfinite(native2_logit).all() or not np.isfinite(depth).all() or
        not np.isfinite(best_score[has_x]).all()):
        raise ValueError("nonfinite model feature")
    if np.any((p1 < 0) | (p2 < 0) | (p1 + p2 > 1.00001)):
        raise ValueError("invalid native probabilities")
    if np.any((depth < 0) | (depth > 6)):
        raise ValueError("feature schema supports draft depths 0..6 only")
    l1, l2 = np.log(np.clip(p1, 1e-7, 1)), np.log(np.clip(p2, 1e-7, 1))
    lr, d = np.log(np.clip(1 - p1 - p2, 1e-7, 1)), depth / 6
    m = np.where(has_x, 2 * best_score - native2_logit - 2, 0)
    return np.stack([l1, l2, lr, d, p1, p2, p1 - p2, d*d, m,
                     m + l2, m*d, has_x.astype(np.float64)], axis=-1)


def masked_softmax(logits, has_x):
    logits = np.array(logits, dtype=np.float64, copy=True)
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [N,4]")
    has_x = np.asarray(has_x, dtype=bool)
    if has_x.shape != (len(logits),):
        raise ValueError("has_x shape mismatch")
    logits[~has_x, 2] = -np.inf
    logits -= np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p


def validate_parameters(p):
    f = len(FEATURE_NAMES)
    expected = {"mean": (f,), "scale": (f,), "coef": (4, f), "intercept": (4,)}
    for k, shape in expected.items():
        a = np.asarray(p.get(k), dtype=np.float64)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("invalid parameter: " + k)
    if np.any(np.asarray(p["scale"]) <= 0):
        raise ValueError("nonpositive feature scale")


def predict(parameters, x, has_x):
    validate_parameters(parameters)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("invalid feature matrix")
    z = (x - parameters["mean"]) / parameters["scale"]
    return masked_softmax(z @ np.asarray(parameters["coef"]).T + parameters["intercept"], has_x)


def fit(x, y, has_x, *, sample_weight=None, l2=0.001, maxiter=350):
    """Train-only standardization, mean masked NLL, fixed L2; no test tuning."""
    from scipy.optimize import minimize
    x, y, has_x = np.asarray(x, float), np.asarray(y, int), np.asarray(has_x, bool)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("invalid training features")
    if not len(y) or y.shape != (len(x),) or has_x.shape != y.shape:
        raise ValueError("invalid training shapes")
    if np.any((y < 0) | (y > 3) | ((y == 2) & ~has_x)):
        raise ValueError("invalid masked target class")
    if l2 <= 0:
        raise ValueError("positive fixed regularization required")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid sample weights")
    weights = weights / weights.sum()
    eye = np.eye(4)[y]
    def objective(flat):
        w = flat.reshape(4, z.shape[1])
        p = masked_softmax(z @ w.T, has_x)
        loss = -(weights * np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))).sum()
        loss += 0.5 * l2 * (w[:, :-1]**2).sum()
        grad = ((p - eye) * weights[:, None]).T @ z
        grad[:, :-1] += l2 * w[:, :-1]
        return loss, grad.ravel()
    opt = minimize(objective, np.zeros(4*z.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7})
    w = opt.x.reshape(4, z.shape[1])
    params = {"mean": mean.tolist(), "scale": scale.tolist(),
              "coef": w[:, :-1].tolist(), "intercept": w[:, -1].tolist()}
    return params, {"converged": bool(opt.success), "iterations": int(opt.nit),
                    "message": str(opt.message), "masked_nll_l2": float(opt.fun), "l2": l2}


def choose_native1_and_second(probabilities, has_x):
    """Keep the existing replacement rule fixed; selected pair is NOT renormalized."""
    p = np.asarray(probabilities)
    use = np.asarray(has_x, bool) & (p[:, 2] > p[:, 1])
    second = np.where(use, 2, 1)
    selected = np.column_stack([p[:, 0], p[np.arange(len(p)), second]])
    return second, selected


def metrics(p, y, qids, has_x):
    y, has_x = np.asarray(y), np.asarray(has_x, bool)
    second, selected = choose_native1_and_second(p, has_x)
    native_hit = (y == 0) | (y == 1)
    chosen_hit = (y == 0) | (y == second)
    use = second == 2
    rescue, loss = chosen_hit & ~native_hit, native_hit & ~chosen_hit
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))
    brier = ((p - np.eye(4)[y])**2).sum(1)
    groups = {}
    for i, key in enumerate(qids):
        groups.setdefault(key, []).append(i)
    return {"states": len(y), "questions": len(groups),
            "four_class_nll": float(nll.mean()), "four_class_brier": float(brier.mean()),
            "question_macro_nll": float(np.mean([nll[v].mean() for v in groups.values()])),
            "native_top2_hit_rate": float(native_hit.mean()),
            "selected_pair_hit_rate": float(chosen_hit.mean()),
            "replacements": int(use.sum()), "rescue": int(rescue.sum()),
            "lost_native2": int(loss.sum()), "net_local_hits": int(rescue.sum()-loss.sum()),
            "no_x_states": int((~has_x).sum()),
            "mean_selected_probability_mass": float(selected.sum(1).mean())}
