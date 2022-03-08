
class OptimizerDict:

    ada_delta = 'adadelta'
    ada_grad = 'ada_grad'
    ada_max = 'adamax'
    adam = 'adam'
    asgd = 'asgd'
    lbfgs = 'lbfgs'
    rmsprop = 'rmsprop'
    rprop = 'rprop'
    sparse_adam = 'sparse_adam'
    sgd = 'sgd'


class ActivationDict:

    elu = 'elu'
    hard_shrink = 'hard_shrink'
    hard_tanh = 'hard_tanh'
    leaky_relu = 'leaky_relu'
    log_sigmoid = 'log_sigmoid'
    p_relu = 'p_relu'
    relu = 'relu'
    relu_6 = 'relu_6'
    r_relu = 'r_relu'
    selu = 'selu'
    celu = 'celu'
    sigmoid = 'sigmoid'
    soft_plus = 'soft_plus'
    soft_shrink = 'soft_shrink'
    soft_sign = 'soft_sign'
    tanh = 'tanh'
    tanh_shrink = 'tanh_shrink'
    threshold = 'threshold'

    softmin = 'softmin'
    softmax = 'softmax'
    softmax_2d = 'softmax_2d'
    log_softmax = 'log_softmax'
    adaptive_log_softmax_with_loss = 'adaptive_log_softmax_with_loss'


class LossDict:

    L1 = 'L1_loss'
    mse = 'mse_loss'
    cross_entropy = 'cross_entropy_loss'
    ctc = 'ctc_loss'
    nll = 'NLL_loss'
    poisson_nll = 'poisson_nll_loss'
    kl_div = 'KL_Div_loss'
    binary_cross_entropy = 'bce_loss'
    bce_with_logits = 'bce_with_logits_loss'
    margin_ranking = 'margin_ranking_loss'
    hinge_embedding = 'hinge_embedding_loss'
    multi_label_margin = 'multi_label_margin_loss'
    smooth_1L = 'smooth_1L_loss'
    soft_margin = 'soft_margin_loss'
    multi_label_soft_margin = 'multi_label_soft_margin_loss'
    cosine_embedding = 'cosine_embedding_loss'
    multi_margin = 'multi_margin_loss'
    triplet_margin = 'triplet_margin_loss'

