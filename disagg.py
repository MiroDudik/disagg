# Import general libraries

import pandas as pd
import numpy as np

from scipy.stats import norm
from sklearn.linear_model import ElasticNet
from sklearn.utils import check_random_state

EPS = 1e-6

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Perform stratified sampling on dataframe df with respect to the variable in stratify_by.
This can also gurantee the presence of the target_var if it is binary and specified.
'''
def stratified_sample(df:pd.DataFrame,
                      stratify_by:str = "grp_name",
                      n:int = 5000, # approximate total size of the final dataset
                      replace:bool = False,
                      train_test:bool = False,
                      random_state = None):
    random_state = check_random_state(random_state)
    new_df = df.groupby(by=stratify_by, group_keys=False).apply(
        lambda x: x.sample(n=rnd_round(x.shape[0]/df.shape[0] * n, random_state=random_state),
                           replace=replace,
                           random_state=random_state))
    if train_test and not replace:
        test_index = df.index.difference(new_df.index)
        test_df = df.loc[test_index,:]
        return new_df, test_df
    else:
        return new_df


def rnd_round(x:float, *, random_state = None):
    random_state = check_random_state(random_state)
    int_part = int(np.floor(x))
    frac_part = x - int_part
    return int_part + int(random_state.random() < frac_part)


####################################################################################################
# DISAGGREGATED EVALUATION METHODS
####################################################################################################

def partial_ridge(X, y, *, alpha, selected, sample_weight=None):
    if alpha==np.infty:
        X = X.copy()
        X.loc[:, ~selected] = 0
        alpha = 1
    wX = X.mul(sample_weight, axis=0)
    diag_selected = pd.DataFrame(data=np.diag(~selected), index=selected.index, columns=selected.index)
    M = alpha * sample_weight.sum() * diag_selected + wX.T @ X
    beta = pd.Series(np.linalg.pinv(M, hermitian=True) @ (wX.T @ y), index=selected.index)
    yhat = X @ beta
    return yhat, beta

def split_df(*, df, metric, feature_names=[]):
    split_df = df.groupby('grp_name')
    X = split_df[feature_names].mean()
    counts = split_df['grp_name'].count()
    if metric is None:
        z = None
    elif type(metric) == str:
        z = split_df[metric].mean()
    else:
        z = split_df.apply(lambda x: metric(x.y_true, x.y_pred))
        select_na = z.isna()
        z[select_na] = 0.0
        counts[select_na] = 0
    return X, z, counts

def naive_estimate(*, df, metric, ci=None):
    n = df.shape[0]
    _, naive_val, counts = split_df(df=df, metric=metric)
    index = naive_val.index

    naive = pd.DataFrame(index=index)
    naive['val'] = naive_val
    naive['count'] = counts
    nboot = 100
    if (ci is not None):
        naive_boot = pd.DataFrame(index=index)
        for b in range(nboot):
            df_b = stratified_sample(df, n=n, replace=True, random_state=b)
            _, mu_b, counts_b = split_df(df=df_b, metric=metric)
            mu_b[counts_b==0] = np.nan
            naive_boot[b] = mu_b
        for uniform_var in [True, False]:
            var, _, _ = bootstrap_variance(
                boot_mu=naive_boot, counts=counts, uniform_var=uniform_var)
            naive[f"var_u{uniform_var:d}"] = var
        for conf in ci:
            lower = (1 - conf/100.0)/2
            upper = conf/100.0 + lower
            lower_label = f"ci{conf:02d}l"
            upper_label = f"ci{conf:02d}u"
            naive[[lower_label, upper_label]] = naive_boot.quantile(q=[lower, upper], axis=1).T
    return naive

def structured_regression(*, df, metric, feature_names, alpha_list, weighted=True, sparser=True, cv=False, ci=None):
    n = df.shape[0]
    X, z, counts = split_df(df=df, metric=metric, feature_names=feature_names)
    index = z.index

    naive = naive_estimate(df=df, metric=metric, ci=[])
    select = (counts>0)
    var = naive['var_u1']
    
    if weighted:
        sample_weight = counts
    else:
        sample_weight = 1.0*(counts > 0)
    sample_weight = sample_weight / sample_weight.mean()
    
    lasso_args = {
        'l1_ratio': 0.95,
        'selection': 'random',
        'max_iter': 10000,
        'random_state': 33
    }
    # ElasticNet with sample_weight = sw minimizes the following problem:
    #   1/2 * [sum_i sw_i * (y_i - x_i @ beta)^2] / [sum_i sw_i] + alpha * penalty
    # where
    #   penalty = l1_ratio * ||beta||_1 + 1/2 * (1 - l1_ratio) * ||beta||^2_2
    #
    # The expected value of the first term is:
    #   1/2 * [sum_i sw_i * var_i] / [sum_i sw_i]
    # So, equalizing the loss and penalty, for ridge (when l1_ratio = 0), we should approximately have
    #   alpha = [sum_i sw_i * var_i] / [sum_i sw_i] / ||beta_OLS||^2_2
    #
    # The lasso with
    #   1/2 * [sum_i sw_i * (y_i - x_i @ beta)^2] + lambda * penalty
    # Should have approximately
    #   lambda = max_j sqrt[sum_i sw_i^2 * var_i * x_ij^2]
    # So, adjusting for rescaling:
    #   alpha = (max_j ...) / [sum_i sw_i]
    
    X_col_norm = X.pow(2).mul(sample_weight.pow(2) * var, axis=0).loc[select,:].sum().pow(0.5)
    alpha_mul = X_col_norm.max() / sample_weight[select].sum()
    
    X_int = X.copy()
    X_int['intercept'] = 1.0
    select_int = pd.Series(X_int.columns == 'intercept', index=X_int.columns)
    
    _, beta_ridge = partial_ridge(X=X_int, y=z, alpha=EPS,
                                  selected=select_int, sample_weight=sample_weight)
    alpha_mul_ridge = 0.5 * (sample_weight[select] * var[select]).sum() / sample_weight[select].sum()
    alpha_mul_ridge = alpha_mul_ridge / beta_ridge[~select_int].pow(2).sum()

    zhat_by_alpha = pd.DataFrame()
    coef_by_alpha = pd.DataFrame()
    for alpha in alpha_list:
        lasso = ElasticNet(alpha=alpha_mul*alpha, **lasso_args).fit(X, z, sample_weight=sample_weight)
        zhat_by_alpha[alpha] = pd.Series(lasso.predict(X), index=index)
        coef_by_alpha[alpha] = pd.Series(lasso.coef_, index=lasso.feature_names_in_)
        coef_by_alpha.loc['intercept',alpha] = lasso.intercept_

    cv_folds = 10
    cv_frac = 0.9
    alpha_cv = None
    result_zhat = pd.DataFrame()
    result_coef = pd.DataFrame()
    cv_by_alpha = pd.DataFrame()

    result_zhat['naive'] = z.copy()
    result_zhat['naive'][counts==0] = np.nan
    result_zhat['naive_var_u1'] = naive['var_u1']
    result_zhat['count'] = counts
    if cv:
        cv_res = pd.DataFrame(index=range(cv_folds), columns=alpha_list)
        for i in range(cv_folds):
            train, test = stratified_sample(df, n=cv_frac*n, replace=False, train_test=True, random_state=i)
            X_train, z_train, counts_train = split_df(df=train, metric=metric, feature_names=feature_names)
            X_test, z_test, counts_test = split_df(df=test, metric=metric, feature_names=feature_names)
            sample_weight_train = sample_weight[z_train.index] * (counts_train>0).astype(float) 
            sample_weight_test = sample_weight[z_test.index] * (counts_test>0).astype(float)
            for alpha in alpha_list:
                lasso = ElasticNet(alpha=alpha_mul*alpha, **lasso_args).fit(X_train, z_train, sample_weight=sample_weight_train)
                z_predict = pd.Series(lasso.predict(X_test), index=X_test.index)
                cv_res.loc[i,alpha] = sample_weight_test @ (z_test-z_predict).pow(2)
        cv_means = cv_res.mean().astype(float)
        cv_sterr = cv_res.std() / np.sqrt(cv_folds)
        if sparser:
            best_robust = (cv_means+cv_sterr).min()
            alpha_cv = cv_means.index[cv_means < best_robust].max() 
        else:
            alpha_cv = cv_means.idxmin()
        cv_by_alpha['mean'] = cv_means
        cv_by_alpha['sterr'] = cv_sterr
        result_zhat['val'] = zhat_by_alpha[alpha_cv]
        result_coef['val'] = coef_by_alpha[alpha_cv]

    nboot = 100
    if (ci is not None) and (alpha_cv is not None):
        z_ols, _ = partial_ridge(X=X_int, y=z, alpha=np.infty, selected=result_coef['val'].abs()>EPS, sample_weight=sample_weight)
        z_ridge, _ = partial_ridge(X=X_int, y=z, alpha=alpha_mul_ridge, selected=result_coef['val'].abs()>EPS, sample_weight=sample_weight)
        result_zhat['val_ols'] = z_ols
        result_zhat['val_lpr'] = z_ridge
        
        z_ridge_rboot = pd.DataFrame()
        for b in range(nboot):
            resids_b = var.pow(0.5) * pd.Series(norm.rvs(size=z.size, random_state=b), index=z.index)
            resids_b[~select] = 0.0
            z_b = z_ols + resids_b

            X_b = X.copy()
            counts_b = counts
            valid_z_b = (counts_b>0)
            sample_weight_b = sample_weight[z_b.index] * valid_z_b
            lasso_b = ElasticNet(alpha=alpha_mul*alpha_cv, **lasso_args).fit(X_b, z_b, sample_weight=sample_weight_b)
            coef_b = pd.Series(lasso_b.coef_, index=lasso_b.feature_names_in_)
            coef_b['intercept'] = lasso_b.intercept_
            
            X_b['intercept'] = 1.0
            z_ridge_b, _ = partial_ridge(X=X_b, y=z_b, alpha=alpha_mul_ridge, selected=coef_b.abs()>EPS, sample_weight=sample_weight_b)
            z_ridge_b[~valid_z_b] = np.nan
            z_ridge_rboot[b] = z_ridge + (z_ols - z_ridge_b)
        
        boot_result_rlpr = boot_ci_analysis(zhat_boot=z_ridge_rboot, ci=ci, suffix="_rlpr")
        result_zhat[boot_result_rlpr.columns] = boot_result_rlpr

    return result_zhat, result_coef, zhat_by_alpha, coef_by_alpha, cv_by_alpha, alpha_cv

def bootstrap_variance(*, boot_mu, counts, uniform_var, invalid_val=np.nan):
    vars = boot_mu.var(axis=1)
    valid_vars = (counts>1) & (~vars.isna())
    vars[~valid_vars] = 0.0
    
    sigma2_single = (vars*counts.pow(2))[valid_vars].sum() / (counts-1)[valid_vars].sum()

    if uniform_var:
        sigma2 = sigma2_single / counts
        sigma2_inv = counts / sigma2_single
        valid_sigma = (counts>0)
    else:
        sigma2 = vars*counts/(counts-1)
        sigma2_inv = 1 / sigma2
        valid_sigma = valid_vars & (vars>EPS)
    
    sigma2[~valid_sigma] = invalid_val
    sigma2_inv[~valid_sigma] = invalid_val
    return sigma2, sigma2_inv, valid_sigma

def calculate_shrinkage(*, means, valid_means, sigma2, sigma2_inv, valid_sigma, w, eb, mom_flag):
    # assume w.sum() == 1
    mom = w @ means
    result = pd.DataFrame(index=means.index)
    posterior_var = None
    if mom_flag:
        shrinkage = 0
    elif eb:
        # Empirical Bayes derivation
        # Consider model:
        #   Z_a = mu + eps_a, E[eps2_a] = tau2 + sigma2_a
        # Let:
        #   mom = sum_a w_a Z_a = mu + sum_a w_a eps_a
        # Then:
        #   E[(Z_a - mom)^2] 
        #     = E[(sum_a' w_a' eps_a' - eps_a)^2]
        #     = sum_{a'\ne a} w_a'^2 E[eps2_a'] + (1-w_a)^2 E[eps2_a]
        #     = sum_a' w_a'^2 E[eps2_a'] + (1 - 2w_a) E[eps2_a]
        # And so:
        #   TOTAL = E[sum_a w_a(Z_a - mom)^2]
        #     = sum_a' w_a'^2 E[eps2_a'] + sum_a w_a(1 - 2w_a) E[eps2_a]
        #     = sum_a  w_a(1 - w_a) E[eps2_a]
        #     = sum_a  w_a(1 - w_a) (tau2 + sigma2_a)
        #   PARTIAL
        #     = sum_a  w_a(1 - w_a) sigma2_a
        #   TOTAL - PARTIAL
        #     = sum_a  w_a(1 - w_a) tau2
        #     = (1 - sum_a w2_a) tau2
        # Posterior (following Gelman):
        #    hmu_a = mom + [tau2 / (tau2 + sigma2_a)] * (Z_a - mom)
        #    hvar_a = tau2 * sigma2_a / (tau2 + sigma2_a)
        total = w @ (means - mom).pow(2)
        partial = sigma2 @ (w*(1-w))
        tau2 = (total - partial) / (1 - w @ w)
        tau2 = max(tau2, 0.0)
        shrinkage = tau2 / (tau2 + sigma2)
        shrinkage[~valid_sigma] = 0.0
        posterior_var = tau2 * (1 - shrinkage)
        revised_mom = (means / (sigma2 + tau2))[valid_sigma].sum() / (1 / (sigma2 + tau2))[valid_sigma].sum()
        mom = revised_mom
    else:
        # Bock's JS, with effective dimension = d, following Feldman et al. (2012)
        shrinkage = 1 - (valid_sigma.sum()-3) / (sigma2_inv @ (means-mom).pow(2))
        shrinkage = max(shrinkage, 0.0)
    
    muhat = mom + shrinkage*(means - mom)
    muhat[~valid_means] = mom
    result['val'] = muhat

    if posterior_var is not None:
        result['posterior_var'] = posterior_var

    return result

def shrinkage(*, df, metric, weighted_mom=True, eb=False, mom_flag=False):
    n = df.shape[0]
    _, means, counts = split_df(df=df, metric=metric)
    index = means.index
    valid_means = (counts>0)
    
    nboot=100
    boot_mu = pd.DataFrame(index=index, columns=range(nboot))
    for b in range(nboot):
        df_b = stratified_sample(df, n=n, replace=True, random_state=b)
        _, mu_b, counts_b = split_df(df=df_b, metric=metric)
        mu_b[counts_b==0] = np.nan
        boot_mu[b] = mu_b

    sigma2, sigma2_inv, valid_sigma = bootstrap_variance(
        boot_mu=boot_mu, counts=counts, uniform_var=True, invalid_val=0.0)
    
    if weighted_mom:
        w = sigma2_inv / sigma2_inv.sum()
    else:
        w = valid_sigma / valid_sigma.sum()
    
    result = calculate_shrinkage(means=means, valid_means=valid_means,
                                 sigma2=sigma2, sigma2_inv=sigma2_inv, valid_sigma=valid_sigma, w=w,
                                 eb=eb, mom_flag=mom_flag)

    result['naive'] = means.copy()
    result['naive'][~valid_means] = np.nan
    result['count'] = counts

    return result

def boot_ci_analysis(*, zhat_boot, ci=None, suffix=""):
    result = pd.DataFrame()

    result[f"boot_var{suffix}"] = zhat_boot.var(axis=1)
    result[f"boot_mean{suffix}"] = zhat_boot.var(axis=1)
    result[[f"median{suffix}"]] = zhat_boot.quantile(q=[0.5], axis=1).T
    if ci is not None:
        for conf in ci:
            lower = (1 - conf/100.0)/2
            upper = conf/100.0 + lower
            lower_label = f"ci{conf:02d}l{suffix}"
            upper_label = f"ci{conf:02d}u{suffix}"
            result[[lower_label, upper_label]] = zhat_boot.quantile(q=[lower, upper], axis=1).T

    return result
