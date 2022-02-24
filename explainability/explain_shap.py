"""

"""
import shap
from matplotlib import pyplot as plt


def explain_shap(model, features, x_train, y_train, x_test, y_test):
    attrib_data = x_train[:200]
    explainer = shap.KernelExplainer(model.predict_proba, attrib_data)
    num_explanations = 20
    shap_vals = explainer.shap_values(x_test[:num_explanations])

    #shap.summary_plot(shap_vals, feature_names=features)
    plot = shap.force_plot(explainer.expected_value[0], shap_vals[0], x_test[:num_explanations])
    plot.html()
    #plt.show()
    # shap.plots.beeswarm(shap_values_ebm, max_display=14)
    # fig,ax = shap.partial_dependence_plot(
    #     "RM", model_xgb.predict, X, model_expected_value=True,
    #     feature_expected_value=True, show=False, ice=False,
    #     shap_values=shap_values_ebm[sample_ind:sample_ind+1,:]
    # )
    #shap.plots.scatter(shap_values_xgb[:,"RM"], color=shap_values)