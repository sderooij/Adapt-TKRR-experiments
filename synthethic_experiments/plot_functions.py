import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary(model, X, y, title=None, ax=None, alpha=0.4, cmap='Blues', xlim=(-1, 1), ylim=(-1, 1),
                           save_path=None, x_hilight=None, plot_method='contourf'):
    if ax is None:
        _, ax = plt.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(model, X,
                                                  response_method='predict',
                                                  alpha=alpha,
                                                  cmap=cmap,
                                                  ax=ax,
                                                  plot_method=plot_method)
    # Overlay the original data points
    colormap = plt.get_cmap('tab20') #'#006BA4'
    disp.ax_.scatter(X[y == -1, 0], X[y == -1, 1], color='#FF800E', alpha=1, marker='.', s=70)
    disp.ax_.scatter(X[y == 1, 0], X[y == 1, 1], c='#006BA4', alpha=1, facecolors='none', marker='^',
                     s=70)
    if x_hilight is not None:
        disp.ax_.scatter(x_hilight[:,0], x_hilight[:,1], color='k', alpha=0.8, marker='s')
    disp.ax_.set_title(title)
    disp.ax_.set_xlim(xmin=xlim[0], xmax=xlim[1])
    disp.ax_.set_ylim(ymin=ylim[0], ymax=ylim[1])
    if save_path is not None:
        plt.savefig(save_path)
    return disp