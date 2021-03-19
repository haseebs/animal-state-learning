import wandb
import matplotlib.pyplot as plt

def upload_to_wandb(obsall, predall, errors, n=10000, nobs=12):
    wandb.log({'final_predictions': wandb.plot.line_series(
        xs=list(range(n)),
        ys=[obsall[:,0][-n:].tolist(), predall[-n:].tolist(), obsall[:,1][-n:].tolist(), errors[2][-n:].tolist() ],
        keys=['US', 'Predicted Value', 'CS', 'Ideal Target'],
        title=f'Last {n} obs & preds',
        xname='t')})

def plot_last_n(obsall, predall, errors=None, n=1000, nobs=12):
    fig, axs = plt.subplots(nobs+3,figsize=(30, 35))
    fig.tight_layout()
    for i in range(nobs):
        axs[i].plot(list(range(n)), obsall[:, i][-n:])

    if errors is not None:
        axs[-3].plot(list(range(n)), errors[2][-n:] )
        axs[-2].plot(list(range(n)), predall[-n:] )
        axs[-1].plot(list(range(n)), errors[1][-n:])
        print(f'MSE: {errors[0]}')

    axs[0].title.set_text('US')
    axs[1].title.set_text('CS')
    axs[-3].title.set_text('return target')
    axs[-2].title.set_text('return prediction')
    axs[-1].title.set_text('return error')
    for ax in axs:
        ax.grid(color='#666666', linestyle='-', alpha=0.5)
    return fig
