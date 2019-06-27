import os
import argparse
import matplotlib.pyplot as plt

# Global constants.
LR_STR = "learning rate of group 0 to "
TR_AC_STR = "Training Accuracy: "
VA_AC_STR = "Validation Accuracy: "
BASELINE_COLOR='#1f77b4'
EXPERIMENT_COLOR='#ff7f0e'

def parse_args():
    parser = argparse.ArgumentParser(description='Generate loss and acc curves from log files')
    parser.add_argument('--base_file', required=True, help='Baseline log file.')
    parser.add_argument('--exp_file', required=True, help='Experiment log file.')
    parser.add_argument('--title', required=True, type=str, nargs='+',
                        help='Plot title. Can have multiple lines.')
    parser.add_argument('--iter_per_epoch', default=195, type=int, help='Number of iterations per epoch.')
    parser.add_argument('--result_dir', default='./../Plots', type=str, help='Result directory to save plots.')
    parser.add_argument('--plot_count', default=1, type=int, help='Plot count, used for file name.')
    args = parser.parse_args()
    return args

def get_info_from_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    lr_iters, lrs = [], []
    tr_acc, va_acc = [], []
    running_loss = []
    for line in lines:
        line = line.strip()
        if LR_STR in line:
            e_num = []
            for c in line[5:]:
                if c==':':
                    break
                if c==' ':
                    continue
                e_num.append(c)
            e_num = int(''.join(e_num)) + 1
            lr_iters.append(e_num*args.iter_per_epoch)
            LR_STR_idx = line.index(LR_STR)
            cur_lr = line[LR_STR_idx+len(LR_STR):]
            cur_lr = cur_lr[:-1] if cur_lr[-1]=='.' else cur_lr
            cur_lr = float(cur_lr)
            lrs.append(cur_lr)
        if line.startswith("Train Iteration"):
            temp_str = "Loss = "
            idx = line.index(temp_str)
            running_loss.append(float(line[idx+len(temp_str):]))
        if TR_AC_STR in line:
            ta_idx = line.index(TR_AC_STR)
            acc = []
            for c in line[ta_idx+len(TR_AC_STR):]:
                if c==' ':
                    break
                acc.append(c)
            acc = float(''.join(acc))
            tr_acc.append(acc)
        if VA_AC_STR in line:
            ta_idx = line.index(VA_AC_STR)
            acc = []
            for c in line[ta_idx+len(VA_AC_STR):]:
                if c==' ':
                    break
                acc.append(c)
            acc = float(''.join(acc))
            va_acc.append(acc)
    return lr_iters, lrs, running_loss, tr_acc, va_acc

def plot_vals(baseline, experiment, b_label, e_label, x_label, y_label, title,
                baseline_2=None, experiment_2=None, b_label_2=None, e_label_2=None,
                plot_count=0, b_lr_iters=None, b_lrs=None, e_lr_iters=None, e_lrs=None):
    _y_val = max(max(baseline), max(experiment)) - 0.1
    x_len = len(baseline)
    if b_lr_iters is not None and b_lrs is not None:
        for iter,cur_lr in zip(b_lr_iters,b_lrs):
            if iter <= x_len:
                plt.axvline(x=iter, linestyle='--', lw=0.5, color=BASELINE_COLOR)
                plt.text(iter-260, _y_val, 'LR='+str(cur_lr), rotation=90, color=BASELINE_COLOR)

    if e_lr_iters and e_lrs:
        for iter,cur_lr in zip(e_lr_iters,e_lrs):
            if iter <= x_len:
                plt.axvline(x=iter, linestyle='--', lw=0.5, color=EXPERIMENT_COLOR)
                plt.text(iter-260, _y_val-0.7, 'LR='+str(cur_lr), rotation=90, color=EXPERIMENT_COLOR)

    plt.plot(range(len(baseline)), baseline, label=b_label, color=BASELINE_COLOR)
    if baseline_2:
        plt.plot(range(len(baseline_2)), baseline_2, label=b_label_2, color=BASELINE_COLOR, linestyle='--')
    plt.plot(range(len(experiment)), experiment, label=e_label, color=EXPERIMENT_COLOR)
    if experiment_2:
        plt.plot(range(len(experiment_2)), experiment_2, label=e_label_2, color=EXPERIMENT_COLOR, linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(args.result_dir, 'Plot_'+str(plot_count)+'.jpg'), dpi=400, bbox_inches='tight')
    plt.clf()

# hack to fix lrs, sometimes learning rate can be printed as 0 in log file
# assumes lr decay rate = 0.1
def lr_fix_hack(lrs):
    for i, lr in enumerate(lrs):
        if i!=0 and lr == 0:
            lrs[i] = lrs[i-1]*0.1
    return lrs

if __name__ == "__main__":
    args = parse_args()
    args.title = '\n'.join(args.title)
    print(args.title)

    lr_iters_1, lrs_1, running_loss_1, tr_acc_1, va_acc_1 = get_info_from_log(args.base_file)
    lr_iters_2, lrs_2, running_loss_2, tr_acc_2, va_acc_2 = get_info_from_log(args.exp_file)

    lrs_1 = lr_fix_hack(lrs_1)
    lrs_2 = lr_fix_hack(lrs_2)

    min_len = min(len(running_loss_1), len(running_loss_2))
    running_loss_1 = running_loss_1[:min_len]
    running_loss_2 = running_loss_2[:min_len]

    min_len = min(min(len(tr_acc_1),len(tr_acc_2)), min(len(va_acc_1),len(va_acc_2)))
    tr_acc_1 = tr_acc_1[:min_len]
    tr_acc_2 = tr_acc_2[:min_len]
    va_acc_1 = va_acc_1[:min_len]
    va_acc_2 = va_acc_2[:min_len]

    plot_vals(running_loss_1, running_loss_2, 'Baseline', 'Experiment', 'Iterations', 'Running Loss', args.title,
                    baseline_2=None, experiment_2=None, b_label_2=None, e_label_2=None,
                    plot_count=args.plot_count, b_lr_iters=lr_iters_1, b_lrs=lrs_1, e_lr_iters=lr_iters_2, e_lrs=lrs_2)

    plot_vals(tr_acc_1, tr_acc_2, 'Baseline Training', 'Experiment Training', 'Iterations', 'Accuracy', args.title,
                    baseline_2=va_acc_1, experiment_2=va_acc_2, b_label_2='Baseline Validation', e_label_2='Experiment Validation',
                    plot_count=args.plot_count+1, b_lr_iters=None, b_lrs=None, e_lr_iters=None, e_lrs=None)
