import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import pickle
import json
from pytracking.evaluation.environment import env_settings
from pytracking.analysis.extract_results import extract_results, get_summary_sizes
import numpy as np

def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []
    
    std_success_rate_plot_overlap_merged = []
    std_success_rate_plot_center_merged = []
    std_success_rate_plot_center_norm_merged = []
    std_overlap_all_merged = []

    min_success_rate_plot_overlap_merged = []
    min_success_rate_plot_center_merged = []
    min_success_rate_plot_center_norm_merged = []
    min_overlap_all_merged = []

    max_success_rate_plot_overlap_merged = []
    max_success_rate_plot_center_merged = []
    max_success_rate_plot_center_norm_merged = []
    max_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))
        
        std_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].std(1))
        std_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].std(1))
        std_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].std(1))
        std_overlap_all_merged.append(avg_overlap_all[:, match].std(1))

        min_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].min(1)[0])
        min_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].min(1)[0])
        min_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].min(1)[0])
        min_overlap_all_merged.append(avg_overlap_all[:, match].min(1)[0])

        max_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].max(1)[0])
        max_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].max(1)[0])
        max_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].max(1)[0])
        max_overlap_all_merged.append(avg_overlap_all[:, match].max(1)[0])

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    std_success_rate_plot_overlap_merged = torch.stack(std_success_rate_plot_overlap_merged, dim=1)
    std_success_rate_plot_center_merged = torch.stack(std_success_rate_plot_center_merged, dim=1)
    std_success_rate_plot_center_norm_merged = torch.stack(std_success_rate_plot_center_norm_merged, dim=1)
    std_overlap_all_merged = torch.stack(std_overlap_all_merged, dim=1)

    min_success_rate_plot_overlap_merged = torch.stack(min_success_rate_plot_overlap_merged, dim=1)
    min_success_rate_plot_center_merged = torch.stack(min_success_rate_plot_center_merged, dim=1)
    min_success_rate_plot_center_norm_merged = torch.stack(min_success_rate_plot_center_norm_merged, dim=1)
    min_overlap_all_merged = torch.stack(min_overlap_all_merged, dim=1)

    max_success_rate_plot_overlap_merged = torch.stack(max_success_rate_plot_overlap_merged, dim=1)
    max_success_rate_plot_center_merged = torch.stack(max_success_rate_plot_center_merged, dim=1)
    max_success_rate_plot_center_norm_merged = torch.stack(max_success_rate_plot_center_norm_merged, dim=1)
    max_overlap_all_merged = torch.stack(max_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()
    
    eval_data['std_success_rate_plot_overlap'] = std_success_rate_plot_overlap_merged.tolist()
    eval_data['std_success_rate_plot_center'] = std_success_rate_plot_center_merged.tolist()
    eval_data['std_success_rate_plot_center_norm'] = std_success_rate_plot_center_norm_merged.tolist()
    eval_data['std_overlap_all'] = std_overlap_all_merged.tolist()

    eval_data['min_success_rate_plot_overlap'] = min_success_rate_plot_overlap_merged.tolist()
    eval_data['min_success_rate_plot_center'] = min_success_rate_plot_center_merged.tolist()
    eval_data['min_success_rate_plot_center_norm'] = min_success_rate_plot_center_norm_merged.tolist()
    eval_data['min_overlap_all'] = min_overlap_all_merged.tolist()

    eval_data['max_success_rate_plot_overlap'] = max_success_rate_plot_overlap_merged.tolist()
    eval_data['max_success_rate_plot_center'] = max_success_rate_plot_center_merged.tolist()
    eval_data['max_success_rate_plot_center_norm'] = max_success_rate_plot_center_norm_merged.tolist()
    eval_data['max_overlap_all'] = max_overlap_all_merged.tolist()

    return eval_data

def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return  disp_name


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts, stddev=None, maxs=None, mins=None):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])
        if stddev is not None:
            ax.fill_between(x.tolist(), (y[id_sort, :]-stddev[id_sort, :]).tolist(), (y[id_sort, :]+stddev[id_sort, :]).tolist(), color=plot_draw_styles[index_sort.numel() - id - 1]['color'], alpha=0.2)
        elif maxs is not None and mins is not None:
            ax.fill_between(x.tolist(), (mins[id_sort, :]).tolist(), (maxs[id_sort, :]).tolist(), color=plot_draw_styles[index_sort.numel() - id - 1]['color'], alpha=0.2)
            
        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()


def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    # Load data
    settings = env_settings()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path) and not force_evaluation:
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        # print('Pre-computed evaluation data not found. Computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)

    if not check_eval_data_is_valid(eval_data, trackers, dataset):
        # print('Pre-computed evaluation data invalid. Re-computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)
    else:
        # Update display names
        tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                         for t in trackers]
        eval_data['trackers'] = tracker_names

    return eval_data

def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc

def get_std_curve(std_success_rate_plot_overlap, valid_sequence):
    std_success_rate_plot_overlap = std_success_rate_plot_overlap[valid_sequence, :, :]
    std_curve = std_success_rate_plot_overlap.mean(0) * 100.0
    std = std_curve.mean(-1)
    return std_curve, std

def get_mins_maxs_curve(mins_success_rate_plot_overlap, maxs_success_rate_plot_overlap, valid_sequence):
    mins_success_rate_plot_overlap = mins_success_rate_plot_overlap[valid_sequence, :, :]
    mins_curve = mins_success_rate_plot_overlap.mean(0) * 100.0
    mins = mins_curve.mean(-1)
    
    maxs_success_rate_plot_overlap = maxs_success_rate_plot_overlap[valid_sequence, :, :]
    maxs_curve = maxs_success_rate_plot_overlap.mean(0) * 100.0
    maxs = maxs_curve.mean(-1)
    return mins_curve, maxs_curve, mins, maxs

def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score


def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, show_stddev=False, show_mins_maxs=False, **kwargs):
    """
    Plot results for the given trackers

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success',
                    'prec' (precision), and 'norm_prec' (normalized precision)
    """

    # Load data
    settings = env_settings()

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    print('\nGenerating plots for: {}'.format(report_name))

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        std_curve = None
        mins_curve = None
        maxs_curve = None
        
        if show_stddev:
            std_success_rate_plot_overlap = torch.tensor(eval_data['std_success_rate_plot_overlap'])
            std_curve, std = get_std_curve(std_success_rate_plot_overlap, valid_sequence)
            
        elif show_mins_maxs:
            mins_success_rate_plot_overlap = torch.tensor(eval_data['min_success_rate_plot_overlap'])
            maxs_success_rate_plot_overlap = torch.tensor(eval_data['max_success_rate_plot_overlap'])

            
            mins_curve, maxs_curve, mins, maxs = get_mins_maxs_curve(mins_success_rate_plot_overlap, maxs_success_rate_plot_overlap, valid_sequence)
            
        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts, stddev=std_curve, mins=mins_curve, maxs=maxs_curve)

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 100), 'title': 'Normalized Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       norm_precision_plot_opts)

    # ********************************  Summary Size Plot **************************************
    if 'summary_size' in plot_types:
        summary_sizes, num_iters_abs, num_iters_diff = get_summary_sizes(trackers, dataset, report_name)

        fig_summary, ax_summary = plt.subplots()

        ave_summary_sizes = []
        std_summary_sizes = []
        merged = np.zeros(len(trackers))
        new_tracker_names = []

        for trk_id, trk in enumerate(trackers):
            if merged[trk_id]:
                continue
            base_tracker = trackers[trk_id]
            new_tracker_names.append(base_tracker.display_name)

            match = trk_id
            if merge_results:
                match = [t.name == base_tracker.name and t.parameter_name == base_tracker.parameter_name for t in trackers]
                match = np.array(match)
            
            ave_summary_sizes.append(np.nanmean(summary_sizes[:, match, :],(0,1)))
            std_summary_sizes.append(np.nanstd(summary_sizes[:, match, :],(0,1)))
            print(np.nanmean(summary_sizes[:, match, :],1).shape)
            merged[match] = 1

        SMALL_SIZE = 1.3*16
        BIGGER_SIZE = 1.3*18

        for trk_id, avg_summary_sz in enumerate(ave_summary_sizes):
            std_summary_sz = std_summary_sizes[trk_id]
            #avg_summary_sz = (avg_summary_sz-15)*2
            xaxis = np.arange(0, len(avg_summary_sz))
            #xaxis = np.log(xaxis)/np.log(10)
            line, = ax_summary.plot(xaxis, avg_summary_sz.tolist(), color = plot_draw_styles[trk_id]['color'])
            ax_summary.fill_between(xaxis, (avg_summary_sz-std_summary_sz).tolist(), (avg_summary_sz+std_summary_sz).tolist(), color = plot_draw_styles[trk_id]['color'], alpha=0.2)
            line.set_label(new_tracker_names[trk_id])

        num_frames = len(avg_summary_sz)
        xaxis = np.arange(0, num_frames)
        yaxis = xaxis * 2. / 20.
        #xaxis = np.log(xaxis)/np.log(10)
        #ax_summary.plot(xaxis, yaxis, color="black")
        ax_summary.set_title("Summary size vs. frames for %s"%(report_name), fontsize=BIGGER_SIZE)
        ax_summary.set_xlabel("Frame number", fontsize=SMALL_SIZE)
        ax_summary.set_ylabel("Summary size", fontsize=SMALL_SIZE)
        ax_summary.legend(prop={"size":SMALL_SIZE})
        ax_summary.tick_params(axis='x', labelsize=SMALL_SIZE)
        ax_summary.tick_params(axis='y', labelsize=SMALL_SIZE)
        plt.savefig("summary_sizes_"+report_name+".png", bbox_inches='tight')

    if 'estimated_ops' in plot_types:
        # shape of each is: [num_sequences, num_trackers, num_frames]
        summary_sizes, num_iters_abs, num_iters_diff = get_summary_sizes(trackers, dataset, report_name)

        fig_ops, ax_ops = plt.subplots()

        ave_ops = []
        std_ops = []
        merged = np.zeros(len(trackers))
        new_tracker_names = []

        for trk_id, trk in enumerate(trackers):
            if merged[trk_id]:
                continue
            base_tracker = trackers[trk_id]
            new_tracker_names.append(base_tracker.display_name)

            match = trk_id
            if merge_results:
                match = [t.name == base_tracker.name and t.parameter_name == base_tracker.parameter_name for t in trackers]
                match = np.array(match)

            est_ops_per_frame = summary_sizes[:, match, :] * num_iters_diff[:, match, :]
            est_cum_ops = np.cumsum(est_ops_per_frame, axis=-1)
            
            ave_ops.append(np.nanmean(est_cum_ops,(0,1)))
            std_ops.append(np.nanstd(est_cum_ops,(0,1)))
            merged[match] = 1

        SMALL_SIZE = 1.3*16
        BIGGER_SIZE = 1.3*18

        for trk_id, avg_op in enumerate(ave_ops):
            std_op = std_ops[trk_id]
            xaxis = np.arange(0, len(avg_op))
            #xaxis = np.log(xaxis)/np.log(10)
            line, = ax_ops.plot(xaxis, avg_op.tolist(), color = plot_draw_styles[trk_id]['color'])
            ax_ops.fill_between(xaxis, (avg_op-std_op).tolist(), (avg_op+std_op).tolist(), color = plot_draw_styles[trk_id]['color'], alpha=0.2)
            line.set_label(new_tracker_names[trk_id])

        ax_ops.set_title("Est. cumulative learning ops. vs. frames for %s"%(report_name), fontsize=BIGGER_SIZE)
        ax_ops.set_xlabel("Frame number", fontsize=SMALL_SIZE)
        ax_ops.set_ylabel("Est. cumulative operations", fontsize=SMALL_SIZE)
        ax_ops.legend(prop={"size":SMALL_SIZE})

        ax_ops.tick_params(axis='x', labelsize=SMALL_SIZE)
        ax_ops.tick_params(axis='y', labelsize=SMALL_SIZE)
        ax_ops.set_ylim([0, 2000])
        
        plt.savefig("estimated_ops_"+report_name+".png", bbox_inches='tight')
    plt.show()
    

def generate_formatted_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text


def print_results(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)


def plot_got_success(trackers, report_name):
    """ Plot success plot for GOT-10k dataset using the json reports.
    Save the json reports from http://got-10k.aitestunion.com/leaderboard in the directory set to
    env_settings.got_reports_path

    The tracker name in the experiment file should be set to the name of the report file for that tracker,
    e.g. DiMP50_report_2019_09_02_15_44_25 if the report is name DiMP50_report_2019_09_02_15_44_25.json

    args:
        trackers - List of trackers to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load data
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    # Load results
    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1:
            raise Exception

        # First field is the tracker name. Index it out
        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id,
                              'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, **kwargs):
    """ Print per-sequence results for the given trackers. Additionally, the sequences to list can be filtered using
    the filter criteria.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        filter_criteria - Filter sequence results which are reported. Following modes are supported
                        None: No filtering. Display results for all sequences in dataset
                        'ao_min': Only display sequences for which the minimum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences where at least one tracker performs poorly.
                        'ao_max': Only display sequences for which the maximum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences all tracker performs poorly.
                        'delta_ao': Only display sequences for which the performance of different trackers vary by at
                                    least filter_criteria['threshold'] in average overlap (AO) score. This mode can
                                    be used to select sequences where the behaviour of the trackers greatly differ
                                    between each other.
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    # Filter sequences
    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist())) if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)

    print(report_text)
