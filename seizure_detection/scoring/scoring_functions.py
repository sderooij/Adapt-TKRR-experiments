from seizure_data_processing.post_processing.scoring import event_scoring, labels_to_events, any_ovlp
from seizure_data_processing.post_processing.post_process import optimize_bias
from seizure_data_processing.config import SEIZE_IT_DIR
from seizure_data_processing.datasets.seize_it import load_annotations
import numpy as np
import pandas as pd

import timescoring as ts
from timescoring.annotations import Annotation
from timescoring import scoring
from epilepsy2bids.annotations import Annotations # to transform annotations to correct format
# from szcore_evaluation.evaluate import Result   # to add results from different files


class Result(scoring._Scoring):
    """Helper class built on top of scoring._Scoring that implements the sum
    operator between two scoring objects. The sum corresponds to the
    concatenation of both objects.
    Adapted from szcore_evaluation.evaluate.Result
    Args:
        scoring (scoring._Scoring): initialized as None (all zeros) or from a
                                    scoring._Scoring object.
    """

    def __init__(self, score: scoring._Scoring = None):
        if score is None:
            self.fs = 0
            self.numSamples = 0
            self.tp = 0
            self.fp = 0
            self.refTrue = 0
        else:
            self.fs = score.ref.fs
            self.numSamples = score.numSamples
            self.tp = score.tp
            self.fp = score.fp
            self.refTrue = score.refTrue

    def __add__(self, other_result: scoring._Scoring):
        new_result = Result()
        new_result.fs = other_result.fs
        new_result.numSamples = self.numSamples + other_result.numSamples
        new_result.tp = self.tp + other_result.tp
        new_result.fp = self.fp + other_result.fp
        new_result.refTrue = self.refTrue + other_result.refTrue

        return new_result

    def __iadd__(self, other_result: scoring._Scoring):
        self.fs = other_result.fs
        self.numSamples += other_result.numSamples
        self.tp += other_result.tp
        self.fp += other_result.fp
        self.refTrue += other_result.refTrue

        return self


def calculate_scores_per_file(filename, output, file_info, parameters, fs=1, mask=None, overlap=0.5, sample_duration=2):
    file_duration = file_info.loc[file_info['file'] == filename, 'file_duration'].values[0]
    if mask is not None:
        start_mask = mask[0]
        stop_mask = mask[1]
        if stop_mask == file_duration and start_mask == 0:
            return None, None

    # load predictions and convert to events
    hyp = labels_to_events(
        output['predicted_label'].values, output['start_time'].values, overlap=overlap,
        seglen=sample_duration,
        min_duration=10.0,
        pos_percent=0.8, total_duration=file_duration, to_dataframe=False
    )
    # hyp = [(hyp['start_time'].values[i], hyp['stop_time'].values[i]) for i in range(len(hyp))]

    # load annotations
    ref_file = filename.replace(".edf", "_a1.tsv")  # reference file
    ref = load_annotations(SEIZE_IT_DIR + ref_file)
    # fill in stop time if not present
    ref.loc[ref['stop_time'].isna(), 'stop_time'] = ref.loc[ref['stop_time'].isna(), 'start_time'] + 10
    if mask is not None:
        ref = ref[(ref['stop_time'] < start_mask) | (ref['start_time'] > stop_mask)]  # only events outside of the
        # mask
    ref = [(ref['start_time'].values[i], ref['stop_time'].values[i]) for i in range(len(ref))]

    hyp = Annotations.loadEvents(hyp, file_duration)
    hyp = Annotation(hyp.getMask(fs), fs)
    ref = Annotations.loadEvents(ref, file_duration)
    ref = Annotation(ref.getMask(fs), fs)

    # compute scores
    sample_score = scoring.SampleScoring(ref, hyp)
    event_score = scoring.EventScoring(ref, hyp, param=parameters)
    return sample_score, event_score


def calculate_scores_per_group(output_df, file_df, parameters, group_col="group", optim_bias=False, bias_score="f3",
                               fs=1, sample_duration=2, des_overlap=0.5, mask_per_group=None, min_sensitivity=None):

    gr_output = output_df.groupby(group_col)

    sample_results = dict()
    event_results = dict()
    for i, (group, output) in enumerate(gr_output):
        # %% Determine optimal threshold based on Fbeta score
        if optim_bias:
            bias = optimize_bias(output["predicted_output"].values, output["true_label"].values, metric=bias_score,
                                 min_sensitivity=min_sensitivity)
            output["predicted_output"] = output["predicted_output"] + bias
            output["predicted_label"] = np.sign(output["predicted_output"].values)
        # %% Calculate the event f1_scores

        # NEW STUFF
        output['filename']=output['filename'].str.split("seize_it/").str[1]
        output_per_file = output.groupby("filename")
        sample_results[group] = Result()
        event_results[group] = Result()

        for j, (filename, out) in enumerate(output_per_file):
            if mask_per_group is not None:
                if filename in mask_per_group[group].keys():
                    mask = mask_per_group[group][filename]
                else:
                    mask = None
            else:
                mask = None
            sample_score, event_score =  calculate_scores_per_file(filename,
                                                                   out,
                                                                   file_df,
                                                                   parameters,
                                                                   fs=fs,
                                                                   mask=mask,
                                                                   overlap=des_overlap,
                                                                   sample_duration=sample_duration
                                                                   )
            if sample_score is None and event_score is None:
                continue
            else:
                if mask is not None:
                    mask_duration = mask[1] - mask[0]   # parts of file not in validation fold of this group
                    # event_score.duration -= mask_duration
                    event_score.numSamples -= mask_duration * event_score.fs
                    # sample_score.duration -= mask_duration
                    sample_score.numSamples -= mask_duration * sample_score.fs
                sample_results[group] += Result(sample_score)
                event_results[group] += Result(event_score)


        sample_results[group].computeScores()
        event_results[group].computeScores()

    return sample_results, event_results


def aggregate_results(sample_results, event_results, *, mode="mean"):
    """
    Code from evaluate.py of szcore_evaluation package
    Args:
        sample_results: sample-based results per group
        event_results: event-based results per group
        mode: "mean" or "cumulative"

    Returns:
        aggregated_sample_results: aggregated sample-based results, mean and std or cumulative
        aggregated_event_results: aggregated event-based results, mean and std or cumulative
    """
    aggregated_sample_results = dict()
    aggregated_event_results = dict()
    if mode == "mean":
        for result_builder, aggregated_result in zip(
                (sample_results, event_results),
                (aggregated_sample_results, aggregated_event_results),
        ):
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = np.nanmean(
                    [getattr(x, metric) for x in result_builder.values()]
                )
                aggregated_result[f"{metric}_std"] = np.nanstd(
                    [getattr(x, metric) for x in result_builder.values()]
                )
    elif mode == "cumulative":
        for result_builder, aggregated_result in zip(
                (sample_results, event_results),
                (aggregated_sample_results, aggregated_event_results),
        ):
            result_builder["cumulated"] = Result()
            for result in result_builder.values():
                result_builder["cumulated"] += result
            result_builder["cumulated"].computeScores()
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = getattr(result_builder["cumulated"], metric)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return aggregated_sample_results, aggregated_event_results


