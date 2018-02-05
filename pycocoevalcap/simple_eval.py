from typing import List, Dict

from pycocoevalcap import Bleu, Rouge, Cider

__scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # TODO: Fix issues with meteor and spice metrics
    # (Meteor(), "METEOR"),
    # (Spice(), "SPICE")
]


def simple_eval(predictions: List[str], truths: List[List[str]]) -> Dict[str, List[float]]:
    predictions_dict = dict(enumerate([[p] for p in predictions]))
    truths_dict = dict(enumerate(truths))

    evaluations: Dict[str, List[float]] = {}
    for scorer, names in __scorers:
        _, scores = scorer.compute_score(predictions_dict, truths_dict)
        # scores.shape = (num_score_types, num_inputs)

        if not isinstance(names, list):
            assert not isinstance(scores, list)
            names = [names]
            scores = [scores]

        for s, n in zip(scores, names):
            if n not in evaluations:
                evaluations[n] = []
            evaluations[n].extend(s)

    return evaluations
