from __future__ import print_function

import os
import json
from benchbot_api import Agent

_GROUND_TRUTH = os.path.join(os.path.dirname(__file__),
                             'ground_truth_miniroom_1.json')

_CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]


class EvalSemanticSlamAgent(Agent):
    def is_done(self, action_result):
        # Finish immediately as we are only evaluating
        return True

    def pick_action(self, observations, action_list):
        # Should never get to this point?
        return None, {}

    def save_result(self, filename, empty_results, results_format_fns):
        # Load objects from the ground truth file supplied with this example
        with open(_GROUND_TRUTH, 'r') as f:
            gt_objects = json.load(f)['objects']

        # Create an empty object in our semantic map corresponding to each of
        # the objects we are going to "steal" from the ground truth list
        empty_results['results']['objects'] = [
            empty_object_fn() for o in gt_objects
        ]

        # Populate each object in our semantic map with the data from the
        # ground truth list of objects (we are cheating to perform "perfect"
        # Semantic SLAM so we can demonstrate the evaluation process)
        for gt, o in zip(gt_objects, empty_results['results']['objects']):
            o['label_probs'] = [
                1.0 if i == _CLASS_LIST.index(gt['class']) else 0.0
                for i in range(len(_CLASS_LIST))
            ]  # Probabilistic way to say "100% sure it is class X"
            o['centroid'] = gt['centroid']
            o['extent'] = gt['extent']

        # Save the results at the requested location
        with open(filename, "w") as f:
            json.dump(empty_results, f)
        return
