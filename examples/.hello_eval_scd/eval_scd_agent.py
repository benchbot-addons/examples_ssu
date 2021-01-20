from __future__ import print_function

import os
import json
from benchbot_api import Agent

_GROUND_TRUTH_1 = os.path.join(os.path.dirname(__file__),
                               'ground_truth_miniroom_1.json')
_GROUND_TRUTH_2 = os.path.join(os.path.dirname(__file__),
                               'ground_truth_miniroom_2.json')

_CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]


class EvalScdAgent(Agent):
    def is_done(self, action_result):
        # Finish immediately as we are only evaluating
        return True

    def pick_action(self, observations, action_list):
        # Should never get to this point?
        return None, {}

    def save_result(self, filename, empty_results, results_format_fns):
        # Load objects from the ground truth files supplied with this example
        with open(_GROUND_TRUTH_1, 'r') as f:
            gt_objects_1 = json.load(f)['objects']
        with open(_GROUND_TRUTH_2, 'r') as f:
            gt_objects_2 = json.load(f)['objects']

        # Generate lists of added & removed objects
        removed_objects = [o for o in gt_objects_1 if o not in gt_objects_2]
        added_objects = [o for o in gt_objects_2 if o not in gt_objects_1]

        # Create an empty object in our semantic map corresponding to each of
        # the added or removed objects we are going to "steal" from the ground
        # truth list
        empty_results['results']['objects'] = [
            empty_object_fn() for o in removed_objects + added_objects
        ]

        # Populate each object in our semantic map of changed objects with the
        # data from the ground truth list of changes (we are cheating to
        # perform "perfect" Scene Change Detection so we can demonstrate the
        # evaluation process)
        for i, (gt, o) in enumerate(
                zip(removed_objects + added_objects,
                    empty_results['results']['objects'])):
            o['label_probs'] = [
                1.0 if i == _CLASS_LIST.index(gt['class']) else 0.0
                for i in range(len(_CLASS_LIST))
            ]  # Probabilistic way to say "100% sure it is class X"
            o['centroid'] = gt['centroid']
            o['extent'] = gt['extent']
            o['state_probs'] = (
                [0.0, 1.0, 0.0]
                if i < len(removed_objects) else [1.0, 0.0, 0.0]
            )  # Probabilistic way to say "100% sure object was added/removed"

        # Save the results at the requested location
        with open(filename, "w") as f:
            json.dump(empty_results, f)
        return
