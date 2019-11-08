"""
Set up
"""
import os
import pandas as pd
import pprint
import numpy as np


from bids import BIDSLayout


ds_path = os.path.join("data", "ds000157")
print("Using dataset path: \"{}\"".format(ds_path))

layout = BIDSLayout(ds_path)
print(layout)


"""
Learn more about our experiment
"""

def get_experiment_details(layout: BIDSLayout):
    description = layout.get_dataset_description()

    # layout.get_entities()

    # Get the metadata
    metadata = layout.get(
        extension='json',
        suffix='bold',
        task='passiveimageviewing',
    )
    assert len(metadata) == 1, "Bad data read"
    metadata = metadata[0]

    # Get the subject answers
    return description, metadata


desc, meta = get_experiment_details(layout)

pprint.pprint(desc)
pprint.pprint(meta.get_dict())


"""
Learn more about our subjects in the experiment
"""

def get_subject_details(layout: BIDSLayout):
    # Get the subject questionaire
    questionaires = layout.get(
        extension='json',
        suffix='participants',
    )
    assert len(questionaires) == 1, "Bad data read"
    questionaire = questionaires[0]

    # Get the subject answers
    answers = layout.get(
        extension='tsv',
        suffix='participants',
    )
    assert len(answers) == 1, "Bad data read"
    answers = answers[0]

    return questionaire, answers


q, a = get_subject_details(layout)

# I believe q is missing diet_success, which I gather is self rating 1-5 of how well your diet is going
df_q = pd.DataFrame.from_dict(q.get_dict())
df_a = a.get_df()

print(df_q)
print(df_a.head())


"""
Gather our neuro imaging data

what is T1w vs task nii.gz?
why are events all identical?
# anat / T1w.nii.gz
# func / task _bold.nii.gz
# events.tsv

"""

# TODO doing one at a time for avoiding memory bloat
subjects = layout.get_subjects()


def get_data(layout: BIDSLayout, subject: str):
    # Get the ????? image
    T1w_images = layout.get(
        datatype='anat',
        subject=subject,
        extension='nii.gz',
        suffix='T1w',
    )
    assert len(T1w_images) == 1, "Bad data read"
    T1w_image = T1w_images[0]

    # Get the ????? image
    bold_images = layout.get(
        datatype='func',
        subject=subject,
        extension='nii.gz',
        suffix='bold',
        task='passiveimageviewing',
    )
    assert len(bold_images) == 1, "Bad data read"
    bold_image = bold_images[0]

    # Get the classification truths
    found_events = layout.get(
        datatype='func',
        subject=subject,
        extension='tsv',
        suffix='events',
        task='passiveimageviewing',
    )
    assert len(found_events) == 1, "Bad data read"
    events = found_events[0]

    # something.tags looked interesting
    return T1w_image, bold_image, events


t_img, b_img, e = get_data(layout, subjects[0])

nibabel_t_img = t_img.get_image()
nibabel_b_img = b_img.get_image()
df_e = e.get_df()

print("T shape: ", nibabel_t_img.shape)
print("B shape: ", nibabel_b_img.shape)
print(df_e.head())
