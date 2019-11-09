"""
Set up
"""
import os
import pandas as pd
import pprint
import numpy as np

from bids import BIDSLayout


# from the paper, time in seconds each scan covers
SCAN_TIME = 1.6 + 0.023  # time plus echo time
USEFUL_SCANS = 370  # from the description of the t2-weighted scans, supposedly there's 370 unless that's a typo


def get_food_temptation_data() -> BIDSLayout:
    """
    This project is hardcoded to work with this specific OpenNeuro dataset
    """
    ds_path = os.path.join("data", "ds000157")
    print("Using dataset path: \"{}\"".format(ds_path))

    layout = BIDSLayout(ds_path)
    print(layout)

    return layout


def get_experiment_details(layout: BIDSLayout):
    """
    Learn more about our experiment
    """
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


def display_experiment_details(layout: BIDSLayout):
    desc, meta = get_experiment_details(layout)

    pprint.pprint(desc)
    pprint.pprint(meta.get_dict())


def get_subject_details(layout: BIDSLayout):
    """
    Learn more about our subjects in the experiment
    """

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


def display_subject_details(layout: BIDSLayout):
    q, a = get_subject_details(layout)

    # I believe q is missing diet_success, which I gather is self rating 1-5 of how well your diet is going
    df_q = pd.DataFrame.from_dict(q.get_dict())
    df_a = a.get_df()

    print(df_q)
    print(df_a.head())


def _get_subject_data(layout: BIDSLayout, subject: str):
    """
    Gathers subject specific experimental data
    """

    # Get the high resolution T1-weighted anatomical MRI scan
    """
    In addition to the functional scan, a high resolution T1-weighted anatomical MRI scan was
    made (3D gradient echo sequence, repetition time = 8.4 ms, echo time = 3.8 ms,
    flip angle = 8◦, FOV= 288 mm × 288 mm × 175 mm, 175 sagittal slices, voxel
    size = 1 mm × 1 mm × 1 mm).
    """
    T1w_images = layout.get(
        datatype='anat',
        subject=subject,
        extension='nii.gz',
        suffix='T1w',
    )
    assert len(T1w_images) == 1, "Bad data read"
    T1w_image = T1w_images[0]

    # Get the functional scan
    """
    The functional scan was a T2-weighted gradient echo 2D-echo planar
    imaging sequence (64 × 64 matrix, repetition time = 1600 ms, echo time = 23 ms,
    flip angle = 72.5◦, FOV= 208 × 119 × 256 mm, SENSE factor AP = 2.4, 30 axial
    3.6 mm slices with 0.4 mm gap, reconstructed voxel size = 4 mm × 4 mm × 4 mm).
    In one functional run 370 scans were made (∼10 min). 
    """
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

    # TODO something.tags looked interesting
    return T1w_image, bold_image, events


def get_all_subject_data(layout: BIDSLayout):
    """
    Get the relevant data for every subject in the experiment
    """
    data = []

    for subject in layout.get_subjects():
        T1w_image, bold_image, events = _get_subject_data(layout, subject)

        data.append(
            (T1w_image.get_image(), bold_image.get_image(), events.get_df()),
        )

    return data


def get_scan_assignments(num_scans: int, events: pd.DataFrame):
    assignments = {
        "food": [],
        "nonfood": [],
        "break": [],
        "unassigned": [],
    }

    for timestep in range(num_scans):

        estimated_time = timestep * SCAN_TIME  # should be either 1600ms or 1623ms

        # Get the event with the highest onset covered by the estimated time of the scan
        event_proximity = events.onset - estimated_time
        latest_elapsed_event = len(event_proximity[event_proximity <= 0.0].index) - 1  # convert len to index
        matched_event = events.iloc[latest_elapsed_event]

        if timestep >= USEFUL_SCANS:
            assignments['unassigned'].append(timestep)
        else:
            assignments[matched_event.trial_type].append(timestep)

    return assignments


def get_machine_learning_data():
    """
    Get brain scans as labeled data, partitioned into Train, Validate, and Test
    """
    np.random.seed(2019)

    layout = get_food_temptation_data()

    subject_data = get_all_subject_data(layout)

    data = []
    for _t1, b, e in subject_data[0:1]:
        image_data = b.get_data()
        num_scans = image_data.shape[-1]

        print("Total number of scans: ", num_scans)
        scan_assignments = get_scan_assignments(num_scans, e)
        print(list("{}: {}".format(k, len(v)) for k, v in scan_assignments.items()))

        # Gather our classification examples
        # Ignore break / unassigned for now
        # Use a numeric 1 (food image) and 0 (nonfood image) for our task
        for timestep in scan_assignments['food']:
            data.append((image_data[:, :, :, timestep], 1))
        for timestep in scan_assignments['nonfood']:
            data.append((image_data[:, :, :, timestep], 0))

    # Stack our information in numpy arrays
    scans, labels = zip(*data)
    scans = np.stack(scans, axis=-1)
    labels = np.array(labels)

    # Shuffle and partition our options
    index_df = pd.DataFrame(data=np.arange(len(data)), columns=["data_index"])
    train_ix, validate_ix, test_ix = np.split(
        index_df.sample(frac=1),
        [int(.7 * len(index_df)),
         int(.9 * len(index_df))])

    # TODO Is it concerning there's a stray newaxis at the end of the indexing?
    X_train = scans[:, :, :, train_ix].squeeze()
    y_train = labels[train_ix].squeeze()
    X_val = scans[:, :, :, validate_ix].squeeze()
    y_val = labels[validate_ix].squeeze()
    X_test = scans[:, :, :, test_ix].squeeze()
    y_test = labels[test_ix].squeeze()

    return X_train, y_train, X_val, y_val, X_test, y_test
