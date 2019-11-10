"""
Data accessors for our chosen experimental data centered on neuro-imaging brain scans
"""
import os
import pandas as pd
import pprint
import numpy as np

from bids import BIDSLayout


DATASET_PATH = os.path.join("data", "ds000157")

# from the paper, time in seconds each scan covers
SCAN_TIME = 1.6  # I believe the 0.023 echo time is contained within total time
USEFUL_SCANS = 370  # from the description of the t2-weighted scans, supposedly there's 370 unless that's a typo

TVT_SPLITS = (0.7, 0.9)


class MLDataset:
    def __init__(self,
                 X_train, y_train,
                 X_val, y_val,
                 X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.mean_image = np.mean(self.X_train, axis=0)

        self._normalized = False
        self._flattened = False

    def __repr__(self):
        return "{} | Features: {}  | Train: {} | Validation: {} | Test: {}".format(
            MLDataset.__name__,
            self.X_train.shape[1:],
            self.X_train.shape[0],
            self.X_val.shape[0],
            self.X_test.shape[0],
        )

    def normalize(self):
        if self._normalized:
            return

        # Subtract the mean training image
        self.X_train -= self.mean_image
        self.X_val -= self.mean_image
        self.X_test -= self.mean_image

        self._normalized = True

    def flatten(self):
        if self._flattened:
            return

        features = np.prod(self.X_train.shape[1:])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], features)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], features)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], features)
        # self.mean_image = self.mean_image.reshape(self.mean_image.shape[0], features)

        self._flattened = True


def get_food_temptation_data(dataset_path: str = DATASET_PATH) -> BIDSLayout:
    """
    This project is hardcoded to work with this specific OpenNeuro dataset
    """
    print("Using dataset path: \"{}\"".format(dataset_path))
    layout = BIDSLayout(dataset_path)

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


def get_subject_data(layout: BIDSLayout, subject: str):
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
    
    During the functional run, stimuli were presented on a screen...
    
    B is the T2-weighted, 
    64 (either SAGITAL/CORONAL) 
      x 64 (other of SAGITAL/CORONAL) 
      x 30 (axial/horizontal, accounts for verticallity) 
      x 370-375 (scans, accounts for TIME)
      
    The functional volumes of every subject were realigned to the first
    volume of the first run, globally normalized to Montreal Neurological Institute space
    (MNI space) retaining 4 mm × 4 mm × 4 mm voxels, and spatially smoothed with a
    gaussian kernel of 8 mm full width at half maximum. A statistical parametric map
    was generated for every subject by fitting a boxcar function to each time series, 
    convolved with the canonical hemodynamic response function. Data were high-pass
    filtered with a cutoff of 128 s. Three conditions were modeled: viewing foods, 
    viewing non-foods and the half-way break.
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
    """
    During scanning, subjects alternately viewed 24 s blocks of palatable food
    images (8 blocks) and non-food images (i.e., office utensils; 8 blocks), interspersed
    with 8–16 s rest blocks showing a crosshair (12 s on average). Halfway the task there
    was a 10 s break. In the image blocks, 8 images were presented for 2.5 s each with a
    0.5 s inter-stimulus interval. A
    """
    found_events = layout.get(
        datatype='func',
        subject=subject,
        extension='tsv',
        suffix='events',
        task='passiveimageviewing',
    )
    assert len(found_events) == 1, "Bad data read"
    events = found_events[0]

    return T1w_image.get_image(), bold_image.get_image(), events.get_df()


def get_all_subject_data(layout: BIDSLayout):
    """
    Get the relevant data for every subject in the experiment
    """
    data = []

    for subject in layout.get_subjects():
        data.append((subject, *get_subject_data(layout, subject)),)

    return data


def get_scan_assignments(num_scans: int, events: pd.DataFrame) -> dict:
    assignments = {
        "food": [],
        "nonfood": [],
        "break": [],
        "unassigned": [],
    }

    # TODO a little slow, and seems like there's only 2 assignments shared across the whole set
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


def get_ml_dataset(layout: BIDSLayout = None, limit: int = 3, splits=TVT_SPLITS) -> MLDataset:
    """
    Get brain scans as labeled data, partitioned into Train, Validate, and Test

    :param layout: optional, supply if you already have a layout on hand
    :param limit: TODO I'm figuring out some memory constraints, the data if mishandled can exceed memory
    :param splits: The proportions of train and validation (test is inferred) to split up
    :return: MLDataset class wrapping the data splits

    TODO how am I going to cope with the memory issue?
    64 * 64 * 30 = Per TimeStep (122,880)

    dtype is <f4, for little endian, 4 bit float (Did I see it was float32 at one point?)
    TS * 4 = 491,520 bits per ts or 61.44 kB

    we have ~364 * 30 TS which is about 670.9248 mB
    """
    np.random.seed(2019)

    if layout is None:
        layout = get_food_temptation_data()

    subject_data = get_all_subject_data(layout)

    data = []
    for s, _t1, b, e in subject_data[:limit]:
        print("Assigning data from subject {}".format(s))
        image_data = b.get_data()
        num_scans = image_data.shape[-1]

        # print("Total number of scans: ", num_scans)
        scan_assignments = get_scan_assignments(num_scans, e)
        # print(list("{}: {}".format(k, len(v)) for k, v in scan_assignments.items()))

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
        [int(splits[0] * len(index_df)),
         int(splits[1] * len(index_df))])

    # TODO Is it concerning there's a stray newaxis at the end of the indexing?
    X_train = scans[:, :, :, train_ix].squeeze().T
    y_train = labels[train_ix].squeeze()
    X_val = scans[:, :, :, validate_ix].squeeze().T
    y_val = labels[validate_ix].squeeze()
    X_test = scans[:, :, :, test_ix].squeeze().T
    y_test = labels[test_ix].squeeze()

    return MLDataset(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )
