# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:21:58 2026
@author: neurogroup
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne_nirs
import mne_bids
from nilearn.plotting import plot_design_matrix
from itertools import compress
from collections import defaultdict
from collections import Counter  # used for the epoch-rejection report
from mne_nirs.io.snirf import write_raw_snirf
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
import h5py

# check that we are working in the folder where the script is
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
 
# adjusted the sensors to the head - Claude AI generated (l.25-61)
LIA_TO_RAS = np.array([[-1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 0]])

def register_montage(raw, snirf_fname):  
    """Put optodes in MNE head coordinates using the SNIRF's own fiducials.
 
    Reads source, detector and landmark positions from the file, converts
    mm to m, flips the axes to RAS, and lets set_montage() build MNE's head
    frame from Nz, LPA and RPA. Stops if the result is not anatomically
    oriented (C3 must be on the left, Cz on top).
    """
    with h5py.File(snirf_fname, "r") as f:
        unit = np.atleast_1d(f["nirs/metaDataTags/LengthUnit"][()])[0].decode()
        scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[unit]
        probe = f["nirs/probe"]
        src = np.asarray(probe["sourcePos3D"][()], dtype=float)[:, :3] * scale
        det = np.asarray(probe["detectorPos3D"][()], dtype=float)[:, :3] * scale
        lm = np.asarray(probe["landmarkPos3D"][()], dtype=float)[:, :3] * scale
        labels = [v.decode() for v in np.atleast_1d(probe["landmarkLabels"][()])]
 
    def to_ras(points):
        return points @ LIA_TO_RAS.T
 
    landmark = {name: to_ras(lm[labels.index(name)]) for name in ("Nz", "LPA", "RPA", "C3", "Cz")}
 
    # Orientation check before changing anything
    head = get_ras_to_neuromag_trans(landmark["Nz"], landmark["LPA"], landmark["RPA"])
    c3, cz = apply_trans(head, landmark["C3"]), apply_trans(head, landmark["Cz"])
    if not (c3[0] < 0 and cz[2] > 0):
        raise RuntimeError(f"Unexpected axis convention in {snirf_fname}: C3 at {c3}, Cz at {cz} (m).")
 
    # MNE's fNIRS channels are named S<i>_D<j>; positions are indexed from 1
    ch_pos = {f"S{i + 1}": p for i, p in enumerate(to_ras(src))}
    ch_pos.update({f"D{i + 1}": p for i, p in enumerate(to_ras(det))})
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, nasion=landmark["Nz"], lpa=landmark["LPA"], rpa=landmark["RPA"],
        coord_frame="unknown",
    )
    raw.set_montage(montage)  # transforms to head coordinates using the three fiducials
    return raw


# Specify BIDS root folder
bids_root = "Data/" ### NEED TO UPDATE THIS FOR YOUR SPECIFIC FILE STRUCTURE

# We have 4 data types: events, channels, optodes, and nirs.
# We will load the nirs, which corresponds to the SNIRF file.
# The other are types are the sidecar JSON files.
datatype = 'nirs'
bids_path = mne_bids.BIDSPath(root=bids_root, datatype=datatype)
nirs_files = bids_path.match()

# Prepare params - WHICH DATA WE WANT
task = 'covert'
run = '01'
suffix = 'nirs'

# Get all subjects in a sorted list
all_subjects = [file.subject for file in nirs_files]
all_subjects = sorted(list(set(all_subjects)))
print("Subjects")
print(all_subjects)

# Prepare folder for processed data 
preproc_folder = "preprocessed_data"
if not os.path.isdir(preproc_folder):
    os.mkdir(preproc_folder)

# Print output nicely
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:,.4f}".format

# Go through data for each subject 
cha_frames = []
for subject in ['01', '02', '03', '04', '05']: #could also write all_subjects

    subject_folder = f"preprocessed_data/subject_{subject}"
    if not os.path.isdir(subject_folder):
        os.mkdir(subject_folder)

    # Find the SNIRF file for this subject
    bids_path = mne_bids.BIDSPath(subject=subject, task=task, run=run,
                            suffix=suffix, datatype=datatype,
                            root=bids_root)

    print("Using BIDS file path..")
    print(bids_path)

    # Load the SNIRF file and convert to optical density
    raw_intensity = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    raw_intensity = register_montage(raw_intensity, bids_path)
    events = pd.read_csv(f'Data/sub-{subject}/nirs/sub-{subject}_task-{task}_run-{run}_events.tsv', sep='\t')
    events.trial_type = events.trial_type.str.replace(" ", "/")
    events = events[events["include"] == 1]
    annotations = mne.Annotations(events['onset'], 
                                  events['duration'], 
                                  events['trial_type'])
    raw_intensity.set_annotations(annotations)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_od.plot(n_channels=len(raw_od.ch_names), duration=500,
                show_scrollbars=False)
    plt.savefig(f'preprocessed_data/subject_{subject}/optical_density.png')

     # Calculate the scalp coupling index (SCI) and mark  #### NEED TO REFERENCE sci_limit SOMEWHERE BEFORE THIS
    # bad quality channels in the data
    sci_limit = 0.7 # to check with Cortivision
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < sci_limit))
    print(f"Bad channels with SCI < {sci_limit}:")
    print(raw_od.info["bads"])

    # Apply TDDR
    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

    # Use the MBLL to convert into hemoglobin changes
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo.resample(4)
    write_raw_snirf(raw_haemo, f'preprocessed_data/subject_{subject}/preprocessed.snirf')



