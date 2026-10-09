#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:17:15 2026

@author: crito25 and carsi10
this script has to be saved in a folder in which we have another folder named Data for the participant data
and then I do a change
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
from collections import Counter  # CHANGED [C13]: used for the epoch-rejection report
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
import h5py 

LIA_TO_RAS = np.array([[-1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 0]])
def register_montage(raw, snirf_fname):  # CHANGED [C18]: new helper
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

# To import one subject
#base_dir = '/Users/crito25/Library/CloudStorage/OneDrive-Linköpingsuniversitet'
#dataset = '/Documents/ds007738'
#os.chdir(f'{base_dir}{dataset}')
snirf_path = "Data/sub-01/nirs/sub-01_task-resting_run-01_nirs.snirf"
raw = mne.io.read_raw_snirf(snirf_path)
raw = register_montage(raw, snirf_path)

events = pd.read_csv('Data/sub-01/nirs/sub-01_task-covert_run-01_events.tsv', 
                     sep='\t')
events = events[events["include"] == 1]
events.trial_type = events.trial_type.str.replace(" ", "/")
annotations = mne.Annotations(events['onset'], 
                              events['duration'], 
                              events['trial_type'])
raw.set_annotations(annotations)


subjects_dir = str(mne.datasets.sample.data_path()) + '/subjects'
brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.9" 
    # Cortex argument is the colour of the brain 0 black, 1 white
)
brain.add_sensors(
    raw.info,
    trans="fsaverage",
    fnirs=["channels", "pairs", "sources", "detectors"],
)
brain.show_view(azimuth=20, elevation=60, distance=400)


# We can plot intensity or optical density or hemoglobin

picks = mne.pick_types(raw.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw.info, picks=picks
)
raw.pick(picks[dists > 0.02])
raw.plot(
    n_channels=len(raw.ch_names), duration=500, show_scrollbars=False
)


raw_od = mne.preprocessing.nirs.optical_density(raw)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, 
            show_scrollbars=False)

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500, 
               show_scrollbars=False)

# You can also filter the heart rate out
raw_haemo_unfiltered = raw_haemo.copy()
raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_haemo).items():
    fig = _raw.compute_psd().plot(
        average=True, amplitude=False, picks="data", exclude="bads"
    )
    fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
    
# Find events across time
events, event_dict = mne.events_from_annotations(raw_haemo)
fig = mne.viz.plot_events(events, event_id=event_dict, 
                          sfreq=raw_haemo.info["sfreq"])


# Extract epochs

reject_criteria = dict(hbo=80e-6)
tmin, tmax = -5, 15

epochs = mne.Epochs(
    raw_haemo,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    #reject=reject_criteria,
    reject_by_annotation=False,
    proj=True,
    baseline=(None, 0),
    preload=True,
    detrend=None,
    verbose=True,
)
epochs.plot_drop_log()

# Plot the mean activation per epoch on Hbo and dHBO for Covert condition and control
epochs["Right"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
)

epochs["Left"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6), 
                         layout="constrained")

        
clims = dict(hbo=[-20, 20], hbr=[-20, 20])
epochs["Right"].average().plot_image(axes=axes[:, 0], clim=clims, show=False)
epochs["Left"].average().plot_image(axes=axes[:, 1], clim=clims, show=False)
for column, condition in enumerate(["Right", "Left"]):
    for ax in axes[:, column]:
        ax.set_title(f"{condition}: {ax.get_title()}")



evoked_dict = {
    "Right/HbO": epochs["Right"].average(picks="hbo"), # It also works if you put Covert/Left or Covert/Right
    "Right/HbR": epochs["Right"].average(picks="hbr"),
    "Left/HbO": epochs["Left"].average(picks="hbo"),
    "Left/HbR": epochs["Left"].average(picks="hbr"),
}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Left=dict(linestyle="dashed"))


mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
)

times = np.arange(-3.5, 13.2, 3.0)
topomap_args = dict(extrapolate="local")
epochs["Covert"].average(picks="hbo").plot_joint(
    times=times, topomap_args=topomap_args
)

times = np.arange(4.0, 11.0, 1.0)
epochs["Covert/Left"].average(picks="hbo").plot_topomap(times=times, 
                                                        **topomap_args)
epochs["Covert/Right"].average(picks="hbo").plot_topomap(times=times, 
                                                         **topomap_args)

epochs["Covert/Left"].average(picks="hbr").plot_topomap(times=times, 
                                                        **topomap_args)
epochs["Covert/Right"].average(picks="hbr").plot_topomap(times=times, 
                                                         **topomap_args)


fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(9, 5),
    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]),
    layout="constrained",
)
vlim = (-8, 8)
ts = 9.0

evoked_left = epochs["Covert/Left"].average()
evoked_right = epochs["Covert/Right"].average()

evoked_left.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 0], vlim=vlim, colorbar=False, 
    show=False,  **topomap_args
)
evoked_left.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 0], vlim=vlim, colorbar=False, 
    show=False,**topomap_args
)
evoked_right.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 1], vlim=vlim, colorbar=False, 
    show=False,**topomap_args
)
evoked_right.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 1], vlim=vlim, colorbar=False, 
    show=False,**topomap_args
)

evoked_diff = mne.combine_evoked([evoked_left, evoked_right], weights=[1, -1])

evoked_diff.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 2:], vlim=vlim, colorbar=True, 
    show=False,**topomap_args
)
evoked_diff.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 2:], vlim=vlim, colorbar=True, 
    show=False,**topomap_args
)

for column, condition in enumerate(["Covert Left", 
                                    "Covert Right", "Left-Right"]):
    for row, chroma in enumerate(["HbO", "HbR"]):
        axes[row, column].set_title(f"{chroma}: {condition}")
        
        
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), 
                         layout="constrained")
mne.viz.plot_evoked_topo(
    epochs["Left"].average(picks="hbo"), color="b", axes=axes, 
    show=False, legend=False
)
mne.viz.plot_evoked_topo(
    epochs["Right"].average(picks="hbo"), color="r", axes=axes, 
    show=False, legend=False
)

# Tidy the legend:
leg_lines = [line for line in axes.lines if line.get_c() == "b"][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == "r"][0])
fig.legend(leg_lines, ["Left", "Right"], loc="lower right")

# To plot only one channel (we can get separate info on left/right if we make a more complete dictionary)
# mne.viz.plot_compare_evokeds(
#     evoked_dict, picks=['S2_D1'])

# mne.viz.plot_compare_evokeds(
#     evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
# )

#%%

####### NEW ADDITION FROM THE MNE_ANALYSIS PIPELINE --- APPEARS TO WORK ########

def individual_analysis(raw_haemo, plot_design=False):

    # Cut out just the short channels for creating a GLM repressor
    #short_chans = mne_nirs.channels.get_short_channels(raw_haemo)
    long_chans = mne_nirs.channels.get_long_channels(raw_haemo)

    # Create a design matrix
    design_matrix = mne_nirs.experimental_design.make_first_level_design_matrix(
        long_chans,
        drift_model="cosine",
        high_pass=0.005,  # Must be specified per experiment
        hrf_model="spm",
        stim_dur=5.0,
    )

    # Append short channels to design matrix
    #design_matrix["ShortHbO"] = np.mean(short_chans.copy().pick(picks="hbo").get_data(), axis=0)
    #design_matrix["ShortHbR"] = np.mean(short_chans.copy().pick(picks="hbr").get_data(), axis=0)

    # Run GLM
    print("Solving GLM..")
    glm_est = mne_nirs.statistics.run_glm(raw_haemo, design_matrix, 
                                          noise_model="auto")

    if plot_design:
        fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        fig = plot_design_matrix(design_matrix, ax=ax1)
        plt.show()

    return glm_est, design_matrix

# Specify BIDS root folder
bids_root = "Data/" ### NEED TO UPDATE THIS FOR YOUR SPECIFIC FILE STRUCTURE

# We have 4 data types: events, channels, optodes, and nirs.
# We will load the nirs, which corresponds to the SNIRF file.
# The other are types are the sidecar JSON files.
datatype = 'nirs'
bids_path = mne_bids.BIDSPath(root=bids_root, datatype=datatype)
nirs_files = bids_path.match()

# Prepare params
task = 'covert'
run = '01'
suffix = 'nirs'

# Get all subjects in a sorted list
all_subjects = [file.subject for file in nirs_files]
all_subjects = sorted(list(set(all_subjects)))
print("Subjects")
print(all_subjects)

# Prepare folder for processed data
glm_folder = "glm_data"
if not os.path.isdir(glm_folder):
    os.mkdir(glm_folder)

# Print output nicely
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:,.4f}".format

# Go through data for each subject 
cha_frames = []
for subject in ['01', '02', '03', '04', '05']:

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

     # Calculate the scalp coupling index (SCI) and mark  #### NEED TO REFERENCE sci_limit SOMEWHERE BEFORE THIS
    # bad quality channels in the data
    sci_limit = 0.7
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < sci_limit))
    print(f"Bad channels with SCI < {sci_limit}:")
    print(raw_od.info["bads"])

    # Apply TDDR
    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

    # Use the MBLL to convert into hemoglobin changes
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo.resample(4)

    # Run the GLM analysis
    glm_fname = f"{glm_folder}/{subject}_glm.h5"
    glm_est, design_matrix = individual_analysis(raw_haemo, plot_design=False)
    
    # You can save GLM results in HDF5 form like this if you want
    #glm_est.save(glm_fname, overwrite=True)

    # Append results per channel and ROI to larger dataframe for later export
    cha = glm_est.to_dataframe()
    cha['subject'] = subject
    cha_frames.append(cha)

cha_data = pd.concat(cha_frames, ignore_index=True)
cha_data.to_csv("glm_data/cha_data.csv")
print('done')

import statsmodels.formula.api as smf

# Let's read the file we saved..
ch_data = pd.read_csv("glm_data/cha_data.csv")
print(ch_data.head())

# Read a SNIRF (just for montage and channel names)

raw_intensity = mne.io.read_raw_snirf(snirf_path)
raw_intensity = register_montage(raw_intensity, snirf_path)
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

# Get data from the channel level frame
ch_summary_hbo = ch_data.query("Condition in ['Covert/Left', 'Covert/Right']")
ch_summary_hbo = ch_summary_hbo.copy()
ch_summary_hbo["theta"] = [t * 1.e6 for t in ch_summary_hbo["theta"]] # Convert unit for nicer plotting
ch_summary_hbo = ch_summary_hbo.query("Chroma in ['hbo']")

# Run group level mixed model and convert to dataframe
ch_model_hbo = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition",
                    ch_summary_hbo, groups=ch_summary_hbo["subject"]).fit(method='nm')
ch_model_df_hbo = mne_nirs.statistics.statsmodels_to_results(ch_model_hbo, order=raw_haemo.copy().pick("hbo").ch_names)

# One for HbR
ch_summary_hbr = ch_data.query("Condition in ['Covert/Left', 'Covert/Right']")
ch_summary_hbr = ch_summary_hbr.copy()
ch_summary_hbr["theta"] = [t * 1.e6 for t in ch_summary_hbr["theta"]] # Convert unit for nicer plotting
ch_summary_hbr = ch_summary_hbr.query("Chroma in ['hbr']")
ch_model_hbr = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition",
                    ch_summary_hbr, groups=ch_summary_hbr["subject"]).fit(method='nm')
ch_model_df_hbr = mne_nirs.statistics.statsmodels_to_results(ch_model_hbr, order=raw_haemo.copy().pick("hbr").ch_names)

# Plot 2D 
conditions = ['Covert/Left', 'Covert/Right']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10),
                        gridspec_kw=dict(width_ratios=[1, 1, 1]))  # 3 values for 3 columns

for idx, cond in enumerate(conditions):
    mne_nirs.visualisation.plot_glm_group_topo(
        raw_haemo.copy().pick(picks="hbo"),
        ch_model_df_hbo.query(f"Condition in ['{cond}']"),
        colorbar=True, axes=axes[idx],
        vlim=(-20, 20)
    )
plt.show()