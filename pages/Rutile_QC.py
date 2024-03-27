import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import json
import base64
import requests
from pymatgen.core import Structure
from powdiffrac.processing import scale_min_max
from powdiffrac.simulation import peak_shapes, generate_noise, Powder
from scipy.ndimage import binary_dilation, convolve1d
import time

st.set_page_config(page_title="Rutile Sample QC", page_icon="🎯")

st.markdown("# TiO$_2$ Nanoparticle Quality control based on spectra")
st.write("Identify impure rutile samples based on their powder XRD pattern")
st.sidebar.header("Rutile QC Demo")

MODEL_NAME = "rutile"
MODEL_URL = st.secrets["model_url"]

TS_USER = st.secrets["ts_user"]
TS_PWD = st.secrets["ts_password"]

@st.cache_data
def get_powder():
    url = "http://www.crystallography.net/cod/9007432.cif"
    res = requests.get(url)
    struct = Structure.from_str(res.content.decode("UTF-8"), fmt="cif")
    powder = Powder(
        struct,
        two_theta=(20,70),
        step_size=0.02,
        max_domain_size=20,
        max_strain=0.01,
        vary_strain=True,
        vary_domain=True
    )
    powder._domain_size = 0
    return powder

def get_signals(
    powder_class,
    n = 25
):
    steps = powder_class.steps
    signals = np.zeros([n, steps.size])
    for i in range(n):
        signals[i] = powder_class.get_signal(vary=True)
    return signals

# variation of speedscreen algorithm
def generate_variations(
    x,
    step_size=0.01,
    impurity_raio = 0.2,
    max_multi_peak = 0.3,
    detection_threshold = 0.5,
    num_multi_peaks = 2,
    restricted_area = 25,
    fwhm_range = (0.2, 0.4),
    noise_lvl = (0.01, 0.03),
    seed = None,
):
    # patterns should be scaled between 0 and 1
    if np.max(x) != 1.0:
        x = scale_min_max(x)

    num_scans, datapoints = x.shape
    rng = np.random.default_rng(seed)

    # %% impurity augmentation

    # before we add the multiphase peaks, we have to draw random noise levels
    # ensure that noise does not cover multi-phase peaks
    noise_lvls = rng.uniform(noise_lvl[0], noise_lvl[1], size=(num_scans))

    # select indices of scans to transform into multiphase
    multi_indices = rng.choice(
        np.arange(num_scans),
        np.ceil(num_scans * impurity_raio).astype(int),
        replace=False,
    )
    # get position of single phase peaks in batch
    cur_peak_pos = np.sum(x, axis=0).astype(bool)
    # dilate positions to avoid too much overlap between single and multiphase peaks
    restricted = binary_dilation(cur_peak_pos, iterations=restricted_area)
    restricted[:200] = True
    restricted[-150:] = True
    # get eligible positions for multiphase peaks
    elig_pos = np.arange(datapoints)[np.invert(restricted)]

    # position of multiphase peaks
    add_pos = rng.choice(elig_pos, (multi_indices.size, num_multi_peaks), replace=True)
    peak_heights = np.random.uniform(
        detection_threshold, max_multi_peak, size=add_pos.shape
    )
    # slice and reshape noise level array to compare with multi-phase peak heights
    # ensure that extra peaks are detectable
    noise_comparison = np.repeat(
        noise_lvls[multi_indices, None], num_multi_peaks, axis=1
    )
    # identify multi-phase peaks too small for recognition (lost in noise)
    mask = np.any(peak_heights < noise_comparison * 2.0 / 3, axis=1)
    # correct height to ensure detectability
    peak_heights[mask] = noise_comparison[mask]

    # array for those multi-phase peaks
    add_peaks = np.zeros_like(x)

    # fill array
    add_peaks[multi_indices[:, None], add_pos] = peak_heights

    # combine additional peaks with signle-phase patterns

    x += add_peaks
    y = np.zeros([num_scans])
    y[multi_indices] = 1.

    # %% convolve peak shapes

    fwhm = rng.uniform(fwhm_range[0], fwhm_range[1], size=num_scans)
    eta = rng.uniform(0.1, 0.9, size=num_scans)
    for n in np.arange(num_scans):
        kernel = peak_shapes.get_pseudo_voigt_kernel(fwhm[n], step_size, eta[n])
        x[n] = convolve1d(x[n], kernel, mode="constant")

    x = scale_min_max(x)

    # correct absolute intensities due to peak broadening
    x *= np.sqrt(np.maximum(1 - fwhm, 0.3))[:, None]
    # x *= np.maximum(1 - fwhm, 0.35)[:, None]
    x = generate_noise(
        x,
        noise_lvl=0
    )
    # add noise, clip extreme values
    gaus = 1 / 3 * np.clip(rng.normal(0, 1, x.shape), -3, 3)
    gaus = (gaus * 0.5) + 0.5
    # scale noise
    gaus *= noise_lvls[:, None]
    x += gaus
    return x, y

def get_batch(n=25):
    c = get_powder()
    s = get_signals(c, n)[:,:-1] # cut last step
    var, y = generate_variations(s, c.step_size)
    return var

def get_new_batch():
    st.session_state['batch'] = get_batch()
    return

def query(payload):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    res = requests.post(
        f"http://{MODEL_URL}/{MODEL_NAME}", data=payload, auth=(TS_USER, TS_PWD), headers=headers
    )
    if res.status_code == 503:
        time.sleep(2)
        return query(payload)
    return res


@st.cache_data(show_spinner=False)
def predict_batch(x):
    rows = x.shape[0]
    probs = np.zeros(rows)
    payload = {}
    for i in range(rows):
        data = x[i].tobytes()
        data_b64 = base64.b64encode(data).decode("utf8")
        payload[str(i)] = data_b64
    payload = json.dumps(payload)
    response = query(payload).json()
    probs = [response[i]["impurity"] for i in range(len(response))]
    return np.array(probs)

if 'batch' not in st.session_state:
    st.session_state['batch'] = get_batch()

st.sidebar.button("Get new batch", on_click=get_new_batch)

signal = st.session_state['batch'][0]
steps = np.linspace(20.,70., 2500, endpoint=False)

st.session_state["probs"] = predict_batch(st.session_state['batch'])#[[0]])

frame = st.empty()
label = st.empty()

all_options = ["Overview"]
all_options.extend([str(i+1) for i in range(25)])

sel = st.sidebar.selectbox("Inspect prediction:", all_options)

if sel != "Overview": # assume str(index)
    signal = st.session_state['batch'][int(sel)-1]
    prob = st.session_state["probs"][int(sel)-1]
    df = pd.DataFrame()
    df['steps'] = steps
    df['signal'] = signal
    fig = px.line(
        df,
        x="steps",
        y="signal",
    )
    frame.plotly_chart(fig)
    if prob < 0.5:
        label.text("Sample predicted pure with confidence "+str(round(1- prob, 4)*100))
    else:
        label.text("Sample predicted impure with confidence "+str(round(prob, 4)*100))
else: # Overview
    all_probs = st.session_state["probs"]
    if all_probs.size == 1:
        all_probs = np.repeat(all_probs, 25, axis=0)
    # Create a DataFrame with indices (1-25)
    df = pd.DataFrame(np.arange(1, 26).reshape(5, 5))
    # Define a function to apply color coding based on array values
    def color_coding(val):
        color = 'green' if all_probs[val-1] < 0.5 else 'red'
        return f'color: {color}'

    frame.dataframe(df.style.applymap(color_coding))
    label.text("Samples identified as impure are highlighted in red. \nSelect sample in sidebar to inspect in detail")
