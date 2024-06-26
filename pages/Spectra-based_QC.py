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

st.set_page_config(page_title="Spectra-based QC", page_icon="🔬")

st.title("Nanoparticle Quality Control")
st.markdown("In a theoretical high-throughput pipeline for producing titanium oxide nanoparticles (rutile structure), an XRD instrument is employed to analyze the resulting samples. A neural network is applied to identify those samples that do not align with the QC guidelines based on the corresponding XRD pattern.")
st.sidebar.header("Rutile QC Demo")
st.sidebar.markdown("Realistic XRD patterns are simulated to demonstrate the convenience of applying neural networks in high-throughput environemnts.")

MODEL_NAME = "rutile"
MODEL_URL = st.secrets["model_url"]

TS_USER = st.secrets["ts_user"]
TS_PWD = st.secrets["ts_password"]

PROPERTIES = ["impurity", "cryst_size"]
PROP_LABELS = {
    "impurity": "Purity",
    "cryst_size": "Crystallite size"
}

@st.cache_data
def get_powder():
    url = "http://www.crystallography.net/cod/9007432.cif"
    res = requests.get(url)
    struct = Structure.from_str(res.content.decode("UTF-8"), fmt="cif")
    ideal_pow = Powder(struct, (20,70), 0.02)
    ideal_pow._domain_size = 0
    ideal_pattern = ideal_pow.get_signal()[:-1] # st.session_state['ideal']
    powder = Powder(
        struct,
        two_theta=(20,70),
        step_size=0.02,
        max_strain=0.01,
        vary_strain=True,
    )
    powder._domain_size = 0
    return powder, ideal_pattern

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
    detection_threshold = 0.05,
    num_multi_peaks = 2,
    restricted_area = 25,
    fwhm_range = (0.15, 0.5),
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
    # eta = rng.uniform(0.1, 0.9, size=num_scans)
    for n in np.arange(num_scans):
        # kernel = peak_shapes.get_pseudo_voigt_kernel(fwhm[n], step_size, eta[n])
        kernel = peak_shapes.get_gaussian_kernel(fwhm[n], step_size)
        x[n] = convolve1d(x[n], kernel, mode="constant")

    x = scale_min_max(x)

    y2 = fwhm >= 0.425
    y2 = y2.astype(int)

    # correct absolute intensities due to peak broadening
    x *= np.sqrt(np.maximum(1 - fwhm, 0.3))[:, None]
    # x *= np.maximum(1 - fwhm, 0.35)[:, None]
    x = generate_noise(
        x,
        noise_lvl=0.
    )
    # add noise, clip extreme values
    gaus = 1 / 3 * np.clip(rng.normal(0, 1, x.shape), -3, 3)
    gaus = (gaus * 0.5) + 0.5
    # scale noise
    gaus *= noise_lvls[:, None]
    x += gaus

    target = np.zeros([y.size, 2])
    target[:,0] = y
    target[:,1] = y2
    return x, target

def get_batch(n=25):
    c, st.session_state['ideal'] = get_powder()
    s = get_signals(c, n)[:,:-1] # cut last step
    return generate_variations(s, c.step_size)

def get_data():
    batch, labels = get_batch()
    st.session_state["batch"] = batch
    st.session_state["labels"] = labels
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
    preds = np.zeros([len(response), len(PROPERTIES)])
    for i, key in enumerate(PROPERTIES):
        probs = [response[j][key] for j in range(len(response))]
        preds[:,i] = np.array(probs)
    return preds

if 'batch' not in st.session_state:
    get_data()

st.sidebar.button("Get new batch", on_click=get_data)

signal = st.session_state['batch'][0]
steps = np.linspace(20.,70., 2500, endpoint=False)

st.session_state["probs"] = predict_batch(st.session_state['batch'])#[[0]])

frame = st.empty()
label = st.empty()

all_options = ["Overview"]
all_options.extend([str(i+1) for i in range(25)])

sel = st.sidebar.selectbox("Inspect prediction:", all_options)

if sel != "Overview":  # assume str(index)
    signal = st.session_state["batch"][int(sel) - 1]
    prob = 1 - st.session_state["probs"][int(sel) - 1]
    df = pd.DataFrame()
    df["steps"] = steps
    df["Measured"] = signal
    df["Ideal-Rutile"] = scale_min_max(st.session_state['ideal'])
    df.set_index("steps", drop=True, inplace=True)
    fig = px.line(
        df,
        x=df.index,
        y=df.columns,
    )
    frame.plotly_chart(fig)
    properties = pd.DataFrame()
    properties["Property"] = ""
    for i, key in enumerate(PROPERTIES):
        properties.loc[i, "Property"] = PROP_LABELS[key]
    properties["Metric"] = np.round(prob, 4) * 100
    label.dataframe(
        properties,
        column_config={
        "Metric": st.column_config.ProgressColumn(
            "Metric",
            help="Compliance with QC",
            format="%.2f%%",
            min_value=0,
            max_value=100,
            ),
        },
        hide_index=True,
        use_container_width=True
    )
else:  # Overview
    all_probs = st.session_state["probs"]
    if all_probs.size == 1:
        all_probs = np.repeat(all_probs, 25, axis=0)
    else:
        # Create a DataFrame with indices (1-25)
        df = pd.DataFrame(np.arange(1, 26).reshape(5, 5))

    # Define a function to apply color coding based on array values
    def color_coding(val):
        color = "green" if np.max(all_probs[val - 1]) < 0.5 else "red"
        return f"color: {color}"

    frame.dataframe(df.style.applymap(color_coding))
    label.text(
        "Samples not compliant with QC guidelines are highlighted in red. \nSelect sample in sidebar to inspect in detail"
    )

st.sidebar.divider()
st.sidebar.write("## Info:")
st.sidebar.markdown(
    "In high-throughput production, the objective is to fabricate a defined substance in large quantity. "\
    + "Spectroscopic techniques enable destruction-free analysis of the resulting samples. "\
    + "Numerous irregularities may arise during substance production, such as the presence of impurities or mismatches in relevant properties. "\
    + "Neural network-based analysis of the raw measurements provides a rapid assessment of the produced samples."
)

st.sidebar.divider()
st.sidebar.markdown("Made by Jan Schuetzke")
col1, col2, col3, col4 = st.sidebar.columns(4)
col1.markdown("[![LinkedIn-Icon](https://img.icons8.com/ios-filled/50/linkedin.png)](https://www.linkedin.com/in/jan-schuetzke/)")
col2.markdown("[![Github-Icon](https://img.icons8.com/ios-filled/50/github.png)](https://github.com/jschuetzke)")
col3.markdown("[![GScholar-Icon](https://img.icons8.com/ios/50/google-scholar--v2.png)](https://scholar.google.com/citations?user=WI2xAokAAAAJ&hl=en)")
col4.markdown("[![Website-Icon](https://img.icons8.com/ios/50/domain--v1.png)](https://jschuetzke.github.io)")
st.sidebar.markdown("Icons by [Icon8](https://icons8.com)")
