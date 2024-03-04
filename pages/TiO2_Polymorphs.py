import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from io import StringIO
import requests

st.set_page_config(page_title="TiO2 Polymorphs", page_icon="🎯")

st.markdown("# TiO$_2$ Nanoparticle Classification")
st.write("Identify TiO$_2$ samples based on their powder XRD pattern")
st.sidebar.header("TiO$_2$ Classification Demo")

MODEL_NAME = "tio2_class"
MODEL_URL = st.secrets["model_url"]

RRUFF_IDs = [
    "R060277",
    "R070582",
    "R120013",
    "R120064",
    "R040049",
    "R050031",
    "R050417",
    "R060493",
    "R050363",
    "R050591",
    "R130225",
]

labels = {
    "R060277": "Anatase",
    "R070582": "Anatase",
    "R120013": "Anatase",
    "R120064": "Anatase",
    "R040049": "Rutile",
    "R050031": "Rutile",
    "R050417": "Rutile",
    "R060493": "Rutile",
    "R050363": "Brookite",
    "R050591": "Brookite",
    "R130225": "Brookite",
}


@st.cache_data
def get_rruff_data():
    # RRUFF Infos
    files = {
        "R060277": "Anatase__R060277-9__Powder__Xray_Data_XY_RAW__5487.txt",
        "R070582": "Anatase__R070582-9__Powder__Xray_Data_XY_RAW__9462.txt",
        "R120013": "Anatase__R120013-9__Powder__Xray_Data_XY_RAW__10993.txt",
        "R120064": "Anatase__R120064-9__Powder__Xray_Data_XY_RAW__11316.txt",
        "R040049": "Rutile__R040049-1__Powder__Xray_Data_XY_RAW__115.txt",
        "R050031": "Rutile__R050031-1__Powder__Xray_Data_XY_RAW__775.txt",
        "R050417": "Rutile__R050417-1__Powder__Xray_Data_XY_RAW__1724.txt",
        "R060493": "Rutile__R060493-1__Powder__Xray_Data_XY_RAW__5130.txt",
        "R050363": "Brookite__R050363-1__Powder__Xray_Data_XY_RAW__1549.txt",
        "R050591": "Brookite__R050591-1__Powder__Xray_Data_XY_RAW__1982.txt",
        "R130225": "Brookite__R130225-9__Powder__Xray_Data_XY_RAW__11921.txt",
    }
    root_url = "https://rruff.info/repository/sample_child_record_powder/by_minerals/"

    # Prepare Numpy array
    steps = np.linspace(5.0, 90.0, 8501, endpoint=True)
    signals = np.zeros([len(RRUFF_IDs), steps.size])

    # iteratively query files and populare signals array
    for i, rid in enumerate(RRUFF_IDs):
        url = root_url + files[rid]
        res = requests.get(url)
        meas = np.loadtxt(
            StringIO(res.content.decode("UTF-8")), comments="#", delimiter=","
        )
        signals[i] = np.interp(steps, meas[:, 0], meas[:, 1])
    return signals


@st.cache_data
def get_prediction(signal):
    data_bytes = signal.tobytes()
    res = requests.post(
        f"http://{MODEL_URL}/predictions/{MODEL_NAME}", files={"data": data_bytes}
    )
    return res.json()


option = st.sidebar.selectbox("Select measurement from RRUFF:", RRUFF_IDs)
st.sidebar.divider()
st.sidebar.write("## Info:")
st.sidebar.write(
    "TiO$_2$ samples can crystallize in different arrangements (see [wikipedia](https://en.wikipedia.org/wiki/Titanium_dioxide#Structure)). "\
    + "Methods including X-ray diffraction (XRD) allow to analyze the crystalline structure of samples. "\
    + "Therefore, analysis of the XRD patterns enables the identification of the underlying TiO$_2$ structure."
)

all_scans = get_rruff_data()

idx = RRUFF_IDs.index(option)
steps = np.linspace(5.0, 90.0, 8501, endpoint=True)
scan_df = pd.DataFrame()
scan_df["2 Theta"] = steps
scan_df["Count"] = all_scans[idx]

fig = px.line(
    scan_df,
    x="2 Theta",
    y="Count",
    title=f'XRD scan with ID {option} and label "{labels[option]}"',
)
st.plotly_chart(fig)

prediction = get_prediction(all_scans[idx])

df = pd.DataFrame.from_dict(prediction, orient="index", columns=["Prediction"])

st.write(
    f'The model predicted "{str.capitalize(df["Prediction"].idxmax())}" with a confidence of {np.format_float_positional(df["Prediction"].max()*100,2)}%'
)

st.divider()

st.write("Full predictions")

st.dataframe(df)
