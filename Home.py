import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="⛺",
)

st.header("""AI-based Spectra Analysis""")
# st.write("### based on their Characteristic Spectra")

st.text("Spectroscopic data encompasses a diverse range of analytical techniques utilized across scientific disciplines. These techniques generate signals based on interactions between matter and electromagnetic radiation, spanning from ultraviolet to infrared wavelengths. The characteristics of spectroscopic signals, such as intensity, frequency, and wavelength, offer invaluable insights into the structure and composition of substances under study.")

st.text("Analyzing spectroscopic signals presents various challenges, including noise, baseline drift, and overlapping peaks, which can obscure meaningful information. Moreover, interpreting complex spectra often requires expertise in both the underlying physics and statistical methods. Despite these obstacles, spectroscopic techniques are frequently integrated into high-throughput experimentation pipelines, enabling rapid screening and analysis of vast datasets. To cope with the volume and complexity of spectroscopic data, researchers increasingly turn to advanced computational approaches, such as deep learning. Leveraging neural networks, deep learning algorithms can automatically extract relevant features from spectra, streamline analysis workflows, and uncover hidden patterns, thereby enhancing the efficiency and accuracy of spectroscopic data interpretation.")

st.sidebar.success("Select a demo above.")

st.divider()

st.markdown(
    """
    👈 Select a demo from the sidebar
    """
)