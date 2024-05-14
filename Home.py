import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="⛺",
)

st.header("""AI-based Spectra Analysis""")
# st.write("### based on their Characteristic Spectra")

st.markdown("Spectroscopic data encompasses a diverse range of analytical techniques utilized across scientific disciplines. These techniques generate signals based on interactions between matter and electromagnetic radiation, spanning from ultraviolet to infrared wavelengths. The characteristics of spectroscopic signals, such as intensity, frequency, and wavelength, offer invaluable insights into the structure and composition of substances under study.")

st.markdown("Analyzing spectroscopic signals presents various challenges, including noise, baseline drift, and overlapping peaks, which can obscure meaningful information. Despite these obstacles, spectroscopic techniques are frequently integrated into high-throughput experimentation pipelines, where the rapid screening and analysis of vast datasets are essential. To cope with the volume and complexity of spectroscopic data, researchers increasingly turn to advanced computational approaches, such as deep learning.")

st.markdown("So far, researchers have successfully applied neural networks to tackle challenges in the interpretation of [Raman spectra](https://www.nature.com/articles/s41467-019-12898-9), [NIR spectra](https://www.sciencedirect.com/science/article/pii/S0023643821016091), [NMR spectra](https://www.nature.com/articles/s41467-022-33879-5), and [XRD patterns](https://journals.iucr.org/m/issues/2021/03/00/fc5051/index.html), among others, showcasing their potential across various spectroscopic domains.")

st.markdown("Explore how neural networks can be efficiently integrated for the automated analysis of spectroscopic data through this web application.")

st.sidebar.success("Select a demo above.")

st.divider()

st.markdown(
    """
    👈 Select a demo from the sidebar
    """
)

st.sidebar.divider()
st.sidebar.markdown("Made by Jan Schuetzke")
col1, col2, col3, col4 = st.sidebar.columns(4)
col1.markdown("[![LinkedIn-Icon](https://img.icons8.com/ios-filled/50/linkedin.png)](https://www.linkedin.com/in/jan-schuetzke/)")
col2.markdown("[![Github-Icon](https://img.icons8.com/ios-filled/50/github.png)](https://github.com/jschuetzke)")
col3.markdown("[![GScholar-Icon](https://img.icons8.com/ios/50/google-scholar--v2.png)](https://scholar.google.com/citations?user=WI2xAokAAAAJ&hl=en)")
col4.markdown("[![Website-Icon](https://img.icons8.com/ios/50/domain--v1.png)](https://jschuetzke.github.io)")
st.sidebar.markdown("Icons by [Icon8](https://icons8.com)")
