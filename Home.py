import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="⛺",
)

st.header("""Substance Identification based on their characteristic spectrum""")
# st.write("### based on their Characteristic Spectra")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    👈 Select a demo from the sidebar
    """
)

st.divider()
st.write("Various demos that utilize neural networks for automated analysis of spectroscopic data.")