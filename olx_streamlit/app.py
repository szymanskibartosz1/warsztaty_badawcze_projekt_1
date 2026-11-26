import streamlit as st


st.set_page_config(page_title="Asystent ogłoszeń i wyceny", page_icon="🏠", layout="wide")

navigation = st.navigation(
    [
        st.Page("listing_page.py", title="Generator ogłoszeń", icon="📷", default=True),
        st.Page("valuation_page.py", title="Wycena nieruchomości", icon="🏠"),
    ]
)
navigation.run()
