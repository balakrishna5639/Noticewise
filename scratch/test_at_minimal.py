from streamlit.testing.v1 import AppTest
at = AppTest.from_string("import streamlit as st; st.title('Hello')")
at.run()
print("Success! Title:", at.title[0].value)
