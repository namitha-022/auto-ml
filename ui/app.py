import streamlit as st
import requests
import tempfile
import os

st.set_page_config(page_title="AutoML Performance Profiler")

st.title("AutoML Performance Profiler")
st.write("Upload a PyTorch model to analyze its performance.")

uploaded_file = st.file_uploader(
    "Upload PyTorch model (.pt)",
    type=["pt"]
)

# Optional input shape (important for ONNX later)
input_shape = st.text_input(
    "Input shape (comma separated, e.g. 1,3,224,224)",
    value="1,3,224,224"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(uploaded_file.read())
        model_path = tmp.name

    st.success("Model uploaded successfully")

    if st.button("Send to MCP Server"):
        response = requests.post(
            "http://localhost:8000/upload-model",
            files={"model": open(model_path, "rb")},
            data={"input_shape": input_shape}
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Model Context")
            st.json({
                "model_name": result["model_name"],
                "params": result["params"],
                "input_shape": result["input_shape"]
            })

            st.subheader("ONNX Conversion Status")

            if result.get("onnx_export"):
                st.success("ONNX export successful")
            else:
                st.warning("ONNX export failed")
                if result.get("onnx_message"):
                    st.code(result["onnx_message"])
        else:
            st.error("Failed to process model")

    os.remove(model_path)

st.subheader("⚡ Run Performance Profiling")

if st.button("Run Profiling"):
    with st.spinner("Profiling model..."):
        response = requests.post(
            "http://localhost:8000/run-profile"
        )

    if response.status_code == 200:
        profiling_data = response.json()["profiling_results"]
        st.subheader("Profiling Results")
        st.json(profiling_data)
    else:
        st.error("Profiling failed")
