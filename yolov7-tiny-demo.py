# To run use
# $ streamlit run yolor_streamlit_demo.py

from yolo_v7 import names, load_yolov7_and_process_each_frame

import tempfile
import cv2

from models.models import *
from utils.datasets import *
from utils.general import *
import streamlit as st


def main():
    
    #title
    st.title('Streamlit dashboard YOLOv7')
    
    #side bar title
    st.sidebar.title('Configuracion')
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    use_webcam = st.sidebar.checkbox('Activar Webcam')

    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confianza',min_value=0.0, max_value=1.0, value = 0.25)
    st.sidebar.markdown('---')

    save_img = st.sidebar.checkbox('Guardar Video')
    enable_GPU = st.sidebar.checkbox('Activar GPU')

    custom_classes = st.sidebar.checkbox('Usar clases')
    assigned_class_id = []
    if custom_classes:
        assigned_class = st.sidebar.multiselect('Seleccionar clases',list(names),default='person')
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
        
    video_file_buffer = st.sidebar.file_uploader('Subir un video', type=['mp4','mov','avi','asf','m4v'])
    DEMO_VIDEO = 'escaleras_Trim4.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0,cv2.CAP_ARAVIS)
            tffile.name = 0
        else:
            vid =cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
            dem_vid = open(tffile.name, 'rb')
            demo_bytes = dem_vid.read()

            st.sidebar.text('Video de entrada')
            st.sidebar.video(demo_bytes)
    else:
        tffile.write(video_file_buffer.read()) 
        dem_vid = open(tffile.name, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('input video')
        st.sidebar.video(demo_bytes)
    
    print(tffile.name)
    stframe = st.empty()
    st.markdown("<hr/>", unsafe_allow_html=True)
    kpi1,kpi2,kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Traked Objects**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown("**Total Count**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    load_yolov7_and_process_each_frame('yolov7-escaleras', tffile.name,enable_GPU,save_img,confidence,assigned_class_id,kpi1_text,kpi2_text,kpi3_text,stframe)

        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

