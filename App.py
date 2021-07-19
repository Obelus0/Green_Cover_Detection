# Core Pkgs
import ee
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from folium import plugins
from streamlit_folium import folium_static
import folium
import os
import io
from PIL import Image
import time
from selenium import webdriver

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=True,
        control=True
    ),
    'Google Satellite': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Google Terrain': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Terrain',
        overlay=True,
        control=True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=True,
        control=True
    )
}


def Green_Cover_Detection(img):
    img = np.array(img)
    # boundary conditions for green color H,S,V (Can be tweaked if required)
    lowerBound = np.array([36, 60, 40])
    upperBound = np.array([102, 255, 255])

    # image processing for easy segmentation
    img_resize = cv2.resize(img, (340, 220))
    img_blur = cv2.GaussianBlur(img_resize, (3, 3), cv2.BORDER_DEFAULT)

    # Colour segmentation is easily achievable in HSV domain
    imgHSV = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # create a filter or mask to filer out a specific color here we filter green color
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # Finding and drawing the largest region of green cover in image
    conts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(conts) != 0:
        # find the biggest countour (c) by the area
        c = max(conts, key=cv2.contourArea)

        # draw in green the largest contour
        cv2.drawContours(img_resize, c, -1, (0, 255, 0), 2)

    percent = mask.mean() * 100 / 255

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img_resize = cv2.resize(img_resize, (img.shape[1], img.shape[0]))

    return mask, img_resize, percent


def main():
    """Green Cover Detection App"""
    st.title("Green Cover Impact")

    activities = ["Detection", "Data", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Detection':
        st.subheader("Green Cover Detection")

        options = ["Upload Image", "Map"]
        selection = st.selectbox("Select Option", options)

        if selection == 'Upload Image':
            image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

            if image_file is not None:
                our_image = Image.open(image_file)
                st.text("Original Image")
                # st.write(type(our_image))
                st.image(our_image)

                green_img, contour_img, percentage = Green_Cover_Detection(our_image)

                st.text("Green Mask")
                st.image(green_img)

                st.text("Contour of Maximal Green Cover")
                st.image(contour_img)

                st.text("Percentage of Green Cover")
                st.write(percentage)

        elif selection == 'Map':

            lat = st.number_input('Enter latitude', min_value=-90.00, max_value=90.00, value=0.00, step=0.0001,
                                  format="%.5f")
            long = st.number_input('Enter longitude', min_value=-180.00, max_value=180.00, value=0.00, step=0.0001,
                                   format="%.5f")

            # Define a method for displaying Earth Engine image tiles on a folium map.
            def add_ee_layer(self, ee_object, vis_params, name):

                try:
                    # display ee.Image()
                    if isinstance(ee_object, ee.image.Image):
                        map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                        folium.raster_layers.TileLayer(
                            tiles=map_id_dict['tile_fetcher'].url_format,
                            attr='Google Earth Engine',
                            name=name,
                            overlay=True,
                            control=True
                        ).add_to(self)
                    # display ee.ImageCollection()
                    elif isinstance(ee_object, ee.imagecollection.ImageCollection):
                        ee_object_new = ee_object.mosaic()
                        map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                        folium.raster_layers.TileLayer(
                            tiles=map_id_dict['tile_fetcher'].url_format,
                            attr='Google Earth Engine',
                            name=name,
                            overlay=True,
                            control=True
                        ).add_to(self)
                    # display ee.Geometry()
                    elif isinstance(ee_object, ee.geometry.Geometry):
                        folium.GeoJson(
                            data=ee.Date(0),
                            name=name,
                            overlay=True,
                            control=True
                        ).add_to(self)
                    # display ee.FeatureCollection()
                    elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
                        ee_object_new = ee.Image().paint(ee_object, 0, 2)
                        map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                        folium.raster_layers.TileLayer(
                            tiles=map_id_dict['tile_fetcher'].url_format,
                            attr='Google Earth Engine',
                            name=name,
                            overlay=True,
                            control=True
                        ).add_to(self)

                except:
                    print("Could not display {}".format(name))

            # Add EE drawing method to folium.
            folium.Map.add_ee_layer = add_ee_layer

            if lat and long is not None:
                # Set visualization parameters.
                vis_params = {
                    'min': 0,
                    'max': 4000,
                    'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

                from branca.element import Figure
                fig = Figure(width=600, height=400)

                # Create a folium map object.
                my_map = folium.Map(location=[lat, long], zoom_start=16, height=500)

                fig.add_child(my_map)

                # Add custom basemaps
                basemaps['Google Terrain'].add_to(my_map)
                basemaps['Google Satellite'].add_to(my_map)

                # Add a layer control panel to the map.
                my_map.add_child(folium.LayerControl())

                # Add fullscreen button
                plugins.Fullscreen().add_to(my_map)

                folium_static(my_map)

                if st.button('Detect Green Cover'):
                    # Save the map as an HTML file
                    fn = 'testmap.html'
                    tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
                    my_map.save(fn)
                    #
                    # # Open a browser window...
                    browser = webdriver.Firefox()
                    # # ..that displays the map...
                    browser.get(tmpurl)
                    # # Grab the screenshot
                    browser.save_screenshot('map.png')
                    # # Close the browser
                    browser.quit()
                    #
                    img = cv2.imread('map.png')
                    crop_img = img[0:490, 300:990]
                    final_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                    # img_data = my_map._to_png()
                    # img = Image.open(io.BytesIO(img_data))

                    green_img, contour_img, percentage = Green_Cover_Detection(final_img)

                    st.text("Original Image")
                    st.image(final_img)

                    st.text("Green Mask")
                    st.image(green_img)

                    st.text("Contour of Maximal Green Cover")
                    st.image(contour_img)

                    st.text("Percentage of Green Cover")
                    st.write(percentage)

    elif choice == 'About':
        st.subheader("About Green Cover Detection App")


if __name__ == '__main__':
    main()
