"""
    Green Cover Impact & Awareness App
    Green Cover Detection
    Plant Trees Near Me
    Impact Statistics in Tamilnadu
    (Dataset provided taken from Government)
"""
import ee
import streamlit as st
import cv2
import numpy as np
from folium import plugins
from matplotlib import pyplot as plt
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import folium
import os
from PIL import Image
from selenium import webdriver
from geopy.geocoders import Nominatim
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import datetime
from dateutil.relativedelta import relativedelta

# Comment this out after Authenticating the earth engine 
ee.Authenticate()

# Initialising the earth engine
ee.Initialize()

# Wide Configuration of Streamlit
st.set_page_config(layout="wide")

# Used to convert a given region into lat-long
geo_url = 'http://maps.googleapis.com/maps/api/geocode/json'

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


def NDVI_Green_Cover_Detection(img):
    """
    Calculates the Green mask for NDVI (Normalised Difference Vegetation Index) images
    :param
                img : image to obtain mask of (NDVI image)

    :return
                mask : image - Green mask
                img_conts : image - Largest contour of Green Cover drawn on Original Image
                percent : floating point - Percentage of Green Cover in Image

    """
    # boundary conditions for green color H,S,V in NDVI image (Can be tweaked if required)
    lowerBound = np.array([36, 25, 25])
    upperBound = np.array([70, 255, 255])

    # image processing for easy segmentation
    img_resize = cv2.resize(img, (520, 705))

    # Colour segmentation is easily achievable in HSV domain
    imgHSV = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)

    # create a filter or mask to filer out a specific color here we filter green color
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # Finding and drawing the largest region of green cover in image
    conts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(conts) != 0:
        # find the biggest contour (c) by the area
        c = max(conts, key=cv2.contourArea)

        # draw in green the largest contour on img_resize
        cv2.drawContours(img_resize, c, -1, (0, 255, 0), 2)

    # percentage of green cover
    percent = mask.mean() * 100 / 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Since we drew the contour on img_resize and we want it back in the orginal shape
    img_conts = cv2.resize(img_resize, (img.shape[1], img.shape[0]))
    img_conts = cv2.cvtColor(img_conts, cv2.COLOR_BGR2RGB)

    return mask, img_conts, percent


def Green_Cover_Detection(img):
    """
    Calculates the Green Mask, Segmented Image & provides percentage of Green Cover
    :param
                img: Input Image
    :return:
                mask: Green Mask
                diff: The difference between Green Mask & Original Image
                segmented_image: Segmented image in HSV Space
                percent: Percentage of Green Cover in Image
    """

    # boundary conditions of HSV image for segmented image
    lowerBound = np.array([18, 25, 25])
    upperBound = np.array([95, 255, 255])

    # image processing for easy segmentation
    N = cv2.resize(img, (705, 520))
    N = cv2.GaussianBlur(N, (7, 7), 1)
    Z = cv2.cvtColor(N, cv2.COLOR_BGR2HSV)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = Z.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    # convert back to 8 bit values
    centers = np.uint8(centers)
    label = list(labels)

    # flatten the labels array
    labels = labels.flatten()

    # If you want the percentage of each segment
    # percent = []
    # for i in range(len(centers)):
    #     j = label.count(i)
    #     j = j / (len(label))
    #     percent.append(j)

    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(Z.shape)

    # create a filter or mask to filer out a specific color here we filter green color
    mask = cv2.inRange(segmented_image, lowerBound, upperBound)

    # Take only region with out green cover
    img2_fg = cv2.bitwise_and(N, N, mask=mask)
    diff = N - img2_fg
    percent = mask.mean() * 100 / 255

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    diff = cv2.resize(diff, (img.shape[1], img.shape[0]))
    segmented_image = cv2.resize(segmented_image, (img.shape[1], img.shape[0]))

    return mask, diff, segmented_image, percent


def add_ee_layer(self, ee_image_object, vis_params, name):
    """
    Adds a method for displaying Earth Engine image tiles to folium map.

    """

    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)


def Map():
    """
    Renders an interactive folium map based on the input parameters

    """
    # Add Earth Engine drawing method to folium.
    folium.Map.add_ee_layer = add_ee_layer

    # Can choose between Lat-Long, or just typing the address
    options = ["Lat-Long", "Address"]
    selection = st.selectbox("Select Option", options)

    if selection == 'Address':
        address = st.text_input("Enter Location Region, Street Name or Postal Code")
        geolocator = Nominatim(user_agent="Green_Cover_Detection")
        location = geolocator.geocode(address)

        if location is None:
            st.text('The region cannot be identified, please enter valid Region, Street Name or Postal Code')
            return

        lat = location.latitude
        long = location.longitude

    elif selection == 'Lat-Long':
        # Input Latitude or Longitude
        lat = st.number_input('Enter latitude', min_value=-90.00, max_value=90.00, value=0.00, step=0.0001,
                              format="%.5f")
        long = st.number_input('Enter longitude', min_value=-180.00, max_value=180.00, value=0.00, step=0.0001,
                               format="%.5f")

    if lat and long is None:
        st.text("Enter Valid Latitude & Longitude (N,E is positive & S,W is negative)")
        return

    else:
        zoom = st.number_input('Enter Zoom, Default Value - 15 (Region of 5 sq km): Can modify based on the '
                               'interactive '
                               'folium map', min_value=1, max_value=18, value=15)

        # Options for Satellite image comparison
        options = ["Time-Series Comparison", "Select Date"]
        selection = st.selectbox("Select Option", options)

        if selection == 'Time-Series Comparison':
            min_date = st.number_input('Enter Initial Year', max_value=2019, min_value=1999,
                                       value=2000)
            max_date = st.number_input('Enter Final Year', max_value=2020, min_value=2000,
                                       value=2020)

            # Converting year into year range
            if max_date > min_date:
                max_date_time = str(max_date) + "-01-01"
                max_date_time_end = str(max_date + 1) + "-01-01"

                min_date_time = str(min_date) + "-01-01"
                min_date_time_end = str(min_date + 1) + "-01-01"

            else:
                st.text('Enter Final Year greater than Initial Year')

            # Get raster data from Landsat Satellite
            max_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_8DAY_NDVI').filterDate(max_date_time,
                                                                                            max_date_time_end)
            min_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_8DAY_NDVI').filterDate(min_date_time,
                                                                                            min_date_time_end)

            # Taking a mean of the image collection
            max_NDVI_img = max_collection.mean()
            min_NDVI_img = min_collection.mean()

            # Set visualization parameters for NDVI land cover
            NDVI_vis_params = {'min': 0,
                               'max': 1, 'bands': ['NDVI'],
                               'palette': [
                                   'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
                                   '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
                                   '012E01', '011D01', '011301'
                               ],
                               }

            my_map_initial = folium.Map(location=[lat, long], zoom_start=zoom)

            # Adding the Google Satellite Map high resolution recent image
            basemaps['Google Satellite'].add_to(my_map_initial)

            # Add the NDVI land cover to the map object.
            my_map_initial.add_ee_layer(min_NDVI_img, NDVI_vis_params, 'NDVI')

            # Add a layer control panel to the map.
            my_map_initial.add_child(folium.LayerControl())

            # Perform the same for the other year
            my_map_final = folium.Map(location=[lat, long], zoom_start=zoom)
            basemaps['Google Satellite'].add_to(my_map_final)
            my_map_final.add_ee_layer(max_NDVI_img, NDVI_vis_params, 'NDVI')
            my_map_final.add_child(folium.LayerControl())

            st.text("Left - Initial Year, Right - Final Year")
            col1, mid, col2 = st.columns([1, 20, 20])
            with col1:
                folium_static(my_map_initial, width=650)
            with col2:
                folium_static(my_map_final, width=650)

            if st.button('Compare Green Cover'):
                """
                Method in order to convert the folium map to an image using selenium
                
                """
                # Save the map as an HTML file
                fn = 'testmap.html'
                tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
                my_map_initial.save(fn)

                # Open a browser window...
                browser = webdriver.Firefox()
                # ..that displays the map...
                browser.get(tmpurl)
                time.sleep(0.1)
                # Grab the screenshot
                browser.save_screenshot('map_initial.png')
                # Close the browser
                browser.quit()

                # Cropping the region of the required image
                img_initial = cv2.imread('map_initial.png')
                crop_img_initial = img_initial[180:700, 295:1000]
                final_img_initial = cv2.cvtColor(crop_img_initial, cv2.COLOR_BGR2RGB)

                green_img_initial, contour_img_initial, percentage_initial = NDVI_Green_Cover_Detection(crop_img_initial)

                st.text("Initial Year")
                st.image(contour_img_initial, width=650)

                st.text("Percentage of Green Cover - Initial Year")
                st.write(percentage_initial)
                
                # Performing the same for the other map
                fn1 = 'testmap1.html'
                tmpurl1 = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn1)
                my_map_final.save(fn1)
                browser = webdriver.Firefox()
                browser.get(tmpurl1)
                time.sleep(0.5)
                browser.save_screenshot('map_final.png')
                browser.quit()

                img1 = cv2.imread('map_final.png')
                crop_img_final = img1[180:700, 295:1000]
                final_img_final = cv2.cvtColor(crop_img_final, cv2.COLOR_BGR2RGB)

                green_img_final, contour_img_final, percentage_final = NDVI_Green_Cover_Detection(crop_img_final)

                st.text("Final Year")
                st.image(contour_img_final, width=650)

                st.text("Percentage of Green Cover - Final Year")
                st.write(percentage_final)

                st.text("Percentage Decrease")
                st.write((percentage_initial - percentage_final) / percentage_initial * 100)

    if selection == 'Select Date':

        date = st.date_input('Enter Date', max_value=datetime.date(2021, 1, 1), min_value=datetime.date(1999, 1, 1),
                             value=datetime.date(2020, 1, 1))
        
        date_time = date.strftime("%Y-%d-%m")
        date_time_end = date + relativedelta(months=12)
        date_time_end = date_time_end.strftime("%Y-%d-%m")

        collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_8DAY_NDVI').filterDate(date_time, date_time_end)

        # Provides Landcover classification for the year 2019
        collection2 = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")

        # Select a specific band and dates for land cover.
        NDVI_img = collection.median()

        LandCover_img = collection2.select('discrete_classification')

        if lat and long is not None:
            
            my_map = folium.Map(location=[lat, long], zoom_start=zoom)
            basemaps['Google Satellite'].add_to(my_map)
            my_map.add_ee_layer(LandCover_img, {}, 'LandCover')

            # Add the land cover to the map object.
            my_map.add_ee_layer(NDVI_img, NDVI_vis_params, 'NDVI')

            # Add a layer control panel to the map.
            my_map.add_child(folium.LayerControl())

            folium_static(my_map)

            if st.button('Detect Green Cover'):
                fn = 'testmap.html'
                tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
                my_map.save(fn)

                browser = webdriver.Firefox()
                browser.get(tmpurl)
                time.sleep(0.1)
                browser.save_screenshot('map.png')
                browser.quit()

                img = cv2.imread('map.png')
                crop_img = img[180:700, 295:1000]
                final_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                green_img, contour_img, percentage = NDVI_Green_Cover_Detection(crop_img)

                st.text("Original Image")
                st.image(final_img)

                st.text("Green Mask")
                st.image(green_img)

                st.text("Contour of Maximal Green Cover")
                st.image(contour_img)

                st.text("Percentage of Green Cover")
                st.write(percentage)

def Plant_Trees_Near_Me():
    
        st.title('Region to plant trees')

        options = ["Lat-Long", "Address"]
        selection = st.selectbox("Select Option", options)

        if selection == 'Address':
            address = st.text_input("Enter Location Region, Street Name or Postal Code")

            geolocator = Nominatim(user_agent="Green_Cover_Detection")

            location = geolocator.geocode(address)

            if location is None:
                st.text('The region cannot be identified, please enter valid Region, Street Name or Postal Code')
                return

            lat = location.latitude

            long = location.longitude

        elif selection == 'Lat-Long':
            lat = st.number_input('Enter latitude', min_value=-90.00, max_value=90.00, value=0.00, step=0.0001,
                                  format="%.5f")
            long = st.number_input('Enter longitude', min_value=-180.00, max_value=180.00, value=0.00, step=0.0001,
                                   format="%.5f")

        Radius = st.number_input('Km radius around given region to search for suitable location - (Warning large '
                                 'radius will take longer time to process)', min_value=1,
                                 max_value=5, value=1)

        if st.button('Find Regions'):
            folium.Map.add_ee_layer = add_ee_layer
            
            #To maintain the processing time for the area covered we change the zoom
            if Radius <= 2:
                zoom = 18
                increment = 0.007
            else:
                zoom = 17
                increment = 0.015
                
            # Here 0.01 of a degree is considered approximately a kilometer of distance
            lat_boundary = lat + 0.01 * Radius
            long_boundary = long - 0.01 * Radius
            Region_Counter = 0
            Potential_Region_Counter = 0
            if lat and long is not None:
                lat = lat - 0.01 * Radius

                while lat < lat_boundary:
                    long = long + 0.01 * Radius
                    while long > long_boundary:
                        
                        my_map = folium.Map(location=[lat, long], zoom_start=zoom)
                        basemaps['Google Satellite'].add_to(my_map)
                        
                        fn = 'testmap.html'
                        tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
                        my_map.save(fn)
                        browser = webdriver.Firefox()
                        browser.get(tmpurl)
                        time.sleep(0.4)
                        browser.save_screenshot('map' + str(Region_Counter) + '.png')
                        browser.quit()
                        
                        img = cv2.imread('map' + str(Region_Counter) + '.png')
                        crop_img = img[180:700, 295:1000]
                        mask, diff, segmented_image, percent = Green_Cover_Detection(crop_img)

                        # Threshold can be changed based on the green cover allowed in a region
                        if percent < 40:
                            Potential_Region_Counter = Potential_Region_Counter + 1
                            st.write('Potential Region ' + str(Potential_Region_Counter))
                            st.image(crop_img)
                           
                            if percent != 0.0:
                                st.write(
                                    str(percent) + ' percent green cover for image at latitude & longitude ' + str(
                                        lat) + ',' + str(
                                        long))
                                
                            # When green cover is minimal then K-means results in a mask of 0.0 even if slight vegetation is present
                            else:
                                st.write('The region at '+ str(
                                        lat) + ',' + str(
                                        long) + ' has very minimal green cover')

                        Region_Counter = Region_Counter + 1
                        long = long - increment
                        # To stimulate loading
                        st.text('>' * Region_Counter)
                    lat = lat + increment

                if Potential_Region_Counter == 0:
                    st.write('Congratulations you live in region of a lot of greenary maybe you can go to the regions '
                             'which we have identified that need trees')
                else:
                    st.text('The potential regions have been highlighted above')



def main():
    """
    Sets up the Streamlit app interface and call other functions based on user input

    """
    st.title("Green Cover Impact Analysis in Tamilnadu")

    activities = ["Green Cover Detection", "Plant Trees Near Me", "Data Visualisation", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Green Cover Detection':
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
                img_array = np.array(our_image)

                green_img, diff_img, segmented_img, percentage = Green_Cover_Detection(img_array)

                st.text("Segmentation of image")
                st.image(segmented_img)

                st.text("Green Mask")
                st.image(green_img)

                st.text("Area without vegetation")
                st.image(diff_img)

                st.text("Percentage of Green Cover")
                st.write(percentage)

        elif selection == 'Map':
            Map()

    elif choice == 'Plant Trees Near Me':
        Plant_Trees_Near_Me()

    elif choice == 'Data Visualisation':

        st.title('Comparison of the environment over 10 years')

        col1, col2 = st.columns(2)

        selection = col1.radio('',('Groundwater','Rainfall'))

        df1 = pd.read_excel(selection.lower() + '_2011.xlsx')
        df2 = pd.read_excel(selection.lower() + '_2016.xlsx')
        df3 = pd.read_excel(selection.lower() + '_2020.xlsx')

        district = col2.selectbox('Choose District', sorted(list(df1['DISTRICT'])))

        df_list = [df[df['DISTRICT'] == district] for df in [df1, df2, df3]]
        plot_df = pd.concat(df_list, axis=0)
        plot_df.index = ['2011', '2016', '2020']
        plot_df = plot_df.T[1:]
        # st.table(plot_df)

        colors = px.colors.qualitative.Plotly
        fig = go.Figure()
        fig.add_traces(
            go.Scatter(x=plot_df.index, y=plot_df['2011'], mode='lines+markers', line=dict(color=colors[0], width=3),
                       name='2011'))
        fig.add_traces(
            go.Scatter(x=plot_df.index, y=plot_df['2016'], mode='lines+markers', line=dict(color=colors[1], width=3),
                       name='2016'))
        fig.add_traces(
            go.Scatter(x=plot_df.index, y=plot_df['2020'], mode='lines+markers', line=dict(color=colors[2], width=3),
                       name='2020'))
        fig.update_layout(paper_bgcolor='rgba(50,50,50,0.5)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)
        st.write('Rainfall in mm')
        st.write('Groundwater in Meters to Groundwater')

    elif choice == 'About':
        st.subheader("About Green Cover Detection App")


if __name__ == '__main__':
    main()
