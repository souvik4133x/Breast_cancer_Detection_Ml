# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 01:44:12 2024

@author: souvi
"""

import numpy as np
import pickle 
import streamlit as st
loaded_model = pickle.load(open('C:/Users/souvi/OneDrive/Desktop/machine_learning/trained_model.sav', 'rb'))

def pre(input_data):
    
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The Breast cancer is Malignant'

    else:
      return 'The Breast Cancer is Benign'







def main():
    st.title('Breast Cancer Diagnosis Predictor')

    # Input fields for mean features
    mean_radius = st.number_input('Mean Radius')
    mean_texture = st.number_input('Mean Texture')
    mean_perimeter = st.number_input('Mean Perimeter')
    mean_area = st.number_input('Mean Area')
    mean_smoothness = st.number_input('Mean Smoothness')
    mean_compactness = st.number_input('Mean Compactness')
    mean_concavity = st.number_input('Mean Concavity')
    mean_concave_points = st.number_input('Mean Concave Points')
    mean_symmetry = st.number_input('Mean Symmetry')
    mean_fractal_dimension = st.number_input('Mean Fractal Dimension')

    # Input fields for standard error features
    radius_error = st.number_input('Radius Error')
    texture_error = st.number_input('Texture Error')
    perimeter_error = st.number_input('Perimeter Error')
    area_error = st.number_input('Area Error')
    smoothness_error = st.number_input('Smoothness Error')
    compactness_error = st.number_input('Compactness Error')
    concavity_error = st.number_input('Concavity Error')
    concave_points_error = st.number_input('Concave Points Error')
    symmetry_error = st.number_input('Symmetry Error')
    fractal_dimension_error = st.number_input('Fractal Dimension Error')

    # Input fields for worst features
    worst_radius = st.number_input('Worst Radius')
    worst_texture = st.number_input('Worst Texture')
    worst_perimeter = st.number_input('Worst Perimeter')
    worst_area = st.number_input('Worst Area')
    worst_smoothness = st.number_input('Worst Smoothness')
    worst_compactness = st.number_input('Worst Compactness')
    worst_concavity = st.number_input('Worst Concavity')
    worst_concave_points = st.number_input('Worst Concave Points')
    worst_symmetry = st.number_input('Worst Symmetry')
    worst_fractal_dimension = st.number_input('Worst Fractal Dimension')

    # code for Prediction
    diagnosis=''

    # creating a button for Prediction
    if st.button('Breast cancer Test Result'):
        input_data = [
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
            mean_fractal_dimension, radius_error, texture_error, perimeter_error,
            area_error, smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area,
            worst_smoothness, worst_compactness, worst_concavity,
            worst_concave_points, worst_symmetry, worst_fractal_dimension]
       
        diagnosis = pre(input_data)

    st.success(diagnosis)

   
if __name__ == '__main__':
    main()
   
   