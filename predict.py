import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model("model/disease_prediction.h5")


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class_names = ['Acne and Rosacea', 'Actinic Keratosis','Atopic Dermatitis','Bullous Disease','Cellulitis Impetigo','Eczema','Exanthems',
               'Herpes HPV','Light disease','Lupus', 'Melanoma Skin Cancer','Poison Ivy','Psoriasis','Seborrheic','Systemic Disease', 
               'Tinea Ringworm', 'Urticaria Hives','Vascular Tumors','Vasculitis','Warts Molluscum']  

def predict_image(img_path):
    
    img = image.load_img(img_path, target_size=(128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    print(f"Prediction: {class_names[predicted_class]}")

if __name__ == "__main__":
    img_path = input("Enter path to image: ")
    predict_image(img_path)

