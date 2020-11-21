# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


start_time = time.time()

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class
def run_example():
#While loop selection for custom image database 
    while(True):
        
        sel=input("""
There are 20 sample image files.
*Note: Images are generic images saved from Google Images*
        
Which would you like to view?
        
    Choices are:
        1-2:   Airplane
        3-4:   Automobile
        5-6:   Bird
        7-8:   Cat
        9-10:  Deer
        11-12: Dog
        13-14: Frog
        15-16: Horse
        17-18: Ship
        19-20: Truck    
        
Type '1' through '20' for the files:""")
            
        if sel.isdigit():
            num=int(sel)
            if num in range(1, 21):
                break
            else:
                print("Incorrect input. Please enter a number '1' through '20' to select the sample files.")
        else:
            print("Incorrect input. Please enter a number '1' through '20' to select the sample files.")
            
    sel=int(sel)        
    if (sel==1):
        image_import='airplane_1_converted'
    if (sel==2):
        image_import='airplane_2_converted'    
    if (sel==3):
        image_import='automobile_1_converted'
    if (sel==4):
        image_import='automobile_2_converted'           
    if (sel==5):
        image_import='bird_1_converted'
    if (sel==6):
        image_import='bird_2_converted'    
    if (sel==7):
        image_import='cat_1_converted'
    if (sel==8):
        image_import='cat_2_converted'
    if (sel==9):
        image_import='deer_1_converted'
    if (sel==10):
        image_import='deer_2_converted'    
    if (sel==11):
        image_import='dog_1_converted'
    if (sel==12):
        image_import='dog_2_converted'           
    if (sel==13):
        image_import='frog_1_converted'
    if (sel==14):
        image_import='frog_2_converted'    
    if (sel==15):
        image_import='horse_1_converted'
    if (sel==16):
        image_import='horse_2_converted'
    if (sel==17):
        image_import='ship_1_converted'
    if (sel==18):
        image_import='ship_2_converted'    
    if (sel==19):
        image_import='truck_1_converted'
    if (sel==20):
        image_import='truck_2_converted'         
    

    #Load the image
    img = load_image(''+image_import+'.png')
    #Load model
    model = load_model('final_model.h5')
    #Predict the class
    result = model.predict_classes(img)
    #Add delay to print after loading console stuff
    plt.imshow(mpimg.imread(''+image_import+'.png'))
    #Probability of each class, multiply by 100 for percentage
    prediction_class=model.predict_proba(img)*100
    
    
    time.sleep(3)
    print("\n\n\nYou have chosen to sample image:",image_import,"\n\nPlease wait for the image prediction....\n")
    print("Image Prediction:")
    print("Airplane:",round(prediction_class[0][0],2),'%')
    print("Automobile:",round(prediction_class[0][1],2),'%')
    print("Bird:",round(prediction_class[0][2],2),'%')
    print("Cat:",round(prediction_class[0][3],2),'%')
    print("Deer:",round(prediction_class[0][4],2),'%')
    print("Dog:",round(prediction_class[0][5],2),'%')
    print("Frog:",round(prediction_class[0][6],2),'%')
    print("Horse:",round(prediction_class[0][7],2),'%')
    print("Ship:",round(prediction_class[0][8],2),'%')
    print("Truck:",round(prediction_class[0][9],2),'%')
    
    print("\nImage Classes from 0 to 9: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']")
    print("Image falls under class:",result[0],'\n')

    
    checker=int(result[0])
    
    if (checker==0):
        print("This image is an airplane.")
        
    if (checker==1):
        print("This image is an automobile.")
        
    if (checker==2):
        print("This image is a bird.")
        
    if (checker==3):
        print("This image is a cat.")
        
    if (checker==4):
        print("This image is a deer.")
        
    if (checker==5):
        print("This image is an a dog.")
        
    if (checker==6):
        print("This image is a frog.")
        
    if (checker==7):
        print("This image is a horse.")
        
    if (checker==8):
        print("This image is a ship.")
        
    if (checker==9):
        print("This image is a truck.")
        
        
run_example()

print("--- %s seconds ---" % (time.time() - start_time))