import customtkinter
from PIL import Image
from win32api import GetSystemMetrics
import random
import threading
import ctypes


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue") 
root=customtkinter.CTk()
root.geometry("1200x800")
root.title("Python Paper Scissor")
#for window and taskbar icons
id='ppsv1'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(id)   
root.iconbitmap('resources/icon.ico')
# icon=customtkinter.CTkImage(Image.open('resources/icon.png'),size=(64,64))
# root.iconphoto(False,icon)

class PlayScreen:
    def __init__(self):
        background= customtkinter.CTkImage(Image.open('resources/pictp.png'),size=(1800,1100))
        backgroundLabel=customtkinter.CTkLabel(root,image=background,text="")
        backgroundLabel.place(relheight=1,relwidth=1)

        def onClickPlay():
            print('Started!')
            backgroundLabel.destroy()
            Playwindow()
            
        
            
        label1=customtkinter.CTkLabel(backgroundLabel,text="Python Paper Scissor",font=("Blogh Trial",100),text_color="white",bg_color="#fccb42",width=1200)
        label1.place(relx=0,rely=0.2)
        # startbtnImage= customtkinter.CTkImage(Image.open('C:/Users/aksho/OneDrive/Desktop/New folder/Start.png'),size=(500,250))
        button=customtkinter.CTkButton(master=backgroundLabel,text="PLAY",border_width=0,command=onClickPlay,bg_color="#1f538d",fg_color="#1f538d",width=220,height=60,corner_radius=0,text_color="white",font=("Blogh Trial",80),hover_color="#fccb42")
        button.place(relx=0.41,rely=0.5)
        
        
            
    
PlayScreen()

instruction="Decide your fate with nunmber of rounds, then, \nclick on ''Begin'' to fight the Legendary battle of Rock,Paper,Pencil,Scissor \nagainst the mighty AI :')."
noOfRuns=0
userChoice=None
compChoice=None
userScore=0
compScore=0
gameOver=True
final_result=None
flag=0

class Playwindow:
    
    
    
    def __init__(self):
        
        
        def onClickBack():
            contain.destroy()
            PlayScreen()
        
        def onClickExit():
            root.destroy()
        
        
        def updateRunCount3():
            global noOfRuns
            noOfRuns=3
            print(noOfRuns)
            startButton.configure(state="normal")
            userText.configure(text="You:")
            compText.configure(text="AI:")
            of1Btn.configure(state="disabled")
            of3Btn.configure(state="disabled")
            of5Btn.configure(state="disabled")
        
        def updateRunCount1():
            global noOfRuns
            noOfRuns=1
            print(noOfRuns)
            startButton.configure(state="normal")
            userText.configure(text="You:")
            compText.configure(text="AI:")
            of1Btn.configure(state="disabled")
            of3Btn.configure(state="disabled")
            of5Btn.configure(state="disabled")
            
        def updateRunCount5():
            global noOfRuns
            noOfRuns=5
            print(noOfRuns)
            startButton.configure(state="normal")
            userText.configure(text="You:")
            compText.configure(text="AI:")
            of1Btn.configure(state="disabled")
            of3Btn.configure(state="disabled")
            of5Btn.configure(state="disabled") 
            
        def onClickStart():
            vsText.configure(text="v/s")
            global userScore,compScore,userChoice,compChoice,final_result,gameOver,flag,noOfRuns
            print(gameOver)
            gameOver=False
            print(gameOver)
            
            startButton.configure(state="disabled")
            
            fun1()
            
            threading.Thread(target=fun2).start()
            
        def fun1():
            global userScore,compScore,userChoice,compChoice,final_result,gameOver,flag,noOfRuns
            if(gameOver==False and flag==0):
                userImage.configure(image=random.choice(imageList))
                # print('sup')
                compImage.configure(image=random.choice(imageList))
                userText.configure(text="You:"+str(userScore))
                compText.configure(text="AI:"+str(compScore))
                
                root.after(50,fun1)
            
            elif(flag==1 and gameOver==False):
                userImage.configure(image=userChoice)
                # print('Hello')
                compImage.configure(image=compChoice)
                userText.configure(text="You:"+str(userScore))
                compText.configure(text="AI:"+str(compScore))
                root.after(50,fun1)
                
            elif(gameOver==True):
                contain.destroy()
                userScore=0;
                compScore=0;
                homeButton=customtkinter.CTkButton(master=root,fg_color="white",text="Home",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickBack)
                homeButton.place(x=10,y=20)
                exitButton=customtkinter.CTkButton(master=root,fg_color="white",text="Exit",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickExit)
                exitButton.place(x=1050,y=20)
                global final_result
                if(final_result=="User"):
                    result="Wohoo!You won."
                elif(final_result=="Computer"):
                    result="Oops!You lost."
                else:
                    result=final_result
                    
                resultLabel=customtkinter.CTkLabel(root,text=result,text_color="white",font=("Blogh Trial",72))
                resultLabel.place(relx=0.28,y=300)
                
                
                
                    
                
            
            
            
        
            
        contain=customtkinter.CTkFrame(master=root,fg_color="transparent")
        contain.place(relheight=1,relwidth=1)
        backButton=customtkinter.CTkButton(master=contain,fg_color="white",text="Back",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickBack)
        backButton.place(x=10,y=20)
        exitButton=customtkinter.CTkButton(master=contain,fg_color="white",text="Exit",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickExit)
        exitButton.place(x=1050,y=20)
        instructionLabel=customtkinter.CTkLabel(master=contain,text="Instructions:",text_color="#00c0ff",font=("Blogh Trial",48),fg_color="white",corner_radius=10,bg_color="transparent")
        instructionLabel.place(relx=0.37,y=18)
        instructionText=customtkinter.CTkLabel(master=contain,text=instruction,text_color="#00c0ff",font=("Blogh Trial",30),fg_color="white",corner_radius=10,bg_color="transparent" )
        instructionText.place(relx=0.03,y=88)
        
        outofFrame=customtkinter.CTkFrame(contain,fg_color="white")
        outofFrame.columnconfigure(0)
        outofFrame.columnconfigure(1)
        outofFrame.columnconfigure(2)
        
        startButton=customtkinter.CTkButton(master=outofFrame,fg_color="black",text="Start",text_color="white",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickStart,state="disabled")
        startButton.grid(row=1,column=1,padx=80,pady=10)
        of1Btn=customtkinter.CTkButton(master=outofFrame,fg_color="black",text="Out Of 1",text_color="white",font=("Aachen",25),hover_color="#fccb42",command=updateRunCount1)
        of1Btn.grid(row=0,column=0,padx=80,pady=10)
        of3Btn=customtkinter.CTkButton(master=outofFrame,fg_color="black",text="Out Of 3",text_color="white",font=("Aachen",25),hover_color="#fccb42",command=updateRunCount3)
        of3Btn.grid(row=0,column=1,padx=80,pady=10)
        of5Btn=customtkinter.CTkButton(master=outofFrame,fg_color="black",text="Out Of 5",text_color="white",font=("Aachen",25),hover_color="#fccb42",command=updateRunCount5)
        of5Btn.grid(row=0,column=2,padx=80,pady=10)
        
        outofFrame.place(relx=0.13,y=210)
        # line1=customtkinter.CTkCanvas(contain,width=2000,height=10)
        # line1.place(x=190,y=400)
        # line2=customtkinter.CTkCanvas(contain,width=2000,height=10)
        # line2.place(x=190,y=673)
        
        
        imageFrame=customtkinter.CTkFrame(contain,fg_color="transparent")
        imageFrame.columnconfigure(0)
        imageFrame.columnconfigure(1)       
        
        rockImage= customtkinter.CTkImage(Image.open('resources/rock.png'),size=(250,250))
        paperImage= customtkinter.CTkImage(Image.open('resources/paper.png'),size=(250,250))
        pencilImage= customtkinter.CTkImage(Image.open('resources/pencil.png'),size=(250,250))
        scissorImage= customtkinter.CTkImage(Image.open('resources/scissor.png'),size=(250,250))
        imageList=[rockImage,paperImage,pencilImage,scissorImage]
        
        userText=customtkinter.CTkLabel(contain,text="",text_color="white",font=("Aachen",40))
        userText.place(x=182,y=370)
        vsText=customtkinter.CTkLabel(contain,text="",text_color="white",font=("Aachen",40))
        vsText.place(x=565,y=540)
        compText=customtkinter.CTkLabel(contain,text="",text_color="white",font=("Aachen",40))
        compText.place(x=900,y=370)
        compImage=customtkinter.CTkLabel(master=imageFrame,fg_color="transparent",text="",image=compChoice)
        compImage.grid(row=0,column=1,padx=310)
        userImage=customtkinter.CTkLabel(master=imageFrame,fg_color="transparent",text="",image=userChoice)
        userImage.grid(row=0,column=0,padx=120)
        
        
        imageFrame.place(relx=0.005,y=450)
        
        
        
        
       
        
        
        
            
    
    
        
        
        root.configure(fg_color="#00c0ff")
        
    
        def fun2():
            global userScore,compScore,userChoice,compChoice,final_result,gameOver,flag,noOfRuns,userImage,compImage
      


            WORKSPACE_PATH = 'Tensorflow/workspace'
            SCRIPTS_PATH = 'Tensorflow/scripts'
            APIMODEL_PATH = 'Tensorflow/models'
            ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
            IMAGE_PATH = WORKSPACE_PATH+'/images'
            MODEL_PATH = WORKSPACE_PATH+'/models'
            PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
            CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
            CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'




            labels = [{'name':'rock', 'id':1},
                      {'name':'paper', 'id':2},
                      {'name':'scissor', 'id':3},
                      {'name':'pencil', 'id':4},
                      {'name':'nothing', 'id':5},
                      
                     ]

            with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
                for label in labels:
                    f.write('item { \n')
                    f.write('\tname:\'{}\'\n'.format(label['name']))
                    f.write('\tid:{}\n'.format(label['id']))
                    f.write('}\n')


      


            get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
            get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}")




            CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 




           


            import tensorflow as tf
            from object_detection.utils import config_util
            from object_detection.protos import pipeline_pb2
            from google.protobuf import text_format


       


            CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'


       


            config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


            


            config


          


            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
                proto_str = f.read()                                                                                                                                                                                                                                          
                text_format.Merge(proto_str, pipeline_config)  


          


            pipeline_config.model.ssd.num_classes = 5
            pipeline_config.train_config.batch_size = 4
            pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
            pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']


            


            config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
            with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
                f.write(config_text)   


         

            


            import os
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as viz_utils
            from object_detection.builders import model_builder
            import random


            


            # Load pipeline config and build a detection model
            configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
            detection_model = model_builder.build(model_config=configs['model'], is_training=False)

            # Restore checkpoint
            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-51')).expect_partial()

            @tf.function
            def detect_fn(image):
                image, shapes = detection_model.preprocess(image)
                prediction_dict = detection_model.predict(image, shapes)
                detections = detection_model.postprocess(prediction_dict, shapes)
                return detections


            
            


            import cv2 
            import numpy as np
            import time


            


            category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')


            


          

            
                
            # global noOfRuns                
            outof=noOfRuns
            print("outof:",outof)
            # Since, opencv is taking a lot of time to launch for the first iteration(run), so, I am running the block four times and discarding the first(0th) output.

            i=0
            comp=0
            user=0
            # global final_result
            final_result=""
            while(i<=outof):
                # global flag 
                
                # userImage.configure(image=random.choice(imageList))
                # compImage.configure(image=random.choice(imageList))
                # userText.configure(text="You:"+str(userScore))
                # compText.configure(text="AI:"+str(compScore))
                
                # root.after(50,fun1)
                cap = cv2.VideoCapture(0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                
                user_action=[]
                t1=time.time()

                while(1): 
                    # userImage.configure(image=random.choice(imageList))
                    # compImage.configure(image=random.choice(imageList))
                    # userText.configure(text="You:"+str(userScore))
                    # compText.configure(text="AI:"+str(compScore))
                    # root.update()
                    # root.after(50,fun1)
                    ret, frame = cap.read()
                    # print('i:',i)
                    image_np = np.array(frame)
                    
                    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                    detections = detect_fn(input_tensor)
                   
                    
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                  for key, value in detections.items()}
                    detections['num_detections'] = num_detections
                
                    # detection_classes should be ints.
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                    
                
                    label_id_offset = 1
                    image_np_with_detections = image_np.copy()
                
                    finalLabel,image=viz_utils.visualize_boxes_and_labels_on_image_array(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes']+label_id_offset,
                                detections['detection_scores'],
                                category_index,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=5,
                                min_score_thresh=.5,
                                agnostic_mode=False)
                    
                    for val in finalLabel:
                        user_action.append(val)
                    
                    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
                    # print(detections)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        break
                    t2=time.time()
                    tinter=0
                    if(i==0):
                        tinter=1
                    else:
                        tinter=10
                    if(t2-t1>=tinter):
                        # global flag,gameOver,userScore,compScore
                        
                        
                        cap.release()
                        root.after(50,fun1)
                        cv2.destroyWindow('object detection')
                        
                        # userImage.configure(image=userChoice)
                        # compImage.configure(image=compChoice)
                        # userText.configure(text="You:"+str(userScore))
                        # compText.configure(text="AI:"+str(compScore))
                        
                        
                        
                        
                        break






                len(user_action)
                rockCount=0
                paperCount=0
                scissorCount=0
                pencilCount=0
                nothingCount=0
                
                
                for val in user_action:
                    if(val=="rock"):
                        rockCount+=1
                    elif(val=="paper"):
                        paperCount+=1
                    elif(val=="scissor"):
                        scissorCount+=1
                    elif(val=="pencil"):
                        pencilCount+=1
                    elif(val=="nothing"):
                        nothingCount+=1
                
                user_result=""
                
                if(rockCount>paperCount and rockCount>scissorCount and rockCount>pencilCount and rockCount>nothingCount):
                    user_result="rock"
                    # print("user_result",user_result)
                elif(paperCount>rockCount and paperCount>scissorCount and paperCount>pencilCount and paperCount>nothingCount):
                    user_result="paper"
                    # print("user_result",user_result)
                elif(scissorCount>paperCount and scissorCount>rockCount and scissorCount>pencilCount and scissorCount>nothingCount):
                    user_result="scissor"
                    # print("user_result",user_result)
                elif(pencilCount>paperCount and pencilCount>scissorCount and pencilCount>rockCount and pencilCount>nothingCount):
                    user_result="pencil"
                    # print("user_result",user_result)
                elif(nothingCount>paperCount and nothingCount>scissorCount and nothingCount>pencilCount and nothingCount>nothingCount):
                    user_result="nothing"
                    
                print("user_result:",user_result)
                
                
                
                
                import random
                
                labelHeadings=["rock","paper","scissor","pencil"]
                comp_result=random.choice(labelHeadings)
                
                
                
                result=""
                
                
                
                if(user_result=="rock"):
                    if(comp_result=="rock"):
                        result="Draw"
                    elif(comp_result=="paper"):
                        result="Computer"
                    elif(comp_result=="scissor"):
                        result="User"
                    elif(comp_result=="pencil"):
                        result="User"   
                        
                elif(user_result=="paper"):
                    if(comp_result=="rock"):
                        result="User"
                    elif(comp_result=="paper"):
                        result="Draw"
                    elif(comp_result=="scissor"):
                        result="Computer"
                    elif(comp_result=="pencil"):
                        result="Computer"    
                
                elif(user_result=="scissor"):
                    if(comp_result=="rock"):
                        result="Computer"
                    elif(comp_result=="paper"):
                        result="User"
                    elif(comp_result=="scissor"):
                        result="Draw"
                    elif(comp_result=="pencil"):
                        result="User"  
                
                elif(user_result=="pencil"):
                    if(comp_result=="rock"):
                        result="Computer"
                    elif(comp_result=="paper"):
                        result="User"
                    elif(comp_result=="scissor"):
                        result="Computer"
                    elif(comp_result=="pencil"):
                        result="Draw"   
                
                elif(user_result=="nothing" or user_result==""):
                    print('No move made.TRY AGAIN.')
                    if(i!=0):
                        outof+=1
                
                print("comp_result:",comp_result)
                

                if(i==0):
                    result=""
                # global userScore,compScore
                print("result:",result)
                if(result=="Computer"):
                    comp+=1
                    compScore=comp
                elif(result=="User"):
                    user+=1
                    userScore=user
                
                if(user_result=="rock"):
                    userChoice=rockImage
                elif(user_result=="paper"):
                    userChoice=paperImage
                elif(user_result=="pencil"):
                    userChoice=pencilImage
                elif(user_result=="scissor"):
                    userChoice=scissorImage
                
                if(comp_result=="rock"):
                    compChoice=rockImage
                elif(comp_result=="paper"):
                    compChoice=paperImage
                elif(comp_result=="pencil"):
                   compChoice=pencilImage
                elif(comp_result=="scissor"):
                    compChoice=scissorImage
                
                if(i!=0):
                    flag=1
                i+=1
                time.sleep(5)
                flag=0

            if(comp>user):
                final_result="Computer"
            elif(user>comp):
                final_result="User"
            else:
                final_result="Draw"

            print("\nfinal_result:",final_result)
            # global gameOver,userChoice,compChoice
            gameOver=True
            
            
            
                
                
            # if(gameOver==True):
            #     contain.destroy()
            #     backButton=customtkinter.CTkButton(master=root,fg_color="white",text="Home",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickBack)
            #     backButton.place(x=10,y=20)
            #     exitButton=customtkinter.CTkButton(master=root,fg_color="white",text="Exit",text_color="black",font=("Blogh Trial",30),hover_color="#fccb42",command=onClickExit)
            #     exitButton.place(x=1050,y=20)
            #     # global final_result
            #     if(final_result=="User"):
            #         result="Wohoo!You won."
            #     elif(final_result=="Computer"):
            #         result="Oops!You lost."
            #     else:
            #         result=final_result
                    
            #     resultLabel=customtkinter.CTkLabel(root,text=result,text_color="white",font=("Blogh Trial",72))
            #     resultLabel.place(relx=0.3,y=50)
                
                 
                
            







root.resizable(False,False)
root.mainloop()