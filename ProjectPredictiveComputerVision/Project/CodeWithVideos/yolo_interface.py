from ultralytics import YOLO

data_yaml_path = 'cs2.1.v3i.yolov8\data.yaml'
model = YOLO('models/test1.pt')

# Try loading the data before training
# try:
#     model.train(data=data_yaml_path, epochs=50, imgsz=640)
# except RuntimeError as e:
#     print("Error encountered during training:")
#     print(e)
#model.train(data='cs2.1.v3i.yolov8\data.yaml', epochs=50, imgsz=640)
results = model.val()  # Evaluate the model
results = model.predict(source='valorantgameplay.mp4',save=True)
#model.save('models/test1.pt')



#results = model.predict('Project/shortvalclip.mp4',save=True)

print(results[0])
print('++++++++++++++++++++')
for box in results[0].boxes:
    print(box)
