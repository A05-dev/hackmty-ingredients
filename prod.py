from roboflow import Roboflow
rf = Roboflow(api_key="5fLOPWRg1OT4LIUuGTsX")
project = rf.workspace().project("sandoval_fridge")
model = project.version(3).model

# infer on a local image

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

prediction = model.predict("2.jpeg", confidence=40, overlap=30).json()
# print(type(prediction))
# print(prediction)

# Assuming you already have the `prediction` dictionary
unique_classes = set()

for item in prediction["predictions"]:
    unique_classes.add(item["class"])

# Convert the set to a list if needed
classes = list(unique_classes)

# Print the array of unique "class" elements
print(classes)