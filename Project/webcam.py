import torch
import torch.nn as nn
import joblib
import cv2
import neuralnet

# load label binarizer
lb = joblib.load(r'C:\Users\Shreya Basu\Workspace\ASL-Translator\Project\outputs\lb.pkl')

# load classification model
device = torch.device('cpu')
model = neuralnet.NeuralNet(nn.CrossEntropyLoss(), 0.001)
model.load_state_dict(torch.load(r'C:\Users\Shreya Basu\Workspace\ASL-Translator\Project\outputs\model.pth', map_location=device))
#print(model)
print('Model loaded')

# Returns a portion of webcam capture resized to 224x224
def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224,224))
    return hand

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Convert frame to proper shape and size
    cv2.rectangle(frame, (100,100), (324, 324), (20,34,225), 2)
    image = hand_area(frame)

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cpu()
    image = image.unsqueeze(0)
    
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)

    cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
