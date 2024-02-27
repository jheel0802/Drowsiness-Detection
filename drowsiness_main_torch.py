import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from torchvision import transforms
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pygame import mixer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)  # Adjusted for 3 classes

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return softmax(x, dim=1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data transformations
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((24, 24)),
                                transforms.ToTensor()])

# Create custom datasets
valid_dataset = CustomDataset(root_dir='data/valid', transform=transform)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

# Initialize the model
model = CNNModel().to(device)
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

# Initialize the sound mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Initialize face and eye cascades
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Initialize variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
path = 'H:\Drowsiness Detection\image.jpg'

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    # Detect left eyes and right eyes in the frame
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    # Draw rectangle for the alarm area
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(1, 1, 24, 24)
        r_eye_tensor = torch.from_numpy(r_eye).float().to(device)
        rpred = torch.argmax(model(r_eye_tensor), dim=-1).item()
        if rpred == 1:
            lbl = 'Open'
        if rpred == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(1, 1, 24, 24)
        l_eye_tensor = torch.from_numpy(l_eye).float().to(device)
        lpred = torch.argmax(model(l_eye_tensor), dim=-1).item()
        if lpred == 1:
            lbl = 'Open'
        if lpred == 0:
            lbl = 'Closed'
        break

    if rpred == 0 and lpred == 0:
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        # Person is feeling sleepy, so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()