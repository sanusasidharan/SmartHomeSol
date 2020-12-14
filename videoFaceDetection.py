import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('3.MP4')


while True:  
    # Read the frame  
    _,img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
  
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face  = gray[y:y+h,x:x+w]
        cv2.imwrite('facedetected.jpg',face)		
	# Display  
    cv2.imshow('Video', gray)
    # Stop if escape key is pressed  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break
          
# Release the VideoCapture object  
cap.release()    
