import cv2
import numpy as np

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read the first frame to initialize background subtraction
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return

    # Convert the frame to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convert the new frame to grayscale
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

        # Threshold the difference to highlight regions of motion
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in small holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore small contours to reduce noise
            if cv2.contourArea(contour) < 500:
                continue

            # Draw a bounding box around the detected motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Motion Detection", frame2)
        cv2.imshow("Threshold", thresh)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame
        frame1_gray = frame2_gray

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
