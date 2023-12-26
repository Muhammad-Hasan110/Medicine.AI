


from ultralytics import YOLO


model = YOLO('best (1).pt')



s=model('static/uploads/saved_image.png',conf=0.5)
s_test=model('static/uploads/saved_image1.png',conf=0.5)



import cv2



for img_result in s:
    annotated_image = img_result.orig_img.copy()  # Create a copy to avoid altering the original image
    for i in range(len(img_result.boxes)):
        x = img_result.boxes[i]
        cords = x.xyxy[0].tolist()
        y1, y2 = int(cords[1]), int(cords[3])
        x1, x2 = int(cords[0]), int(cords[2])

        height = y2 - y1  # Calculate the height of the box

        # Display x and y coordinates
        x_text = f'X1:{x1}, X2:{x2}'
        cv2.putText(
            annotated_image,
            x_text,
            (x1, y1 - 10),  # Position the text above the box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Overlay bounding box on the image
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),  # Box color (in BGR)
            2,  # Box thickness
        )

        # Display height below the box
        height_text = f'H:{height}'
        cv2.putText(
            annotated_image,
            height_text,
            (x1, y2 + 15),  # Position the text below the box
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            0.2,  # Font scale
            (0, 0, 0),  # Text color (in BGR)
            1,  # Text thickness
            cv2.LINE_AA,
        )


    import matplotlib.pyplot as plt


    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.savefig('saved_image.png')

    plt.show()




for img_result in s_test:
    annotated_image = img_result.orig_img.copy()  # Create a copy to avoid altering the original image
    for i in range(len(img_result.boxes)):
        x = img_result.boxes[i]
        cords = x.xyxy[0].tolist()
        y1, y2 = int(cords[1]), int(cords[3])
        x1, x2 = int(cords[0]), int(cords[2])

        height = y2 - y1  # Calculate the height of the box

        # Display x and y coordinates
        x_text = f'X1:{x1}, X2:{x2}'
        cv2.putText(
            annotated_image,
            x_text,
            (x1, y1 - 10),  # Position the text above the box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Overlay bounding box on the image
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),  # Box color (in BGR)
            2,  # Box thickness
        )

        # Display height below the box
        height_text = f'H:{height}'
        cv2.putText(
            annotated_image,
            height_text,
            (x1, y2 + 15),  # Position the text below the box
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            0.2,  # Font scale
            (0, 0, 0),  # Text color (in BGR)
            1,  # Text thickness
            cv2.LINE_AA,
        )




    # Use matplotlib for image display instead of cv2.imshow()
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.savefig('saved_image1.png')

    plt.show()

import string
import csv

# Initialize an empty list to store box details
detected_boxes = []
quantities = [1,2,3,4,1,2]

for img_result in s:
    for i in range(len(img_result.boxes)):
        x = img_result.boxes[i]
        cords = x.xyxy[0].tolist()
        y1, y2 = int(cords[1]), int(cords[3])
        x1, x2 = int(cords[0]), int(cords[2])

        height = y2 - y1  # Calculate the height of the box

        # Add box details to the list as a dictionary without assigning names yet
        detected_boxes.append({
            'x1': x1,
            'x2': x2,
            'dx': x2-x1,
            'y1': y1,
            'y2': y2,
            'height': height
        })

# Sort the detected boxes based on 'x1' in ascending order
sorted_boxes = sorted(detected_boxes, key=lambda box: box['x1'])

# Assign alphabetical name to each medicine box after sorting
for index, box in enumerate(sorted_boxes):
    box_name = string.ascii_uppercase[index]
    box['name'] = box_name
    box['quantity'] = quantities[index]



# Writing data to a CSV file
csv_file_path = 'detected_boxes.csv'
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['name', 'x1', 'x2','dx', 'y1', 'y2', 'height','quantity']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()  # Write header
    for box in sorted_boxes:
        writer.writerow(box)  # Write each box's details

print(f"CSV file created successfully at: {csv_file_path}")

import string
import csv

# Initialize an empty list to store box details
detected_boxes_test = []
quantities = [1,2,3,4,1,2]

for img_result in s_test:
    for i in range(len(img_result.boxes)):
        x = img_result.boxes[i]
        cords = x.xyxy[0].tolist()
        y1, y2 = int(cords[1]), int(cords[3])
        x1, x2 = int(cords[0]), int(cords[2])

        height = y2 - y1  # Calculate the height of the box

        # Add box details to the list as a dictionary without assigning names yet
        detected_boxes_test.append({
            'x1': x1,
            'x2': x2,
            'dx': x2-x1,
            'y1': y1,
            'y2': y2,
            'height': height
        })

# Sort the detected boxes based on 'x1' in ascending order
sorted_boxes = sorted(detected_boxes_test, key=lambda box: box['x1'])

# Assign alphabetical name to each medicine box after sorting
for index, box in enumerate(sorted_boxes):
    box_name = string.ascii_uppercase[index]
    box['name'] = box_name
    box['quantity'] = quantities[index]



# Writing data to a CSV file
csv_file_path = 'detected_boxes_1.csv'
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['name', 'x1', 'x2','dx', 'y1', 'y2', 'height','quantity']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()  # Write header
    for box in sorted_boxes:
        writer.writerow(box)  # Write each box's details

print(f"CSV file created successfully at: {csv_file_path}")


import csv
import math
# Load the existing CSV file
existing_boxes = []
with open('detected_boxes.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        existing_boxes.append(row)

# Iterate through the new boxes and compare with existing heights
changed_boxes = []
offsets_to_check = [0, 1, 2, 3, 4, 5, 6]  # Modify this list as per your desired offsets

for new_box in detected_boxes_test:  # Replace 'detected_boxes_test' with your new data
    for old_box in existing_boxes:
        old_x1 = int(old_box['x1'])
        new_x1 = int(new_box['x1'])
        old_height = int(old_box['height'])
        old_quantity = int(old_box['quantity'])
        new_quantity = int(new_box['quantity'])

        if any(new_x1 == old_x1 - offset or new_x1 == old_x1 + offset for offset in offsets_to_check) \
            and abs(int(new_box['height']) - old_height) > 10:

            # Calculate the quantity change based on height change

            quantity_change = math.floor(float((int(old_height-new_box['height']) ) / (old_height / old_quantity)))
            print(old_box['name'])
            print(quantity_change)
            print("UPDATED QUANTITY: ",  old_quantity- quantity_change)

            # Update the quantity in the new detection
            new_box['quantity'] = str(new_quantity + quantity_change)

            changed_boxes.append(old_box['name'])

        # Get unique names of medicines with height changes
        changed_medicines = list(set(changed_boxes))

print("Medicines with significant height changes: ", changed_medicines)

