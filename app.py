import os
import cv2
import string
import csv
import math
from flask import Flask, request
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def model_testing(uploaded_files):
    model = YOLO('best (1).pt')

    s = model(uploaded_files[0], conf=0.5)
    s_test = model(uploaded_files[1], conf=0.5)

    for idx, results in enumerate([s, s_test]):
        for img_result in results:
            annotated_image = img_result.orig_img.copy()
            for i in range(len(img_result.boxes)):
                x = img_result.boxes[i]
                cords = x.xyxy[0].tolist()
                y1, y2 = int(cords[1]), int(cords[3])
                x1, x2 = int(cords[0]), int(cords[2])
                height = y2 - y1

                x_text = f'X1:{x1}, X2:{x2}'
                cv2.putText(
                    annotated_image,
                    x_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    2,
                )

                height_text = f'H:{height}'
                cv2.putText(
                    annotated_image,
                    height_text,
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.savefig(f'static/results/saved_image{idx}.png')
            plt.close()


    quantities = [1, 2, 3, 4, 1, 2]

    for idx, results in enumerate([s, s_test]):
        detected_boxes = []
        for img_result in results:
            for i in range(len(img_result.boxes)):
                x = img_result.boxes[i]
                cords = x.xyxy[0].tolist()
                y1, y2 = int(cords[1]), int(cords[3])
                x1, x2 = int(cords[0]), int(cords[2])
                height = y2 - y1
                detected_boxes.append({
                    'x1': x1,
                    'x2': x2,
                    'dx': x2 - x1,
                    'y1': y1,
                    'y2': y2,
                    'height': height
                })

        sorted_boxes = sorted(detected_boxes, key=lambda box: box['x1'])

        for index, box in enumerate(sorted_boxes):
            box_name = string.ascii_uppercase[index]
            box['name'] = box_name
            box['quantity'] = quantities[index]

        csv_file_path = f'detected_boxes_{idx}.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            fieldnames = ['name', 'x1', 'x2', 'dx', 'y1', 'y2', 'height', 'quantity']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for box in sorted_boxes:
                writer.writerow(box)

        print(f"CSV file created successfully at: {csv_file_path}")

    existing_boxes = []
    with open('detected_boxes_0.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing_boxes.append(row)

    changed_boxes = []
    offsets_to_check = [0, 1, 2, 3, 4, 5, 6]

    for new_box in detected_boxes[1:]:
        for old_box in existing_boxes:
            old_x1 = int(old_box['x1'])
            new_x1 = int(new_box['x1'])
            old_height = int(old_box['height'])
            old_quantity = int(old_box['quantity'])
            new_quantity = int(new_box['quantity'])

            if any(new_x1 == old_x1 - offset or new_x1 == old_x1 + offset for offset in offsets_to_check) \
                    and abs(int(new_box['height']) - old_height) > 10:

                quantity_change = math.floor(float((int(old_height - new_box['height'])) / (old_height / old_quantity)))
                changed_quantity = old_quantity - quantity_change
                new_box['quantity'] = str(new_quantity + quantity_change)
                changed_boxes.append(old_box['name'])

        changed_medicines = list(set(changed_boxes))

    print("Medicines with significant height changes: ", changed_medicines)
    print("New Height: ", changed_quantity)
    return detected_boxes



def update_changed_quantities(detected_boxes):
    # ... (code for model_testing function remains unchanged)

    # This part remains the same as previously provided
    existing_boxes = []
    with open('detected_boxes_0.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing_boxes.append(row)

    changed_boxes = []
    offsets_to_check = [0, 1, 2, 3, 4, 5, 6]

    # ... (code for checking changed heights remains unchanged)

    # Code to update the quantities in the second CSV file
    updated_csv_file_path = 'detected_boxes_1_updated.csv'
    with open(updated_csv_file_path, mode='w', newline='') as file:
        fieldnames = ['name', 'x1', 'x2', 'dx', 'y1', 'y2', 'height', 'quantity']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through existing and changed boxes to update quantities
        for old_box in existing_boxes:
            for new_box in detected_boxes[1:]:
                if old_box['name'] == new_box['name']:
                    old_x1 = int(old_box['x1'])
                    new_x1 = int(new_box['x1'])
                    old_height = int(old_box['height'])
                    old_quantity = int(old_box['quantity'])
                    new_quantity = int(new_box['quantity'])

                    if any(new_x1 == old_x1 - offset or new_x1 == old_x1 + offset for offset in offsets_to_check) \
                            and abs(int(new_box['height']) - old_height) > 10:
                        quantity_change = math.floor(
                            float((int(old_height - new_box['height'])) / (old_height / old_quantity)))
                        changed_quantity = old_quantity - quantity_change
                        new_box['quantity'] = changed_quantity
                        changed_boxes.append(old_box['name'])

                        # Update the quantity in the CSV file
                        writer.writerow(new_box)

    print(f"Updated CSV file created successfully at: {updated_csv_file_path}")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_files = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        files = request.files.getlist('file')

        for file in files:
            if file.filename == '':
                return 'No selected file'

            file_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            file.save(file_path)
            uploaded_files.append(file_path.replace('\\', '/'))

        print("Uploaded:", uploaded_files)

    images_html = ''.join([f'<img src="{file}" alt="Uploaded Image" style="max-width: 500px; margin: 10px;">' for file in uploaded_files])

    if len(uploaded_files) >= 2:
        detected_boxes = model_testing(uploaded_files)
        update_changed_quantities(detected_boxes)

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Medicine.ai</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 50px;
                background-color: #f3f3f3;
            }}
            h1, form {{
                text-align: center;
                color: #004080;
                width: 50%;
                margin: 0 auto;
            }}
            form input[type=file] {{
                display: block;
                margin: 20px auto;
                padding: 10px;
                border-radius: 5px;
                background-color: #fff;
                color: #004080;
                border: 2px solid #004080;
                width: 80%;
            }}
            form input[type=submit] {{
                padding: 5px 10px;
                border: none;
                background-color: #004080;
                color: white;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 10px auto;
                cursor: pointer;
                border-radius: 5px;
                width: auto;
            }}
            form input[type=submit]:hover {{
                background-color: #00264d;
            }}
            .uploaded-images {{
                text-align: center;
                align-items: center;
            }}
            .uploaded-images img {{
                max-width: 500px;
                margin: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Medicine.ai</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*, video/*" multiple>
            <input type="submit" value="Upload">
        </form>
        <div class="uploaded-images">
            <h2>Uploaded Images:</h2>
            {images_html}
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)


