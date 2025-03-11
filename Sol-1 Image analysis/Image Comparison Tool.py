import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

# Reference image tester App.py

def load_image(image_path):
    """ loads the image from a file path """
    return cv2.imread(image_path)

def preprocess_image(image):
    """ converts the image to grayscale and blurs it slightly """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def find_differences(reference_image, test_image):
    """ finds differences between the reference and test images """
    # computes the absolute difference between the two images
    diff = cv2.absdiff(reference_image, test_image)
    # thresholds the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # dilates the threshold image to get the contours
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    # finds contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mark_differences(image, contours):
    """ marks the differences on the image """
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue  # skips small differences
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

def compare_images(reference_image_path, test_image_path):
    """ loads images, finds differences, and marks them """
    reference_image = load_image(reference_image_path)
    test_image = load_image(test_image_path)

    if reference_image.shape != test_image.shape:
        test_image = cv2.resize(test_image, (reference_image.shape[1], reference_image.shape[0]))

    reference_gray = preprocess_image(reference_image)
    test_gray = preprocess_image(test_image)

    contours = find_differences(reference_gray, test_gray)
    output_image = mark_differences(test_image.copy(), contours)

    return output_image, len(contours)

def resize_image(image, scale):
    """ resizes the image by a given scale """
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

def display_image(image):
    """ displays the output image in a Tkinter window """
    global img_tk, zoomed_image

    # Resize the image according to the current zoom level
    zoomed_image = resize_image(image, canvas_scale)

    # Convert the image to a format suitable for Tkinter
    image_rgb = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(image_pil)

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def choose_file(entry):
    """ opens a file chooser dialog and sets the chosen file path to the entry widget """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def choose_directory(entry):
    """ opens a directory chooser dialog and sets the chosen directory path to the entry widget """
    dir_path = filedialog.askdirectory()
    if dir_path:
        entry.delete(0, tk.END)
        entry.insert(0, dir_path)
        load_images_from_directory(dir_path)

def load_images_from_directory(directory):
    """ loads images from the specified directory """
    global image_paths, current_image_index
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_image_index = 0
    if image_paths:
        display_current_image()

def display_current_image():
    """ displays the current image with marked differences """
    global current_image_index, image_paths, reference_image_path, current_image, original_image
    if 0 <= current_image_index < len(image_paths):
        test_image_path = image_paths[current_image_index]
        try:
            current_image, num_differences = compare_images(reference_image_path, test_image_path)
            original_image = current_image.copy()
            display_image(current_image)
            differences_label.config(text=f"Differences detected: {num_differences}")
            image_label.config(text=f"Image {current_image_index + 1} of {len(image_paths)}")
            size_label.config(text=f"Image size: {current_image.shape[1]}x{current_image.shape[0]}")
        except cv2.error as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            display_image(load_image(test_image_path))
            differences_label.config(text="Differences detected: N/A")
            image_label.config(text=f"Image {current_image_index + 1} of {len(image_paths)}")
            size_label.config(text="Image size: N/A")

def next_image():
    """ displays the next image in the directory """
    global current_image_index
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        display_current_image()

def previous_image():
    """ displays the previous image in the directory """
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        display_current_image()

def compare_and_display():
    """ compares the reference image with the test images and displays the result """
    global reference_image_path, image_paths, current_image_index

    reference_image_path = reference_entry.get()
    test_image_dir = test_entry.get()
    single_test_image_path = single_test_image_entry.get()

    if not reference_image_path:
        messagebox.showerror("Error", "Please select a reference image.")
        return

    if mode_combobox.get() == "Single File":
        if single_test_image_path:
            image_paths = [single_test_image_path]
        else:
            messagebox.showerror("Error", "Please select a test image file.")
            return
    elif mode_combobox.get() == "Directory":
        if test_image_dir:
            load_images_from_directory(test_image_dir)
        else:
            messagebox.showerror("Error", "Please select a test image directory.")
            return

    current_image_index = 0
    display_current_image()

def update_mode(event):
    """ updates the visibility of the entry fields based on the selected mode """
    if mode_combobox.get() == "Single File":
        single_test_image_frame.grid()
        test_dir_frame.grid_remove()
    elif mode_combobox.get() == "Directory":
        test_dir_frame.grid()
        single_test_image_frame.grid_remove()

def capture_image(entry):
    """ opens a new window to capture an image from the webcam """
    capture_window = tk.Toplevel(app)
    capture_window.title("Capture Image")

    def capture_and_save():
        ret, frame = cap.read()
        if ret:
            cropped_frame = crop_image(frame)
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if file_path:
                cv2.imwrite(file_path, cropped_frame)
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
                cap.release()
                capture_window.destroy()
        else:
            messagebox.showerror("Error", "Failed to capture image from webcam.")

    def crop_image(image):
        """ crops the image to remove green background areas """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pcb = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(green_pcb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 150)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx_polygon)
            buffer = 10
            x = max(0, x - buffer)
            y = max(0, y - buffer)
            w += 2 * buffer
            h += 2 * buffer
            return image[y:y + h, x:x + w]
        else:
            return image

    def show_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
        camera_label.after(10, show_frame)

    cap = cv2.VideoCapture(int(camera_index_combobox.get()))
    camera_label = ttk.Label(capture_window)
    camera_label.pack()
    capture_button = ttk.Button(capture_window, text="Capture", command=capture_and_save)
    capture_button.pack()

    show_frame()

def on_mouse_wheel(event):
    """ zooms in or out on mouse wheel scroll """
    global canvas_scale, current_image
    scale_factor = 1.1 if event.delta > 0 else 0.9
    canvas_scale *= scale_factor
    display_image(current_image)

def on_mouse_drag(event):
    """ pans the image on mouse drag """
    global last_x, last_y
    dx = event.x - last_x
    dy = event.y - last_y
    canvas.move(tk.ALL, dx, dy)
    last_x = event.x
    last_y = event.y

def on_mouse_press(event):
    """ records the position on mouse press """
    global last_x, last_y
    last_x = event.x
    last_y = event.y

def zoom_in():
    """ zooms in on the image """
    global canvas_scale
    canvas_scale *= 1.1
    display_image(current_image)

def zoom_out():
    """ zooms out on the image """
    global canvas_scale
    canvas_scale /= 1.1
    display_image(current_image)

app = tk.Tk()
app.title("Image Comparison")

# Apply styles for a better look
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12))
style.configure('TEntry', font=('Helvetica', 12))
style.configure('TCombobox', font=('Helvetica', 12))
style.map('TCombobox', fieldbackground=[('readonly', 'white')], background=[('readonly', 'white')])

main_frame = ttk.Frame(app, padding="20 20 20 20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create and place the reference image frame
ref_frame = ttk.LabelFrame(main_frame, text="Reference Image", padding="10 10 10 10")
ref_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.EW)

reference_label = ttk.Label(ref_frame, text="Reference Image Path:")
reference_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
reference_entry = ttk.Entry(ref_frame, width=50)
reference_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
reference_button = ttk.Button(ref_frame, text="Browse", command=lambda: choose_file(reference_entry))
reference_button.grid(row=0, column=2, padx=5, pady=5)
reference_capture_button = ttk.Button(ref_frame, text="Capture", command=lambda: capture_image(reference_entry))
reference_capture_button.grid(row=0, column=3, padx=5, pady=5)

# Create and place the mode selection frame
mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding="10 10 10 10")
mode_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.EW)

mode_label = ttk.Label(mode_frame, text="Select Mode:")
mode_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
mode_combobox = ttk.Combobox(mode_frame, values=["Single File", "Directory"], state="readonly")
mode_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
mode_combobox.current(0)
mode_combobox.bind("<<ComboboxSelected>>", update_mode)

# Add camera selection frame
camera_selection_frame = ttk.LabelFrame(main_frame, text="Camera Selection", padding="10 10 10 10")
camera_selection_frame.grid(row=2, column=0, padx=10, pady=10, sticky=tk.EW)

camera_index_label = ttk.Label(camera_selection_frame, text="Camera Index:")
camera_index_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
camera_index_combobox = ttk.Combobox(camera_selection_frame, values=[0, 1, 2, 3], state="readonly")
camera_index_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
camera_index_combobox.current(0)

# Create and place the single test image frame
single_test_image_frame = ttk.LabelFrame(main_frame, text="Test Image", padding="10 10 10 10")
single_test_image_frame.grid(row=3, column=0, padx=10, pady=10, sticky=tk.EW)

test_label = ttk.Label(single_test_image_frame, text="Test Image Path:")
test_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
single_test_image_entry = ttk.Entry(single_test_image_frame, width=50)
single_test_image_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
single_test_image_button = ttk.Button(single_test_image_frame, text="Browse", command=lambda: choose_file(single_test_image_entry))
single_test_image_button.grid(row=0, column=2, padx=5, pady=5)
single_test_image_capture_button = ttk.Button(single_test_image_frame, text="Capture", command=lambda: capture_image(single_test_image_entry))
single_test_image_capture_button.grid(row=0, column=3, padx=5, pady=5)

# Create and place the test image directory frame
test_dir_frame = ttk.LabelFrame(main_frame, text="Test Image Directory", padding="10 10 10 10")
test_dir_frame.grid(row=4, column=0, padx=10, pady=10, sticky=tk.EW)
test_dir_frame.grid_remove()  # Hide initially

test_dir_label = ttk.Label(test_dir_frame, text="Test Image Directory:")
test_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
test_entry = ttk.Entry(test_dir_frame, width=50)
test_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
test_button = ttk.Button(test_dir_frame, text="Browse", command=lambda: choose_directory(test_entry))
test_button.grid(row=0, column=2, padx=5, pady=5)

# Create and place the compare button
compare_button = ttk.Button(main_frame, text="Compare", command=compare_and_display)
compare_button.grid(row=5, column=0, padx=10, pady=10)

# Create and place the output frame
output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10 10 10 10")
output_frame.grid(row=6, column=0, padx=10, pady=10, sticky=tk.NSEW)
output_frame.columnconfigure(0, weight=1)
output_frame.rowconfigure(0, weight=8)  # Increase weight for canvas
output_frame.rowconfigure(2, weight=1)  # Adjust this value if needed

# Create and configure the canvas
canvas = tk.Canvas(output_frame)
canvas.grid(row=0, column=0, sticky=tk.NSEW)
canvas.bind("<MouseWheel>", on_mouse_wheel)
canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_drag)

# Configure the canvas scrollbars
x_scroll = tk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=canvas.xview)
x_scroll.grid(row=1, column=0, sticky=tk.EW)
y_scroll = tk.Scrollbar(output_frame, orient=tk.VERTICAL, command=canvas.yview)
y_scroll.grid(row=0, column=1, sticky=tk.NS)
canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

# Set the minimum size for the output frame
output_frame.update_idletasks()
output_frame.grid_propagate(False)  # Disable resizing based on children's sizes
output_frame.config(width=800, height=490)  # Set a minimum size for the output frame

# Add zoom controls
zoom_frame = ttk.Frame(output_frame)
zoom_frame.grid(row=2, column=0, pady=10)

zoom_in_button = ttk.Button(zoom_frame, text="Zoom In", command=zoom_in)
zoom_in_button.grid(row=0, column=0, padx=5)

zoom_out_button = ttk.Button(zoom_frame, text="Zoom Out", command=zoom_out)
zoom_out_button.grid(row=0, column=1, padx=5)

# Create and place the navigation frame
navigation_frame = ttk.Frame(output_frame)
navigation_frame.grid(row=3, column=0, pady=10)

prev_button = ttk.Button(navigation_frame, text="Previous", command=previous_image)
prev_button.grid(row=0, column=0, padx=5)

next_button = ttk.Button(navigation_frame, text="Next", command=next_image)
next_button.grid(row=0, column=1, padx=5)

# Create and place the status frame
status_frame = ttk.Frame(main_frame)
status_frame.grid(row=7, column=0, padx=10, pady=10, sticky=tk.EW)

image_label = ttk.Label(status_frame, text="No images loaded")
image_label.grid(row=0, column=0, padx=5, pady=5)

differences_label = ttk.Label(status_frame, text="Differences detected: 0")
differences_label.grid(row=0, column=1, padx=5, pady=5)

size_label = ttk.Label(status_frame, text="Image size: N/A")
size_label.grid(row=0, column=2, padx=5, pady=5)

# Ensure that the entry fields expand to fill available space
main_frame.columnconfigure(0, weight=1)
ref_frame.columnconfigure(1, weight=1)
single_test_image_frame.columnconfigure(1, weight=1)
test_dir_frame.columnconfigure(1, weight=1)

# Initialize global variables
image_paths = []
current_image_index = 0
reference_image_path = ""
canvas_scale = 1.0
img_on_canvas = None
img_tk = None
last_x, last_y = 0, 0
current_image = None
original_image = None

app.mainloop()
